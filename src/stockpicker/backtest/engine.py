"""Walk-forward backtest engine — all 5 factors.

Factors used (identical to live pipeline):
  1. 3-month momentum
  2. 6-month momentum
  3. Volatility
  4. P/E ratio          ← historical, computed from quarterly EPS + prices
  5. News sentiment     ← historical, from GDELT (same concept as FinBERT)

Prices and historical P/E are downloaded in one shot and cached.
GDELT sentiment is fetched once per ticker and cached.
All caches are CSV files in the working directory — delete them to force a refresh.
"""

from datetime import timedelta

import pandas as pd

from stockpicker.backtest.pe_history import build_pe_history, get_pe_for_date
from stockpicker.backtest.sentiment_history import (
    build_sentiment_history,
    get_sentiment_for_date,
)
from stockpicker.data.prices import compute_momentum_volatility, fetch_prices
from stockpicker.data.universe import (
    build_universe,
    get_all_historical_tickers,
    get_constituents_for_date,
    load_sp500_history,
)
from stockpicker.scoring.backtest_score import build_backtest_scores


def run_backtest(
    start_date: str = "2015-01-01",
    end_date: str = "2025-04-01",
    top_n: int = 30,
    benchmark: str = "SPY",
) -> dict:
    """Run a fully-featured walk-forward backtest with all 5 factors.

    On the first run this takes ~30–60 minutes to pre-fetch and cache all data.
    Subsequent runs complete in ~5 minutes using the cached CSVs.

    Cache files (delete to force refresh):
      backtest_sentiment_cache.csv  — GDELT daily tone per ticker
      backtest_pe_cache.csv         — daily P/E per ticker
    """

    # ── Step 1: Universe ─────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1/5  Loading S&P 500 universe & sector labels")
    print("=" * 60)
    history = load_sp500_history()
    _, t2name, t2sector = build_universe()

    buffer_start = (pd.Timestamp(start_date) - timedelta(days=275)).strftime("%Y-%m-%d")
    sp500_tickers = get_all_historical_tickers(history, buffer_start, end_date)
    all_tickers = list(set(sp500_tickers + [benchmark]))
    print(f"  {len(sp500_tickers)} unique tickers across the backtest window.\n")

    # ── Step 2: Prices ───────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 2/5  Downloading historical prices (batched)")
    print("=" * 60)
    all_prices = fetch_prices(all_tickers, start=buffer_start, end=end_date)
    print(f"  Price data loaded: {all_prices.shape[1]} tickers × {len(all_prices)} days.\n")

    # ── Step 3: Historical P/E ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3/5  Building historical P/E (quarterly EPS from yfinance)")
    print("=" * 60)
    stock_tickers = [t for t in all_tickers if t != benchmark]
    pe_df = build_pe_history(stock_tickers, all_prices)
    print(f"  P/E data: {pe_df.shape[1]} tickers with historical earnings.\n")

    # ── Step 4: Historical Sentiment ─────────────────────────────────────────
    print("=" * 60)
    print("STEP 4/5  Building historical sentiment (GDELT)")
    print("=" * 60)
    sentiment_df = build_sentiment_history(
        tickers=stock_tickers,
        t2name=t2name,
        start=start_date,
        end=end_date,
    )
    print(f"  Sentiment data: {sentiment_df.shape[1]} tickers with GDELT tone.\n")

    # ── Step 5: Walk-forward loop ────────────────────────────────────────────
    print("=" * 60)
    print("STEP 5/5  Running walk-forward monthly rebalances")
    print("=" * 60)
    trading_days = all_prices.index
    rebal_calendar = pd.date_range(start=start_date, end=end_date, freq="MS")
    rebal_dates = []
    for d in rebal_calendar:
        idx = min(trading_days.searchsorted(d, side="left"), len(trading_days) - 1)
        rebal_dates.append(trading_days[idx])

    n_periods = len(rebal_dates) - 1
    print(f"  {n_periods} monthly rebalances | top {top_n} stocks | equal weight")
    print(f"  All 5 factors active: momentum (3m+6m) + volatility + P/E + sentiment")
    print(f"\n  [DIAG] all_prices shape: {all_prices.shape}")
    print(f"  [DIAG] all_prices sample columns: {list(all_prices.columns[:5])}")
    print(f"  [DIAG] all_prices index dtype: {all_prices.index.dtype}\n")

    portfolio_returns: list[float] = []
    spy_returns: list[float] = []
    period_labels: list[str] = []

    for i in range(n_periods):
        t0, t1 = rebal_dates[i], rebal_dates[i + 1]

        # Correct historical universe for this date
        constituents = get_constituents_for_date(history, t0)
        available = [t for t in constituents if t in all_prices.columns and t != benchmark]

        # First month diagnostics — tells us exactly where the loop breaks
        if i == 0:
            print(f"  [DIAG] Month {t0.date()}: constituents={len(constituents)}, "
                  f"available={len(available)}, need={top_n}")
            if constituents:
                print(f"  [DIAG] Sample constituents: {constituents[:5]}")
            if available:
                print(f"  [DIAG] Sample available:    {available[:5]}")
            else:
                sample_cols = list(all_prices.columns[:5])
                print(f"  [DIAG] No overlap — sample price cols: {sample_cols}")

        if len(available) < top_n:
            continue

        # Price factors
        price_subset = all_prices[available]
        factors = compute_momentum_volatility(price_subset, as_of=t0)

        if i == 0:
            print(f"  [DIAG] factors shape after momentum/vol filter: {factors.shape}")

        if factors.empty or len(factors) < top_n:
            continue

        # P/E for each scored stock as of t0
        pe_values = {t: get_pe_for_date(pe_df, t, t0) for t in factors.index}
        pe_series = pd.Series(pe_values, name="pe_ratio", dtype=float)

        # Sentiment for each scored stock in the 7 days before t0
        sent_values = {t: get_sentiment_for_date(sentiment_df, t, t0) for t in factors.index}
        sent_series = pd.Series(sent_values, name="sentiment", dtype=float)

        # Score with all 5 factors
        scores = build_backtest_scores(factors, pe_series, sent_series, t2sector)
        portfolio = scores.head(top_n).index.tolist()

        # Equal-weighted return t0 → t1
        stock_rets = []
        for ticker in portfolio:
            p0_s = all_prices[ticker].loc[:t0].dropna()
            p1_s = all_prices[ticker].loc[t1:].dropna()
            if p0_s.empty or p1_s.empty:
                continue
            p0, p1 = p0_s.iloc[-1], p1_s.iloc[0]
            if p0 > 0 and not pd.isna(p0) and not pd.isna(p1):
                stock_rets.append(p1 / p0 - 1)

        if not stock_rets:
            continue

        # SPY return for same period
        spy_p0 = all_prices[benchmark].loc[:t0].dropna()
        spy_p1 = all_prices[benchmark].loc[t1:].dropna()
        if spy_p0.empty or spy_p1.empty:
            continue
        spy_ret = spy_p1.iloc[0] / spy_p0.iloc[-1] - 1

        portfolio_returns.append(sum(stock_rets) / len(stock_rets))
        spy_returns.append(spy_ret)
        period_labels.append(t0.strftime("%Y-%m"))

        if (i + 1) % 12 == 0 or i == n_periods - 1:
            cumret = (pd.Series(portfolio_returns) + 1).prod() - 1
            print(f"  {period_labels[-1]}  ({i + 1}/{n_periods} months) "
                  f"cumulative return so far: {cumret:+.1%}", flush=True)

    strategy_r = pd.Series(portfolio_returns, index=period_labels, name="strategy")
    spy_r      = pd.Series(spy_returns,       index=period_labels, name=benchmark)

    return {
        "monthly_returns":  pd.DataFrame({"strategy": strategy_r, benchmark: spy_r}),
        "strategy_cumret":  (1 + strategy_r).cumprod() - 1,
        "spy_cumret":       (1 + spy_r).cumprod() - 1,
        "strategy_r":       strategy_r,
        "spy_r":            spy_r,
    }
