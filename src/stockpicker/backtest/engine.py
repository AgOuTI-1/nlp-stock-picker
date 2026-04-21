from datetime import timedelta

import pandas as pd

from stockpicker.data.prices import compute_momentum_volatility, fetch_prices
from stockpicker.data.universe import (
    build_universe,
    get_all_historical_tickers,
    get_constituents_for_date,
    load_sp500_history,
)
from stockpicker.scoring.backtest_score import build_backtest_scores


def run_backtest(
    start_date: str = "2010-01-01",
    end_date: str = "2025-04-01",
    top_n: int = 30,
    benchmark: str = "SPY",
) -> dict:
    """Walk-forward monthly backtest using the 3 price-based factors.

    Uses historical S&P 500 constituent data so the correct universe is used
    at each rebalance date — no survivorship bias.

    On the first trading day of each month between start_date and end_date:
      - Looks up which stocks were actually in the S&P 500 on that date
      - Scores them using price history up to that date
      - Forms an equal-weighted portfolio of the top `top_n` stocks
      - Holds for one month and records the return

    Returns a dict with monthly_returns, cumulative returns, and stats for
    both the strategy and the benchmark.
    """
    # Load historical constituent data (no survivorship bias)
    history = load_sp500_history()

    # Get current universe for sector labels (best available free source)
    print("Fetching current S&P 500 sector labels...")
    _, _t2name, t2sector = build_universe()

    # Collect every ticker that ever appeared in the index across the backtest window.
    # 9-month buffer before start_date ensures enough price history for 6m momentum
    # on the very first rebalance date.
    buffer_start = (pd.Timestamp(start_date) - timedelta(days=275)).strftime("%Y-%m-%d")
    print("Collecting all historical tickers across backtest window...")
    sp500_tickers = get_all_historical_tickers(history, buffer_start, end_date)
    all_tickers = list(set(sp500_tickers + [benchmark]))

    print(f"Downloading prices for {len(all_tickers)} tickers from {buffer_start} to {end_date}...")
    print("(This covers all stocks ever in the S&P 500 during the period — may take 2–4 minutes)\n")
    all_prices = fetch_prices(all_tickers, start=buffer_start, end=end_date)

    trading_days = all_prices.index

    # Build rebalance dates: first calendar day of each month → nearest trading day
    rebal_calendar = pd.date_range(start=start_date, end=end_date, freq="MS")
    rebal_dates = []
    for d in rebal_calendar:
        idx = trading_days.searchsorted(d, side="left")
        idx = min(idx, len(trading_days) - 1)
        rebal_dates.append(trading_days[idx])

    print(f"Running {len(rebal_dates) - 1} monthly rebalances (top {top_n} stocks, equal weight)...")
    print("Factors: 3m momentum, 6m momentum, volatility")
    print("Note: sentiment and P/E excluded — no historical data available\n")

    portfolio_returns = []
    spy_returns = []
    period_labels = []

    for i in range(len(rebal_dates) - 1):
        t0 = rebal_dates[i]
        t1 = rebal_dates[i + 1]

        # Look up which stocks were actually in the S&P 500 on t0
        constituents = get_constituents_for_date(history, t0)
        available = [t for t in constituents if t in all_prices.columns and t != benchmark]

        if len(available) < top_n:
            continue

        # Score only the historically-correct universe as of t0
        price_subset = all_prices[available]
        factors = compute_momentum_volatility(price_subset, as_of=t0)
        if factors.empty or len(factors) < top_n:
            continue

        scores = build_backtest_scores(factors, t2sector)
        portfolio = scores.head(top_n).index.tolist()

        # Equal-weighted portfolio return from t0 close to t1 close
        stock_rets = []
        for ticker in portfolio:
            if ticker not in all_prices.columns:
                continue
            p0_series = all_prices[ticker].loc[:t0].dropna()
            p1_series = all_prices[ticker].loc[t1:].dropna()
            if p0_series.empty or p1_series.empty:
                continue
            p0, p1 = p0_series.iloc[-1], p1_series.iloc[0]
            if p0 == 0 or p0 != p0 or p1 != p1:
                continue
            stock_rets.append(p1 / p0 - 1)

        if not stock_rets:
            continue

        # SPY return for the same period
        spy_p0 = all_prices[benchmark].loc[:t0].dropna()
        spy_p1 = all_prices[benchmark].loc[t1:].dropna()
        if spy_p0.empty or spy_p1.empty:
            continue
        spy_ret = spy_p1.iloc[0] / spy_p0.iloc[-1] - 1

        portfolio_returns.append(sum(stock_rets) / len(stock_rets))
        spy_returns.append(spy_ret)
        period_labels.append(t0.strftime("%Y-%m"))

        if (i + 1) % 12 == 0:
            print(f"  {period_labels[-1]} — {i + 1}/{len(rebal_dates) - 1} months done")

    strategy_r = pd.Series(portfolio_returns, index=period_labels, name="strategy")
    spy_r = pd.Series(spy_returns, index=period_labels, name=benchmark)

    return {
        "monthly_returns": pd.DataFrame({"strategy": strategy_r, benchmark: spy_r}),
        "strategy_cumret": (1 + strategy_r).cumprod() - 1,
        "spy_cumret": (1 + spy_r).cumprod() - 1,
        "strategy_r": strategy_r,
        "spy_r": spy_r,
    }
