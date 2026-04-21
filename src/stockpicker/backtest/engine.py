from datetime import timedelta

import pandas as pd

from stockpicker.data.prices import compute_momentum_volatility, fetch_prices
from stockpicker.data.universe import build_universe
from stockpicker.scoring.backtest_score import build_backtest_scores


def run_backtest(
    start_date: str = "2021-01-01",
    end_date: str = "2025-01-01",
    top_n: int = 30,
    benchmark: str = "SPY",
) -> dict:
    """Walk-forward monthly backtest using the 3 price-based factors.

    On the first trading day of each month between start_date and end_date:
      - Scores all S&P 500 stocks using price history up to that date
      - Forms an equal-weighted portfolio of the top `top_n` stocks
      - Holds for one month and records the return

    Returns a dict with monthly_returns, cumulative returns, and stats for
    both the strategy and the benchmark.
    """
    print("Building S&P 500 universe...")
    tickers, _t2name, t2sector = build_universe()

    # Download all prices in one bulk call — 9-month buffer before start_date
    # ensures enough history for 6m momentum on the very first rebalance.
    buffer_start = (pd.Timestamp(start_date) - timedelta(days=275)).strftime("%Y-%m-%d")
    all_tickers = list(set(tickers + [benchmark]))

    print(f"Downloading prices from {buffer_start} to {end_date} (single bulk call)...")
    all_prices = fetch_prices(all_tickers, start=buffer_start, end=end_date)

    trading_days = all_prices.index

    # Build rebalance dates: first calendar day of each month, mapped to nearest trading day
    rebal_calendar = pd.date_range(start=start_date, end=end_date, freq="MS")
    rebal_dates = []
    for d in rebal_calendar:
        idx = trading_days.searchsorted(d, side="left")
        idx = min(idx, len(trading_days) - 1)
        rebal_dates.append(trading_days[idx])

    print(f"Running {len(rebal_dates) - 1} monthly rebalances (top {top_n} stocks, equal weight)...")
    print("Note: factors used are 3m momentum, 6m momentum, volatility only.")
    print("      Sentiment and P/E cannot be backtested (no historical data available).\n")

    portfolio_returns = []
    spy_returns = []
    period_labels = []

    for i in range(len(rebal_dates) - 1):
        t0 = rebal_dates[i]
        t1 = rebal_dates[i + 1]

        # Score stocks as of t0
        factors = compute_momentum_volatility(all_prices.drop(columns=[benchmark], errors="ignore"), as_of=t0)
        if factors.empty:
            continue

        scores = build_backtest_scores(factors, t2sector)
        portfolio = scores.head(top_n).index.tolist()

        # Compute equal-weighted portfolio return from t0 to t1
        stock_rets = []
        for ticker in portfolio:
            if ticker not in all_prices.columns:
                continue
            p0 = all_prices[ticker].loc[:t0].iloc[-1]
            p1_series = all_prices[ticker].loc[t1:]
            if p0 == 0 or p0 != p0 or p1_series.empty:
                continue
            p1 = p1_series.iloc[0]
            if p1 != p1:
                continue
            stock_rets.append(p1 / p0 - 1)

        if not stock_rets:
            continue

        port_ret = sum(stock_rets) / len(stock_rets)

        # SPY return for the same period
        spy_p0 = all_prices[benchmark].loc[:t0].iloc[-1]
        spy_p1_series = all_prices[benchmark].loc[t1:]
        if spy_p1_series.empty or spy_p0 == 0:
            continue
        spy_ret = spy_p1_series.iloc[0] / spy_p0 - 1

        portfolio_returns.append(port_ret)
        spy_returns.append(spy_ret)
        period_labels.append(t0.strftime("%Y-%m"))

    strategy_r = pd.Series(portfolio_returns, index=period_labels, name="strategy")
    spy_r = pd.Series(spy_returns, index=period_labels, name=benchmark)

    monthly_returns = pd.DataFrame({"strategy": strategy_r, benchmark: spy_r})

    strategy_cumret = (1 + strategy_r).cumprod() - 1
    spy_cumret = (1 + spy_r).cumprod() - 1

    return {
        "monthly_returns": monthly_returns,
        "strategy_cumret": strategy_cumret,
        "spy_cumret": spy_cumret,
        "strategy_r": strategy_r,
        "spy_r": spy_r,
    }
