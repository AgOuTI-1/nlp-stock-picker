"""Historical P/E ratios computed from yfinance quarterly earnings + prices.

For each ticker we fetch quarterly EPS going as far back as yfinance provides
(typically 3–5 years). For each trading day we compute:
    TTM EPS = sum of last 4 quarters of EPS reported BEFORE that date
    P/E     = closing price / TTM EPS

Results are cached to a CSV file so the slow fetch only happens once.
Tickers with no earnings data or negative TTM EPS default to NaN, and
build_backtest_scores() fills those in with the sector-median P/E — the same
fallback used in the live scoring pipeline.
"""

import concurrent.futures
import os

import pandas as pd
import yfinance as yf


def _fetch_quarterly_eps(ticker: str) -> tuple[str, pd.Series | None]:
    """Return (ticker, quarterly EPS Series) or (ticker, None) on failure."""
    try:
        t = yf.Ticker(ticker)
        income = t.quarterly_income_stmt
        if income is None or income.empty:
            return ticker, None

        # Try EPS rows first (fastest path)
        for row in ("Diluted EPS", "Basic EPS"):
            if row in income.index:
                s = income.loc[row].dropna()
                if not s.empty:
                    return ticker, s.sort_index()

        # Fallback: Net Income ÷ share count
        if "Net Income" not in income.index:
            return ticker, None
        net_income = income.loc["Net Income"].dropna()

        shares = None
        for row in ("Diluted Average Shares", "Basic Average Shares", "Ordinary Shares Number"):
            if row in income.index:
                shares = income.loc[row].dropna()
                break

        if shares is None or shares.empty:
            return ticker, None

        eps = (net_income / shares).dropna().sort_index()
        return ticker, eps if not eps.empty else None

    except Exception:
        return ticker, None


def build_pe_history(
    tickers: list[str],
    prices: pd.DataFrame,
    cache_path: str = "backtest_pe_cache.csv",
    max_workers: int = 20,
) -> pd.DataFrame:
    """Return a (date × ticker) DataFrame of daily P/E ratios.

    On the first run, fetches quarterly EPS from yfinance, computes daily P/E,
    and saves to cache_path. Subsequent runs load from cache instantly.
    """
    if os.path.exists(cache_path):
        print(f"  Loading P/E cache from {cache_path}...", flush=True)
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"  Fetching quarterly earnings for {len(tickers)} tickers (parallel)...")
    print("  One-time operation — cached after this run.\n", flush=True)

    eps_map: dict[str, pd.Series] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_quarterly_eps, t): t for t in tickers}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            if i % 50 == 0:
                print(f"    {i}/{len(tickers)} tickers...", flush=True)
            ticker, eps = fut.result()
            if eps is not None:
                eps_map[ticker] = eps

    print(f"  Got EPS data for {len(eps_map)}/{len(tickers)} tickers.")
    print("  Computing daily P/E ratios...", flush=True)

    pe_cols: dict[str, pd.Series] = {}
    for ticker, eps_series in eps_map.items():
        if ticker not in prices.columns:
            continue
        price_series = prices[ticker].dropna()
        pe_by_date: dict[pd.Timestamp, float] = {}

        for date in price_series.index:
            past = eps_series[eps_series.index < date]
            if len(past) < 4:
                continue
            ttm_eps = past.iloc[-4:].sum()
            if ttm_eps <= 0:
                continue
            price = price_series.loc[date]
            if pd.isna(price) or price <= 0:
                continue
            pe = price / ttm_eps
            if 0 < pe < 2000:   # filter absurd values
                pe_by_date[date] = pe

        if pe_by_date:
            pe_cols[ticker] = pd.Series(pe_by_date)

    df = pd.DataFrame(pe_cols)
    df.to_csv(cache_path)
    print(f"  P/E cache saved → {cache_path}", flush=True)
    return df


def get_pe_for_date(
    pe_df: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
) -> float | None:
    """Return the most recent P/E for a ticker as of as_of, or None."""
    if ticker not in pe_df.columns:
        return None
    past = pe_df[ticker].loc[:as_of].dropna()
    return float(past.iloc[-1]) if not past.empty else None
