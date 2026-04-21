"""Historical news sentiment via the GDELT Project API.

GDELT (gdeltproject.org) provides free historical news tone scores going back
to 2015. The tone scale is negative (bad news) to positive (good news),
conceptually equivalent to what FinBERT measures from headlines. Since all
sentiment values are z-scored before use, the absolute scale doesn't matter.

One API call per ticker fetches the full historical time series. Results are
cached to a CSV file so the slow fetch only happens once.
"""

import os
import time

import pandas as pd
import requests

_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_DELAY = 0.5  # seconds between calls — be polite to GDELT


def _fetch_one(company_name: str, start: str, end: str) -> pd.Series:
    """Fetch daily average tone from GDELT for one company over a date range.

    Args:
        company_name: e.g. "Apple" or "Apple Inc"
        start / end: "YYYY-MM-DD"
    Returns:
        pd.Series indexed by date, values are average tone (float).
        Empty series on failure.
    """
    params = {
        "query": f'"{company_name}" stock',
        "mode": "timelinetone",
        "STARTDATETIME": start.replace("-", "") + "000000",
        "ENDDATETIME": end.replace("-", "") + "000000",
        "format": "json",
    }
    try:
        resp = requests.get(_GDELT_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        timeline = payload.get("timeline", [])
        if not timeline:
            return pd.Series(dtype=float)
        records = {}
        for point in timeline[0].get("data", []):
            records[pd.Timestamp(point["date"])] = float(point["value"])
        return pd.Series(records).sort_index()
    except Exception:
        return pd.Series(dtype=float)


def build_sentiment_history(
    tickers: list[str],
    t2name: dict[str, str],
    start: str,
    end: str,
    cache_path: str = "backtest_sentiment_cache.csv",
) -> pd.DataFrame:
    """Return a (date × ticker) DataFrame of daily GDELT tone scores.

    On the first run, fetches data from GDELT and saves to cache_path.
    On subsequent runs, loads from cache instantly.
    """
    if os.path.exists(cache_path):
        print(f"  Loading sentiment cache from {cache_path}...", flush=True)
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"  Fetching GDELT sentiment for {len(tickers)} companies ({start} → {end}).")
    print("  One-time operation — cached after this run.\n", flush=True)

    series_dict: dict[str, pd.Series] = {}
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"    {i}/{len(tickers)} tickers...", flush=True)
        name = t2name.get(ticker, ticker)
        series_dict[ticker] = _fetch_one(name, start, end)
        time.sleep(_DELAY)

    df = pd.DataFrame(series_dict)
    df.to_csv(cache_path)
    print(f"  Sentiment cache saved → {cache_path}", flush=True)
    return df


def get_sentiment_for_date(
    sentiment_df: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    lookback_days: int = 7,
) -> float:
    """Return the mean GDELT tone for a ticker in the 7 days ending on as_of."""
    if ticker not in sentiment_df.columns:
        return 0.0
    window = sentiment_df[ticker].loc[as_of - pd.Timedelta(days=lookback_days): as_of].dropna()
    return float(window.mean()) if not window.empty else 0.0
