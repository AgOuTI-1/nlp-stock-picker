"""Historical news sentiment via the GDELT Project API.

GDELT (gdeltproject.org) provides free historical news tone scores going back
to 2015. The tone scale is negative (bad news) to positive (good news),
conceptually equivalent to what FinBERT measures from headlines. Since all
sentiment values are z-scored before use, the absolute scale doesn't matter.

One API call per ticker fetches the full historical time series. Requests run
in parallel (10 workers) with a short timeout so a single slow call never
blocks the whole batch. Results are cached to CSV — the fetch only runs once.
"""

import concurrent.futures
import os

import pandas as pd
import requests

_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT   = 10   # seconds per request — bail fast if GDELT is slow
_WORKERS   = 10   # parallel connections


def _fetch_one(args: tuple[str, str, str, str]) -> tuple[str, pd.Series]:
    """Fetch daily average tone from GDELT for one ticker.

    Args:
        args: (ticker, company_name, start "YYYY-MM-DD", end "YYYY-MM-DD")
    Returns:
        (ticker, pd.Series indexed by date) — empty series on any failure.
    """
    ticker, company_name, start, end = args
    params = {
        "query": f'"{company_name}" stock',
        "mode": "timelinetone",
        "STARTDATETIME": start.replace("-", "") + "000000",
        "ENDDATETIME":   end.replace("-", "") + "000000",
        "format": "json",
    }
    try:
        resp = requests.get(_GDELT_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        payload  = resp.json()
        timeline = payload.get("timeline", [])
        if not timeline:
            return ticker, pd.Series(dtype=float)
        records = {
            pd.Timestamp(pt["date"]): float(pt["value"])
            for pt in timeline[0].get("data", [])
        }
        return ticker, pd.Series(records).sort_index()
    except Exception:
        return ticker, pd.Series(dtype=float)


def build_sentiment_history(
    tickers: list[str],
    t2name:  dict[str, str],
    start:   str,
    end:     str,
    cache_path: str = "backtest_sentiment_cache.csv",
) -> pd.DataFrame:
    """Return a (date × ticker) DataFrame of daily GDELT tone scores.

    On the first run, fetches data from GDELT in parallel and saves to
    cache_path. Subsequent runs load from cache instantly.
    """
    if os.path.exists(cache_path):
        print(f"  Loading sentiment cache from {cache_path}...", flush=True)
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"  Fetching GDELT sentiment for {len(tickers)} tickers "
          f"({start} → {end}) using {_WORKERS} parallel workers...")
    print("  One-time operation — cached after this run.\n", flush=True)

    tasks = [(t, t2name.get(t, t), start, end) for t in tickers]
    series_dict: dict[str, pd.Series] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, task): task[0] for task in tasks}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            ticker, series = future.result()
            series_dict[ticker] = series
            done += 1
            if done % 50 == 0 or done == len(tickers):
                print(f"    {done}/{len(tickers)} tickers done...", flush=True)

    df = pd.DataFrame(series_dict)
    df.to_csv(cache_path)
    print(f"  Sentiment cache saved → {cache_path}", flush=True)
    return df


def get_sentiment_for_date(
    sentiment_df: pd.DataFrame,
    ticker:       str,
    as_of:        pd.Timestamp,
    lookback_days: int = 7,
) -> float:
    """Return mean GDELT tone for a ticker in the 7 days ending on as_of."""
    if ticker not in sentiment_df.columns:
        return 0.0
    window = sentiment_df[ticker].loc[
        as_of - pd.Timedelta(days=lookback_days): as_of
    ].dropna()
    return float(window.mean()) if not window.empty else 0.0
