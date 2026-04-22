import io
import re

import pandas as pd
import requests

_HISTORICAL_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
)


def build_universe() -> tuple[list[str], dict[str, str], dict[str, str]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    df = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]

    # yfinance uses hyphens, NYSE uses dots (BRK.B -> BRK-B)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)

    tickers = df["Symbol"].tolist()
    t2name = dict(zip(df["Symbol"], df["Security"]))
    t2sector = dict(zip(df["Symbol"], df["GICS Sector"]))

    return tickers, t2name, t2sector


def _removal_date(raw: str) -> "pd.Timestamp | None":
    """If raw has a YYYYMM removal-date suffix, return that date; else None.

    Examples:
      'SE-201702'  → Timestamp('2017-02-01')   (removed Feb 2017)
      'AAPL'       → None                       (no removal date)
      'BRK.B'      → None                       (dot is not a date suffix)
    """
    if "-" in raw:
        suffix = raw.split("-", 1)[1]
        if re.match(r"^\d{6,8}$", suffix):
            year, month = int(suffix[:4]), int(suffix[4:6])
            return pd.Timestamp(year, month, 1)
    return None


def _base_ticker(raw: str) -> str:
    """Strip removal-date suffix and convert dots to hyphens for yfinance.

    'SE-201702' → 'SE'   'BRK.B' → 'BRK-B'   'AAPL' → 'AAPL'
    """
    rd = _removal_date(raw)
    if rd is not None:
        base = raw.rsplit("-", 1)[0]
    else:
        base = raw
    return base.replace(".", "-")


def load_sp500_history() -> pd.DataFrame:
    """Download the historical S&P 500 constituent CSV from GitHub.

    Returns a DataFrame with columns [date, tickers] sorted ascending by date.
    Each row represents a change event; the tickers column contains all members
    active on that date.  Removed members are annotated with a YYYYMM suffix
    indicating when they left the index (e.g. 'SE-201702').
    """
    print("Downloading historical S&P 500 constituent data from GitHub...")
    resp = requests.get(_HISTORICAL_CSV_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_constituents_for_date(history: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    """Return the active S&P 500 ticker list for the closest snapshot <= as_of.

    A ticker is active on as_of if:
      • it has no removal-date suffix (still in index as of the CSV's last update), OR
      • its removal date is strictly after as_of (it was removed later).
    """
    mask = history["date"] <= as_of
    if not mask.any():
        return []
    row = history[mask].iloc[-1]
    raw = [t.strip() for t in row["tickers"].split(",")]
    result = []
    for t in raw:
        rd = _removal_date(t)
        if rd is None:
            result.append(_base_ticker(t))       # currently active, no removal date
        elif rd > as_of:
            result.append(_base_ticker(t))       # removed after as_of → was active
        # else: already removed before as_of, skip
    return result


def get_all_historical_tickers(
    history: pd.DataFrame,
    start: str,
    end: str,
) -> list[str]:
    """Union of every ticker that was active at any point between start and end.

    Includes tickers with no removal date (currently active) and tickers whose
    removal date falls after start (they were in the index for at least part of
    the requested window).
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    mask = (history["date"] >= start_ts) & (history["date"] <= end_ts)
    seen: set[str] = set()
    for _, row in history[mask].iterrows():
        raw = [t.strip() for t in row["tickers"].split(",")]
        for t in raw:
            rd = _removal_date(t)
            # Include if no removal date, or removed after the window start
            if rd is None or rd > start_ts:
                seen.add(_base_ticker(t))
    return list(seen)
