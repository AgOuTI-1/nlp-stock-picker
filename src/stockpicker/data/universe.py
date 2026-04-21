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


def _is_active(raw: str) -> bool:
    """Return True if a ticker has no removal-date suffix (e.g. TMC-200006)."""
    if "-" in raw:
        suffix = raw.split("-", 1)[1]
        if re.match(r"^\d{6,8}$", suffix):
            return False
    return True


def _clean(raw: str) -> str:
    """Convert CSV ticker format to yfinance format (dots → hyphens)."""
    return raw.replace(".", "-")


def load_sp500_history() -> pd.DataFrame:
    """Download the historical S&P 500 constituent CSV from GitHub.

    Returns a DataFrame with columns [date, tickers] sorted ascending by date.
    Each row is a daily snapshot of all index members on that trading day.
    """
    print("Downloading historical S&P 500 constituent data from GitHub...")
    resp = requests.get(_HISTORICAL_CSV_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_constituents_for_date(history: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    """Return the active S&P 500 ticker list for the closest snapshot <= as_of."""
    mask = history["date"] <= as_of
    if not mask.any():
        return []
    row = history[mask].iloc[-1]
    raw = [t.strip() for t in row["tickers"].split(",")]
    return [_clean(t) for t in raw if _is_active(t)]


def get_all_historical_tickers(
    history: pd.DataFrame,
    start: str,
    end: str,
) -> list[str]:
    """Union of every active ticker that appeared in the index between start and end."""
    mask = (history["date"] >= pd.Timestamp(start)) & (history["date"] <= pd.Timestamp(end))
    seen: set[str] = set()
    for _, row in history[mask].iterrows():
        raw = [t.strip() for t in row["tickers"].split(",")]
        seen.update(_clean(t) for t in raw if _is_active(t))
    return list(seen)
