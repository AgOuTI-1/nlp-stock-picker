import pandas as pd


def build_universe() -> tuple[list[str], dict[str, str]]:
    """
    Fetch the current S&P 500 constituents from Wikipedia.

    Returns:
        tickers: list of ticker symbols (yfinance-compatible, dots replaced with dashes)
        t2name:  dict mapping ticker -> company name
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url, attrs={"id": "constituents"})[0]

    # yfinance uses hyphens where the NYSE uses dots (e.g. BRK.B -> BRK-B)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)

    tickers = df["Symbol"].tolist()
    t2name = dict(zip(df["Symbol"], df["Security"]))

    return tickers, t2name
