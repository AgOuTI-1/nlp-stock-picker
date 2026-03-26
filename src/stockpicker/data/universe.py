import pandas as pd


def build_universe() -> tuple[list[str], dict[str, str]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url, attrs={"id": "constituents"})[0]

    # yfinance uses hyphens, NYSE uses dots (BRK.B -> BRK-B)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)

    tickers = df["Symbol"].tolist()
    t2name = dict(zip(df["Symbol"], df["Security"]))

    return tickers, t2name
