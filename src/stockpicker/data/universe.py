import io

import pandas as pd
import requests


def build_universe() -> tuple[list[str], dict[str, str]]:
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
