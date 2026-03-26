import numpy as np
import pandas as pd
import yfinance as yf


def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"]
    prices = data[["Close"]]
    prices.columns = tickers
    return prices


def compute_momentum_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    records = {}
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        if len(s) < 130:
            continue

        mom_3m = s.iloc[-1] / s.iloc[-63] - 1
        mom_6m = s.iloc[-1] / s.iloc[-126] - 1
        vol = s.pct_change().dropna().tail(126).std() * np.sqrt(252)

        records[ticker] = {"mom_3m": mom_3m, "mom_6m": mom_6m, "volatility": vol}

    return pd.DataFrame.from_dict(records, orient="index")
