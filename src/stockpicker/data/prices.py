import numpy as np
import pandas as pd
import yfinance as yf


def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Download adjusted close prices for all tickers in one batch call."""
    data = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    # yfinance returns MultiIndex columns when >1 ticker; single ticker gives flat
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers
    return prices


def compute_momentum_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker quant factors from price history.

    Factors:
        mom_3m     - 63-trading-day return
        mom_6m     - 126-trading-day return
        volatility - annualized 126-day realized vol (daily returns)
    """
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
