import numpy as np
import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers: list[str],
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if start is not None:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    else:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"]
    prices = data[["Close"]]
    prices.columns = tickers
    return prices


def compute_momentum_volatility(
    prices: pd.DataFrame,
    as_of: "pd.Timestamp | None" = None,
) -> pd.DataFrame:
    records = {}
    for ticker in prices.columns:
        s = prices[ticker]
        if as_of is not None:
            s = s.loc[:as_of]
        s = s.dropna()
        if len(s) < 130:
            continue

        mom_3m = s.iloc[-1] / s.iloc[-63] - 1
        mom_6m = s.iloc[-1] / s.iloc[-126] - 1
        vol = s.pct_change().dropna().tail(126).std() * np.sqrt(252)

        records[ticker] = {"mom_3m": mom_3m, "mom_6m": mom_6m, "volatility": vol}

    df = pd.DataFrame.from_dict(records, orient="index")

    # Drop stocks whose factor values are extreme outliers (> 4 std devs from
    # the cross-sectional mean). These almost always reflect M&A events,
    # reverse splits, or data errors rather than real tradeable returns.
    for col in df.columns:
        col_std = df[col].std()
        if col_std > 0:
            zscores = (df[col] - df[col].mean()).abs() / col_std
            df = df[zscores <= 4.0]

    return df
