import time

import numpy as np
import pandas as pd
import yfinance as yf

_BATCH_SIZE = 100  # max tickers per yfinance call; larger batches overwhelm DNS


def _download_one_batch(
    tickers: list[str],
    period: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Download a single batch and return a DataFrame with one column per ticker."""
    kwargs = dict(auto_adjust=True, progress=False, threads=True)
    if start is not None:
        data = yf.download(tickers, start=start, end=end, **kwargs)
    else:
        data = yf.download(tickers, period=period, **kwargs)

    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"]
    prices = data[["Close"]]
    prices.columns = tickers
    return prices


def fetch_prices(
    tickers: list[str],
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download closing prices, batching requests to avoid DNS overload."""
    if len(tickers) <= _BATCH_SIZE:
        return _download_one_batch(tickers, period, start, end)

    batches = [tickers[i:i + _BATCH_SIZE] for i in range(0, len(tickers), _BATCH_SIZE)]
    frames = []
    for n, batch in enumerate(batches, 1):
        print(f"  Price batch {n}/{len(batches)} ({len(batch)} tickers)...", flush=True)
        df = _download_one_batch(batch, period, start, end)
        if not df.empty:
            frames.append(df)
        if n < len(batches):
            time.sleep(1)  # brief pause to avoid rate limiting

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


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
