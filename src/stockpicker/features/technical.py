from __future__ import annotations

import pandas as pd


def compute_features(adj_close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute daily returns, 3M/6M momentum, and 3M rolling volatility."""
    rets = adj_close.pct_change(fill_method=None)
    mom_3m = adj_close.pct_change(63, fill_method=None)
    mom_6m = adj_close.pct_change(126, fill_method=None)
    vol_3m = rets.rolling(63).std()
    return {"rets": rets, "mom_3m": mom_3m, "mom_6m": mom_6m, "vol_3m": vol_3m}


def build_monthly_dates(trading_index: pd.DatetimeIndex, start: str) -> pd.Series:
    monthly = pd.Series(trading_index).groupby(trading_index.to_period("M")).last()
    monthly = monthly[monthly >= pd.Timestamp(start)]
    return monthly.sort_values()


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp("M")
