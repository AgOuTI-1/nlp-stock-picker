from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


def make_equal_weights(tickers: List[str]) -> Dict[str, float]:
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def make_inv_vol_weights(
    tickers: List[str],
    asof_date: pd.Timestamp,
    vol_3m: pd.DataFrame,
    max_weight: float = 0.10,
) -> Dict[str, float]:
    if not tickers:
        return {}

    vols = vol_3m.loc[asof_date, tickers].astype(float)
    vols = vols.replace([0.0, np.inf, -np.inf], np.nan).dropna()
    if len(vols) < 1:
        return make_equal_weights(tickers)

    inv = 1.0 / (vols + 1e-12)
    w = (inv / inv.sum()).to_dict()

    w = {k: min(float(v), max_weight) for k, v in w.items()}
    s = sum(w.values())
    if s <= 0:
        return make_equal_weights(tickers)

    w = {k: v / s for k, v in w.items()}
    for t in tickers:
        w.setdefault(t, 0.0)
    return w
