from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


def compute_turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w) | set(new_w)
    return float(sum(abs(new_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in names))


def make_trade_blotter(
    asof_date: pd.Timestamp,
    portfolio_value: float,
    prev_weights: Dict[str, float],
    target_weights: Dict[str, float],
    price_df: pd.DataFrame,
    allow_fractional: bool = True,
) -> pd.DataFrame:
    tickers = sorted(set(prev_weights) | set(target_weights))
    px = price_df.loc[asof_date, tickers].astype(float)

    rows = []
    for t in tickers:
        p = float(px.get(t, np.nan))
        if not np.isfinite(p) or p <= 0:
            continue

        prev_notional = float(prev_weights.get(t, 0.0) * portfolio_value)
        tgt_notional = float(target_weights.get(t, 0.0) * portfolio_value)
        delta_notional = tgt_notional - prev_notional

        if allow_fractional:
            delta_shares = delta_notional / p
        else:
            delta_shares = np.trunc(delta_notional / p)

        action = "BUY" if delta_shares > 0 else ("SELL" if delta_shares < 0 else "HOLD")

        rows.append({
            "date": asof_date,
            "ticker": t,
            "price": p,
            "prev_weight": prev_weights.get(t, 0.0),
            "target_weight": target_weights.get(t, 0.0),
            "delta_notional": float(delta_notional),
            "delta_shares": float(delta_shares),
            "action": action,
        })

    return pd.DataFrame(rows).sort_values(["action", "ticker"]).reset_index(drop=True)
