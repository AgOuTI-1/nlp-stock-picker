from __future__ import annotations

import numpy as np
import pandas as pd


def perf_stats_from_equity(equity: pd.Series) -> dict:
    eq = pd.Series(equity).dropna()
    if len(eq) < 2:
        return {
            "months": 0,
            "total_return": np.nan,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    rets = eq.pct_change().dropna()
    months = int(len(rets))
    years = months / 12.0 if months > 0 else np.nan

    start_val = float(eq.iloc[0])
    end_val = float(eq.iloc[-1])
    total_return = (end_val / start_val - 1.0) if start_val > 0 else np.nan
    cagr = ((end_val / start_val) ** (1.0 / years) - 1.0) if (start_val > 0 and years and years > 0) else np.nan

    ann_vol = float(rets.std(ddof=0) * np.sqrt(12)) if months > 1 else np.nan
    sharpe = float((rets.mean() / (rets.std(ddof=0) + 1e-12)) * np.sqrt(12)) if months > 1 else np.nan

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "months": months,
        "total_return": float(total_return),
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "ann_vol": float(ann_vol) if np.isfinite(ann_vol) else np.nan,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(max_drawdown),
    }
