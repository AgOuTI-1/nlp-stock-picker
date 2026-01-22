from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from stockpicker.features.technical import build_monthly_dates
from stockpicker.models.scoring import ScoringContext, score_universe
from stockpicker.portfolio.weights import make_equal_weights, make_inv_vol_weights
from stockpicker.portfolio.trades import compute_turnover


@dataclass
class BacktestConfig:
    top_n: int = 20
    start: str = "2017-01-31"
    cost_rate: float = 0.001
    weighting: str = "equal"   # equal | inv_vol
    max_weight: float = 0.10
    lambda_sent: float = 0.0


def _make_target_weights(cfg: BacktestConfig, picks: list[str], asof_date: pd.Timestamp, vol_3m: pd.DataFrame) -> Dict[str, float]:
    if cfg.weighting == "equal":
        return make_equal_weights(picks)
    if cfg.weighting == "inv_vol":
        return make_inv_vol_weights(picks, asof_date, vol_3m, max_weight=cfg.max_weight)
    raise ValueError("weighting must be 'equal' or 'inv_vol'")


def run_monthly_backtest(
    adj_close: pd.DataFrame,
    mom_3m: pd.DataFrame,
    mom_6m: pd.DataFrame,
    vol_3m: pd.DataFrame,
    ctx: ScoringContext,
    cfg: BacktestConfig,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    dates = build_monthly_dates(adj_close.index, start=cfg.start)

    equity_rows = []
    portfolio_value = float(initial_capital)
    prev_weights: Dict[str, float] = {}

    for i in range(len(dates) - 1):
        asof_date = pd.Timestamp(dates.iloc[i])
        next_date = pd.Timestamp(dates.iloc[i + 1])

        picks = score_universe(
            asof_date=asof_date,
            mom_3m=mom_3m,
            mom_6m=mom_6m,
            vol_3m=vol_3m,
            ctx=ctx,
            top_n=cfg.top_n,
            lambda_sent=cfg.lambda_sent,
        )
        if len(picks) < cfg.top_n:
            continue

        # ensure we have prices at both endpoints for all picks
        if adj_close.loc[asof_date, picks].isna().any() or adj_close.loc[next_date, picks].isna().any():
            continue

        target_weights = _make_target_weights(cfg, picks, asof_date, vol_3m)

        turnover = compute_turnover(prev_weights, target_weights)
        cost_paid_frac = cfg.cost_rate * turnover
        portfolio_value_after_cost = portfolio_value * (1.0 - cost_paid_frac)

        start_px = adj_close.loc[asof_date, picks].astype(float)
        end_px = adj_close.loc[next_date, picks].astype(float)
        rel = (end_px / start_px - 1.0).replace([np.inf, -np.inf], np.nan).dropna()

        if len(rel) == 0:
            continue

        names = rel.index.tolist()
        wvec = np.array([target_weights.get(t, 0.0) for t in names], dtype=float)
        wsum = wvec.sum()
        if wsum <= 0:
            continue
        wvec = wvec / wsum
        port_ret = float(np.dot(wvec, rel.values))

        portfolio_value = portfolio_value_after_cost * (1.0 + port_ret)

        equity_rows.append({
            "date": next_date,
            "portfolio_return": port_ret,
            "turnover": turnover,
            "cost_paid_frac": cost_paid_frac,
            "equity": portfolio_value,
        })

        prev_weights = target_weights

    bt = pd.DataFrame(equity_rows)
    if bt.empty:
        return bt
    bt = bt.set_index("date").sort_index()
    bt["equity_norm"] = bt["equity"] / bt["equity"].iloc[0]
    return bt
