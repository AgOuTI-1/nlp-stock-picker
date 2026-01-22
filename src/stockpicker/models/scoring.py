from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from stockpicker.features.technical import zscore, month_end


@dataclass
class ScoringContext:
    ticker_to_sector: Dict[str, str]
    sent_lookup: Optional[pd.DataFrame] = None  # MultiIndex (month_end, ticker) -> sent_z


def _get_sent_z(sent_lookup: pd.DataFrame | None, ticker: str, asof_date: pd.Timestamp) -> float:
    if sent_lookup is None:
        return 0.0
    me = month_end(asof_date)
    try:
        return float(sent_lookup.loc[(me, ticker), "sent_z"])
    except KeyError:
        return 0.0


def score_universe(
    asof_date: pd.Timestamp,
    mom_3m: pd.DataFrame,
    mom_6m: pd.DataFrame,
    vol_3m: pd.DataFrame,
    ctx: ScoringContext,
    top_n: int = 20,
    lambda_sent: float = 0.0,
) -> List[str]:
    rankable = mom_3m.loc[asof_date].dropna().index.tolist()
    if len(rankable) < top_n:
        return []

    df = pd.DataFrame({
        "ticker": rankable,
        "sector": [ctx.ticker_to_sector[t] for t in rankable],
        "mom_3m": mom_3m.loc[asof_date, rankable].values,
        "mom_6m": mom_6m.loc[asof_date, rankable].values,
        "vol_3m": vol_3m.loc[asof_date, rankable].values,
    }).dropna()

    if df.empty or df["ticker"].nunique() < top_n:
        return []

    df["mom3_z_sector"] = df.groupby("sector")["mom_3m"].transform(zscore)
    df["mom6_z_sector"] = df.groupby("sector")["mom_6m"].transform(zscore)
    df["vol_z"] = zscore(df["vol_3m"])
    df["QuantScore"] = 0.6*df["mom3_z_sector"] + 0.4*df["mom6_z_sector"] - 0.3*df["vol_z"]

    if lambda_sent != 0.0:
        df["sent_z"] = df["ticker"].apply(lambda t: _get_sent_z(ctx.sent_lookup, t, asof_date))
    else:
        df["sent_z"] = 0.0

    df["FinalScore"] = df["QuantScore"] + lambda_sent * df["sent_z"]
    return df.sort_values("FinalScore", ascending=False).head(top_n)["ticker"].tolist()
