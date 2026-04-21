"""5-factor backtest scorer — identical logic to the live pipeline.

Uses the same sector-relative z-scoring as build_scores() in score.py,
with the same 5 factors: 3m momentum, 6m momentum, volatility, P/E, sentiment.
Missing P/E values are filled with the sector median (same fallback as live).
Missing sentiment values default to 0.0 (neutral — same as live).
"""

import pandas as pd

from stockpicker.scoring.score import _sector_zscore


def build_backtest_scores(
    price_factors: pd.DataFrame,
    pe_series: pd.Series,
    sentiment_series: pd.Series,
    t2sector: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = price_factors.copy()
    df["pe_ratio"] = pe_series
    df["sentiment"] = sentiment_series
    df = df.dropna(subset=["mom_3m", "mom_6m", "volatility"])

    sectors = pd.Series(t2sector or {}, name="sector").reindex(df.index).fillna("Unknown")

    # Fill missing P/E with sector median then market median — same as live pipeline
    pe_filled = df["pe_ratio"].fillna(
        df["pe_ratio"].groupby(sectors).transform("median")
    )
    pe_filled = pe_filled.fillna(df["pe_ratio"].median())

    df["z_mom_3m"] = _sector_zscore(df["mom_3m"], sectors)
    df["z_mom_6m"] = _sector_zscore(df["mom_6m"], sectors)
    df["z_vol"]    = _sector_zscore(-df["volatility"], sectors)   # lower vol = better
    df["z_pe"]     = _sector_zscore(-pe_filled, sectors)           # lower P/E = better
    df["z_sent"]   = _sector_zscore(df["sentiment"].fillna(0.0), sectors)

    factor_cols = ["z_mom_3m", "z_mom_6m", "z_vol", "z_pe", "z_sent"]
    df["composite_score"] = df[factor_cols].mean(axis=1)
    df["sector"] = sectors

    return df.sort_values("composite_score", ascending=False)
