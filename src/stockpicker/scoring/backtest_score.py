import pandas as pd

from stockpicker.scoring.score import _sector_zscore


def build_backtest_scores(
    price_factors: pd.DataFrame,
    t2sector: dict[str, str] | None = None,
) -> pd.DataFrame:
    """3-factor composite score (momentum + volatility only — no P/E or sentiment).

    Used by the backtester, which cannot access historical P/E snapshots or
    historical news headlines. Reuses the same sector-relative z-score logic
    as the live scoring pipeline.
    """
    df = price_factors.copy()
    df = df.dropna(subset=["mom_3m", "mom_6m", "volatility"])

    sectors = pd.Series(t2sector or {}, name="sector").reindex(df.index).fillna("Unknown")

    df["z_mom_3m"] = _sector_zscore(df["mom_3m"], sectors)
    df["z_mom_6m"] = _sector_zscore(df["mom_6m"], sectors)
    df["z_vol"] = _sector_zscore(-df["volatility"], sectors)  # lower vol = better

    factor_cols = ["z_mom_3m", "z_mom_6m", "z_vol"]
    df["composite_score"] = df[factor_cols].mean(axis=1)
    df["sector"] = sectors

    return df.sort_values("composite_score", ascending=False)
