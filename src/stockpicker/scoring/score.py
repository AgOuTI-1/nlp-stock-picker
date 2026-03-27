import pandas as pd


def _sector_zscore(series: pd.Series, sectors: pd.Series, min_n: int = 15) -> pd.Series:
    """
    Compute a blended z-score: sector-relative when the sector is large enough,
    market-wide when the sector is too small to produce reliable statistics.

    Blend weight w = min(1, sector_size / min_n):
      - w=1  → pure sector z-score  (sector has >= min_n stocks)
      - w=0  → pure market z-score  (sector has 1 stock)
      - in between → proportional mix
    """
    mkt_mean = series.mean()
    mkt_std = series.std() if series.std() > 0 else 1.0
    z_mkt = (series - mkt_mean) / mkt_std

    grp = series.groupby(sectors)
    sec_mean = grp.transform("mean")
    sec_std = grp.transform("std").fillna(mkt_std).replace(0, mkt_std)
    z_sec = (series - sec_mean) / sec_std

    sec_n = grp.transform("count")
    w = (sec_n / min_n).clip(upper=1.0)

    return w * z_sec + (1 - w) * z_mkt


def build_scores(
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

    pe_filled = df["pe_ratio"].fillna(df["pe_ratio"].groupby(sectors).transform("median"))
    pe_filled = pe_filled.fillna(df["pe_ratio"].median())  # fallback for unknown sectors

    df["z_mom_3m"] = _sector_zscore(df["mom_3m"], sectors)
    df["z_mom_6m"] = _sector_zscore(df["mom_6m"], sectors)
    df["z_vol"] = _sector_zscore(-df["volatility"], sectors)   # lower vol = better
    df["z_pe"] = _sector_zscore(-pe_filled, sectors)            # lower P/E = better
    df["z_sent"] = _sector_zscore(df["sentiment"].fillna(0.0), sectors)

    factor_cols = ["z_mom_3m", "z_mom_6m", "z_vol", "z_pe", "z_sent"]
    df["composite_score"] = df[factor_cols].mean(axis=1)

    df["sector"] = sectors

    return df.sort_values("composite_score", ascending=False)
