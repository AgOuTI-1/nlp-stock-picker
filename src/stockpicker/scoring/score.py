import pandas as pd


def _rank_norm(series: pd.Series) -> pd.Series:
    """Percentile-rank a series and scale to [0, 1]. NaNs rank at the bottom."""
    ranked = series.rank(method="average", na_option="bottom")
    return (ranked - ranked.min()) / (ranked.max() - ranked.min())


def build_scores(
    price_factors: pd.DataFrame,
    pe_series: pd.Series,
    sentiment_series: pd.Series,
) -> pd.DataFrame:
    """
    Combine all quant factors into a single composite score.

    Factors (equal-weighted):
        mom_3m     higher is better
        mom_6m     higher is better
        volatility lower is better  -> inverted before ranking
        pe_ratio   lower is better  -> inverted before ranking
        sentiment  higher is better

    Each factor is percentile-ranked across the universe before averaging,
    so scale differences don't matter and outliers don't dominate.
    """
    df = price_factors.copy()
    df["pe_ratio"] = pe_series
    df["sentiment"] = sentiment_series

    # drop tickers missing the price-based factors (not enough history)
    df = df.dropna(subset=["mom_3m", "mom_6m", "volatility"])

    df["r_mom_3m"] = _rank_norm(df["mom_3m"])
    df["r_mom_6m"] = _rank_norm(df["mom_6m"])
    df["r_vol"] = _rank_norm(-df["volatility"])   # lower vol -> higher rank

    # P/E: fill missing with median so absent data is neutral, then invert
    pe_filled = df["pe_ratio"].fillna(df["pe_ratio"].median())
    df["r_pe"] = _rank_norm(-pe_filled)            # lower PE -> higher rank

    # Sentiment: fill missing with 0 (neutral)
    df["r_sent"] = _rank_norm(df["sentiment"].fillna(0.0))

    factor_cols = ["r_mom_3m", "r_mom_6m", "r_vol", "r_pe", "r_sent"]
    df["composite_score"] = df[factor_cols].mean(axis=1)

    return df.sort_values("composite_score", ascending=False)
