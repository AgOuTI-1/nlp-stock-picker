import pandas as pd


def _rank_norm(series: pd.Series) -> pd.Series:
    ranked = series.rank(method="average", na_option="bottom")
    return (ranked - ranked.min()) / (ranked.max() - ranked.min())


def build_scores(
    price_factors: pd.DataFrame,
    pe_series: pd.Series,
    sentiment_series: pd.Series,
) -> pd.DataFrame:
    df = price_factors.copy()
    df["pe_ratio"] = pe_series
    df["sentiment"] = sentiment_series
    df = df.dropna(subset=["mom_3m", "mom_6m", "volatility"])

    df["r_mom_3m"] = _rank_norm(df["mom_3m"])
    df["r_mom_6m"] = _rank_norm(df["mom_6m"])
    df["r_vol"] = _rank_norm(-df["volatility"])   # lower vol = better

    pe_filled = df["pe_ratio"].fillna(df["pe_ratio"].median())
    df["r_pe"] = _rank_norm(-pe_filled)            # lower P/E = better

    df["r_sent"] = _rank_norm(df["sentiment"].fillna(0.0))

    factor_cols = ["r_mom_3m", "r_mom_6m", "r_vol", "r_pe", "r_sent"]
    df["composite_score"] = df[factor_cols].mean(axis=1)

    return df.sort_values("composite_score", ascending=False)
