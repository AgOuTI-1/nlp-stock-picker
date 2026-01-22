from __future__ import annotations

import argparse
import pandas as pd

from stockpicker.data.universe import build_universe
from stockpicker.data.prices import download_adj_close, filter_downloaded_universe
from stockpicker.features.technical import compute_features, build_monthly_dates
from stockpicker.nlp.sentiment_store import MonthlySentimentStore
from stockpicker.models.scoring import ScoringContext, score_universe
from stockpicker.portfolio.weights import make_equal_weights, make_inv_vol_weights
from stockpicker.portfolio.trades import make_trade_blotter


def main():
    p = argparse.ArgumentParser(description="Generate portfolio + trade blotter for latest rebalance date.")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--bt-start", default="2017-01-31")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--lambda-sent", type=float, default=0.0)
    p.add_argument("--sent-parquet", default="sentiment_monthly.parquet")
    p.add_argument("--weighting", choices=["equal","inv_vol"], default="equal")
    p.add_argument("--max-weight", type=float, default=0.10)
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--allow-fractional", action="store_true")
    args = p.parse_args()

    tickers, t2s = build_universe()
    adj = download_adj_close(tickers, start=args.start, end=None)
    tickers, t2s = filter_downloaded_universe(adj, tickers, t2s)
    feats = compute_features(adj)

    sent_lookup = None
    if args.lambda_sent != 0.0:
        sent_lookup = MonthlySentimentStore.load_lookup(args.sent_parquet)

    ctx = ScoringContext(ticker_to_sector=t2s, sent_lookup=sent_lookup)

    monthly = build_monthly_dates(adj.index, start=args.bt_start)
    asof = pd.Timestamp(monthly.iloc[-1])

    picks = score_universe(asof, feats["mom_3m"], feats["mom_6m"], feats["vol_3m"], ctx, top_n=args.top_n, lambda_sent=args.lambda_sent)

    if args.weighting == "equal":
        target_w = make_equal_weights(picks)
    else:
        target_w = make_inv_vol_weights(picks, asof, feats["vol_3m"], max_weight=args.max_weight)

    portfolio = pd.DataFrame({
        "ticker": picks,
        "weight": [target_w[t] for t in picks],
        "price": [float(adj.loc[asof, t]) for t in picks],
    })
    portfolio["dollars"] = portfolio["weight"] * args.capital
    if args.allow_fractional:
        portfolio["shares"] = portfolio["dollars"] / portfolio["price"]
    else:
        portfolio["shares"] = (portfolio["dollars"] / portfolio["price"]).apply(lambda x: float(int(x)))
    portfolio = portfolio.sort_values("weight", ascending=False).reset_index(drop=True)

    trades = make_trade_blotter(
        asof_date=asof,
        portfolio_value=args.capital,
        prev_weights={},  # fill with your broker holdings if you want
        target_weights=target_w,
        price_df=adj,
        allow_fractional=args.allow_fractional,
    )

    portfolio.to_csv("portfolio_recommendation.csv", index=False)
    trades.to_csv("trade_blotter.csv", index=False)

    print("ASOF:", asof.date())
    print("Saved: portfolio_recommendation.csv, trade_blotter.csv")


if __name__ == "__main__":
    main()
