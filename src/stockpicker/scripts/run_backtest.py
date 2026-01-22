from __future__ import annotations

import argparse
import pandas as pd

from stockpicker.data.universe import build_universe
from stockpicker.data.prices import download_adj_close, filter_downloaded_universe
from stockpicker.features.technical import compute_features
from stockpicker.nlp.sentiment_store import MonthlySentimentStore
from stockpicker.models.scoring import ScoringContext
from stockpicker.backtest.engine import BacktestConfig, run_monthly_backtest
from stockpicker.backtest.metrics import perf_stats_from_equity


def main():
    p = argparse.ArgumentParser(description="Run monthly backtest (quant + optional sentiment).")
    p.add_argument("--start", default="2016-01-01", help="Price download start date (YYYY-MM-DD).")
    p.add_argument("--bt-start", default="2017-01-31", help="Backtest start month-end (YYYY-MM-DD).")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--cost-rate", type=float, default=0.001)
    p.add_argument("--weighting", choices=["equal","inv_vol"], default="equal")
    p.add_argument("--max-weight", type=float, default=0.10)
    p.add_argument("--lambda-sent", type=float, default=0.0, help="Sentiment weight (0 disables).")
    p.add_argument("--sent-parquet", default="sentiment_monthly.parquet")
    p.add_argument("--build-sent", action="store_true", help="Build sentiment parquet before backtest (slow).")
    args = p.parse_args()

    tickers, t2s = build_universe()
    adj = download_adj_close(tickers, start=args.start, end=None)
    tickers, t2s = filter_downloaded_universe(adj, tickers, t2s)
    feats = compute_features(adj)

    sent_lookup = None
    if args.lambda_sent != 0.0:
        if args.build_sent:
            monthly_dates = pd.Series(adj.index).groupby(adj.index.to_period("M")).last()
            monthly_dates = monthly_dates[monthly_dates >= pd.Timestamp(args.bt_start)]
            store = MonthlySentimentStore(parquet_path=args.sent_parquet, overwrite=False)
            store.build_resumable(tickers=tickers, monthly_dates=monthly_dates)
        sent_lookup = MonthlySentimentStore.load_lookup(args.sent_parquet)

    ctx = ScoringContext(ticker_to_sector=t2s, sent_lookup=sent_lookup)
    cfg = BacktestConfig(
        top_n=args.top_n,
        start=args.bt_start,
        cost_rate=args.cost_rate,
        weighting=args.weighting,
        max_weight=args.max_weight,
        lambda_sent=args.lambda_sent,
    )

    bt = run_monthly_backtest(
        adj_close=adj,
        mom_3m=feats["mom_3m"],
        mom_6m=feats["mom_6m"],
        vol_3m=feats["vol_3m"],
        ctx=ctx,
        cfg=cfg,
        initial_capital=1.0
    )

    if bt.empty:
        print("Backtest returned empty results (check dates / data coverage).")
        return

    stats = perf_stats_from_equity(bt["equity_norm"])
    print("Perf stats:", stats)
    out_csv = "backtest_results.csv"
    bt.to_csv(out_csv)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
