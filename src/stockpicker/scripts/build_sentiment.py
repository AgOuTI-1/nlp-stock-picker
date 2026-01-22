from __future__ import annotations

import argparse
import pandas as pd

from stockpicker.data.universe import build_universe
from stockpicker.data.prices import download_adj_close, filter_downloaded_universe
from stockpicker.features.technical import build_monthly_dates
from stockpicker.nlp.sentiment_store import MonthlySentimentStore


def main():
    p = argparse.ArgumentParser(description="Build monthly sentiment parquet (resumable).")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--bt-start", default="2017-01-31")
    p.add_argument("--parquet", default="sentiment_monthly.parquet")
    p.add_argument("--max-items", type=int, default=30)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    tickers, t2s = build_universe()
    adj = download_adj_close(tickers, start=args.start, end=None)
    tickers, _ = filter_downloaded_universe(adj, tickers, t2s)

    monthly_dates = build_monthly_dates(adj.index, start=args.bt_start)

    store = MonthlySentimentStore(
        parquet_path=args.parquet,
        max_items=args.max_items,
        overwrite=args.overwrite,
    )
    df = store.build_resumable(tickers=tickers, monthly_dates=monthly_dates)
    print("Wrote rows:", df.shape)


if __name__ == "__main__":
    main()
