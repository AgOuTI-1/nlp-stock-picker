from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from stockpicker.features.technical import month_end, zscore
from stockpicker.nlp.finbert import FinBertScorer
from stockpicker.nlp.news_rss import GoogleNewsRSS


@dataclass
class MonthlySentimentStore:
    parquet_path: str = "sentiment_monthly.parquet"
    max_items: int = 30
    batch_size: int = 64
    overwrite: bool = False
    rss: Optional[GoogleNewsRSS] = None
    finbert: Optional[FinBertScorer] = None

    def __post_init__(self):
        if self.rss is None:
            self.rss = GoogleNewsRSS()
        if self.finbert is None:
            self.finbert = FinBertScorer()

    def build_resumable(self, tickers: List[str], monthly_dates: pd.Series) -> pd.DataFrame:
        """Build/update a monthly sentiment parquet, resumable if the file exists."""
        if os.path.exists(self.parquet_path) and not self.overwrite:
            existing = pd.read_parquet(self.parquet_path)
            existing["month_end"] = pd.to_datetime(existing["month_end"])
            done = set(zip(existing["month_end"], existing["ticker"]))
            rows = existing.to_dict("records")
        else:
            if self.overwrite and os.path.exists(self.parquet_path):
                os.remove(self.parquet_path)
            done = set()
            rows = []

        for asof_date in monthly_dates:
            me = month_end(asof_date)
            month_rows = []
            for t in tickers:
                key = (me, t)
                if key in done:
                    continue

                query = f"{t} stock"
                cache_key = f"rss_{query}_{me.date()}_{self.max_items}"

                headlines = self.rss.get_cached(
                    cache_key,
                    fetch_fn=lambda: self.rss.fetch(query=query, max_items=self.max_items),
                )
                titles = [h.get("title", "") for h in headlines if h.get("title")]

                if not titles:
                    s = 0.0
                    n = 0
                else:
                    parts = []
                    for i in range(0, len(titles), self.batch_size):
                        parts.append(self.finbert.score_batch(titles[i:i+self.batch_size]))
                    scores = np.concatenate(parts) if parts else np.array([])
                    s = float(np.mean(scores)) if len(scores) else 0.0
                    n = int(len(titles))

                month_rows.append({"month_end": me, "ticker": t, "sentiment_mean": s, "n_headlines": n})
                done.add(key)

            if month_rows:
                rows.extend(month_rows)
                pd.DataFrame(rows).to_parquet(self.parquet_path, index=False)

        return pd.read_parquet(self.parquet_path)

    @staticmethod
    def load_lookup(parquet_path: str) -> pd.DataFrame:
        df = pd.read_parquet(parquet_path).copy()
        df["month_end"] = pd.to_datetime(df["month_end"])
        df["ticker"] = df["ticker"].astype(str)
        df["sent_z"] = df.groupby("month_end")["sentiment_mean"].transform(zscore)
        return df.set_index(["month_end", "ticker"]).sort_index()
