import socket

import numpy as np
import pandas as pd
import requests as _requests
import torch
from gnews import GNews
from transformers import pipeline

_GNEWS_TIMEOUT = 10  # seconds per network call inside GNews

# requests ignores socket.setdefaulttimeout(), so patch it directly
_original_request = _requests.Session.request


def _request_with_timeout(self, *args, **kwargs):
    kwargs.setdefault("timeout", _GNEWS_TIMEOUT)
    return _original_request(self, *args, **kwargs)


_requests.Session.request = _request_with_timeout


def _fetch_headlines(ticker: str, company_name: str | None, n: int = 5) -> list[str]:
    gn = GNews(language="en", country="US", max_results=n, period="7d")
    query = f"{company_name} stock" if company_name else f"{ticker} stock"
    try:
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(_GNEWS_TIMEOUT)
        try:
            articles = gn.get_news(query)
        finally:
            socket.setdefaulttimeout(old_timeout)
        return [a["title"] for a in articles if a.get("title")]
    except Exception:
        return []


def _score_headlines(headlines: list[str], finbert) -> float:
    if not headlines:
        return 0.0

    results = finbert(headlines[:5], truncation=True, max_length=512)
    scores = []
    for result in results:
        by_label = {r["label"]: r["score"] for r in result}
        net = by_label.get("positive", 0.0) - by_label.get("negative", 0.0)
        scores.append(net)
    return float(np.mean(scores))


def compute_sentiment_scores(
    tickers: list[str],
    t2name: dict[str, str] | None = None,
) -> pd.Series:
    t2name = t2name or {}

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1

    device_name = {0: "CUDA GPU", "mps": "Apple MPS GPU", -1: "CPU"}[device]
    print(f"Loading FinBERT on {device_name}...", flush=True)
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,
        device=device,
    )

    sentiment = {}
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} tickers scored", flush=True)
        try:
            headlines = _fetch_headlines(ticker, t2name.get(ticker))
            sentiment[ticker] = _score_headlines(headlines, finbert)
        except Exception:
            sentiment[ticker] = 0.0

    return pd.Series(sentiment, name="sentiment", dtype=float)
