import numpy as np
import pandas as pd
from gnews import GNews
from transformers import pipeline


def _fetch_headlines(ticker: str, company_name: str | None, n: int = 5) -> list[str]:
    """Fetch up to n recent news headlines for a ticker."""
    gn = GNews(language="en", country="US", max_results=n, period="7d")
    query = f"{company_name} stock" if company_name else f"{ticker} stock"
    try:
        articles = gn.get_news(query)
        return [a["title"] for a in articles if a.get("title")]
    except Exception:
        return []


def _score_headlines(headlines: list[str], finbert) -> float:
    """
    Run FinBERT on a list of headlines and return the mean net sentiment
    (positive score minus negative score). Returns 0.0 for empty input.
    """
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
    """
    For each ticker, fetch recent headlines and score them with FinBERT.
    Returns a Series of net sentiment scores indexed by ticker.
    """
    t2name = t2name or {}

    print("  Loading FinBERT model...")
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        return_all_scores=True,
        device=-1,  # CPU; set to 0 if you have a GPU
    )

    sentiment = {}
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  Sentiment: {i}/{len(tickers)}")
        try:
            headlines = _fetch_headlines(ticker, t2name.get(ticker))
            sentiment[ticker] = _score_headlines(headlines, finbert)
        except Exception:
            sentiment[ticker] = 0.0

    return pd.Series(sentiment, name="sentiment", dtype=float)
