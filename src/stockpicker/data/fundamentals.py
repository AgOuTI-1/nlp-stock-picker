import concurrent.futures

import pandas as pd
import yfinance as yf


def _get_pe(ticker: str) -> tuple[str, float | None]:
    try:
        info = yf.Ticker(ticker).info
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe and pe > 0:
            return ticker, float(pe)
    except Exception:
        pass
    return ticker, None


def fetch_pe_ratios(tickers: list[str], max_workers: int = 20) -> pd.Series:
    """
    Fetch trailing (or forward) P/E ratios for each ticker via yfinance.
    Uses a thread pool so 500 tickers don't take forever.
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_get_pe, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker, pe = future.result()
            if pe is not None:
                results[ticker] = pe

    return pd.Series(results, name="pe_ratio", dtype=float)
