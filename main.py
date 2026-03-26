import sys
from pathlib import Path

# allow running as `python main.py` without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stockpicker.data.fundamentals import fetch_pe_ratios
from stockpicker.data.prices import compute_momentum_volatility, fetch_prices
from stockpicker.data.universe import build_universe
from stockpicker.nlp.sentiment import compute_sentiment_scores
from stockpicker.scoring.score import build_scores

OUTPUT_FILE = "results.csv"
DISPLAY_COLS = ["composite_score", "mom_3m", "mom_6m", "volatility", "pe_ratio", "sentiment"]


def main():
    print("=== S&P 500 Stock Picker ===\n")

    print("Step 1/5  Building universe from Wikipedia...")
    tickers, t2name = build_universe()
    print(f"          {len(tickers)} tickers loaded\n")

    print("Step 2/5  Fetching 1-year price history...")
    prices = fetch_prices(tickers)
    price_factors = compute_momentum_volatility(prices)
    print(f"          {len(price_factors)} tickers with sufficient history\n")

    print("Step 3/5  Fetching P/E ratios...")
    pe = fetch_pe_ratios(list(price_factors.index))
    print(f"          P/E available for {pe.notna().sum()} tickers\n")

    print("Step 4/5  Fetching headlines and scoring sentiment...")
    sentiment = compute_sentiment_scores(list(price_factors.index), t2name)
    print()

    print("Step 5/5  Building composite scores...")
    results = build_scores(price_factors, pe, sentiment)

    results.to_csv(OUTPUT_FILE)
    print(f"          Full rankings saved to {OUTPUT_FILE}\n")

    print("=== Top 20 ===")
    print(results[DISPLAY_COLS].head(20).to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
