import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from stockpicker.data.fundamentals import fetch_pe_ratios
from stockpicker.data.prices import compute_momentum_volatility, fetch_prices
from stockpicker.data.universe import build_universe
from stockpicker.nlp.sentiment import compute_sentiment_scores
from stockpicker.scoring.score import build_scores

OUTPUT_FILE = "results.csv"
DISPLAY_COLS = ["composite_score", "mom_3m", "mom_6m", "volatility", "pe_ratio", "sentiment"]


def main():
    print("Loading S&P 500 tickers...")
    tickers, t2name = build_universe()

    print(f"Fetching prices for {len(tickers)} stocks...")
    prices = fetch_prices(tickers)
    price_factors = compute_momentum_volatility(prices)

    print("Fetching P/E ratios...")
    pe = fetch_pe_ratios(list(price_factors.index))

    print("Fetching headlines and running sentiment...")
    sentiment = compute_sentiment_scores(list(price_factors.index), t2name)

    results = build_scores(price_factors, pe, sentiment)
    results.to_csv(OUTPUT_FILE)
    print(f"\nSaved to {OUTPUT_FILE}\n")

    print("Top 20:")
    print(results[DISPLAY_COLS].head(20).to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
