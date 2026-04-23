# NLP Stock Picker

This program was a vibe-coded project that ranks S&P 500 stocks by combining momentum, volatility, and P/E signals with NLP sentiment from recent news headlines into a "quant score". The idea is to see if knowing what people are saying about a stock adds anything on top of price-based signals. Essentially, we're trying to add qualitative flavor to a traditional stock scoring model.

## How it works

Pulls the full S&P 500 constituent list from Wikipedia, then for each stock computes five factors:

- **3 & 6 month momentum** — recent price performance
- **Volatility** — lower is better, penalises jumpy stocks
- **P/E ratio** — lower is better
- **News sentiment** — recent headlines scored with FinBERT (a financial NLP model)

### Sector-relative scoring

Each factor is z-scored within its GICS sector rather than across the whole universe. Tech stocks typically have higher volatility than staples — that doesn't make them bad investments, it just reflects the nature of the business. Comparing them sector-relative means the score captures how a stock looks among its peers, which is a more meaningful signal.

To handle small sectors (e.g. Communication Services has ~23 S&P 500 constituents), the sector z-score is blended with the market-wide z-score in proportion to sector size. Sectors with fewer than 15 stocks lean toward the market-wide score to avoid noisy small-sample statistics; sectors with 15+ stocks use the pure sector z-score.

The five z-scores are averaged into a **composite score** (units: standard deviations above/below the sector average). Higher is better.

### Outlier filtering

Stocks whose momentum or volatility factors fall more than 4 standard deviations from the cross-sectional mean are excluded before scoring. These outliers almost always reflect M&A events, reverse splits, or data errors rather than genuine tradeable returns.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

TRANSFORMERS_OFFLINE=1 python main.py
```

`TRANSFORMERS_OFFLINE=1` tells the FinBERT model to use its local cache and skip post-load network checks, which avoids a hang on first run after the model is cached, but this stuff is over my head, honestly. I used claude code to help me out here. Shoutout Claude Code man. Incredible product.

When you run, it will take a minute due to how computationally intensive FinBERT is, but eventually after ~15 mins it outputs `results.csv` with the full ranked list and prints the top 20 to the terminal. Expect it to take 20–60 minutes — fetching and scoring headlines for 500 stocks is the slow part.

## Structure

```
src/stockpicker/
  data/         universe (with GICS sectors), prices, P/E ratios
  nlp/          headline fetching + FinBERT scoring
  scoring/      sector-relative z-score normalisation and composite ranking
main.py
```

## Output columns

| Column | Description |
|--------|-------------|
| `sector` | GICS sector |
| `composite_score` | Average of the five z-scores (higher = better) |
| `mom_3m` / `mom_6m` | Raw 3- and 6-month price returns |
| `volatility` | Annualised volatility (126-day) |
| `pe_ratio` | Trailing or forward P/E |
| `sentiment` | Mean FinBERT net sentiment across recent headlines |
| `z_mom_3m` / `z_mom_6m` | Sector-blended z-score for 3- and 6-month momentum |
| `z_vol` | Sector-blended z-score for volatility (inverted: lower vol = higher score) |
| `z_pe` | Sector-blended z-score for P/E (inverted: lower P/E = higher score) |
| `z_sent` | Sector-blended z-score for sentiment |

## Notes

- Sentiment is headline-only — full article parsing would probably help but it's much slower
- FinBERT works better here than general-purpose sentiment models since it's trained on financial text
- Google News rate limits occasionally, so a handful of tickers may come back with no headlines; those tickers default to neutral sentiment (0.0)

## License

MIT
