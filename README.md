# NLP Stock Picker

Ranks S&P 500 stocks by combining momentum, volatility, and P/E signals with NLP sentiment from recent news headlines into a "quant score". The idea is to see if knowing what people are saying about a stock adds anything on top of price-based signals. Essentially, we're trying to add qualitative flavor to a traditional stock scoring model.

## How it works

Pulls the full S&P 500 constituent list from Wikipedia, then for each stock computes:

- **3 & 6 month momentum** — recent price performance
- **Volatility** — lower is better, penalises jumpy stocks
- **P/E ratio** — lower is better
- **News sentiment** — recent headlines scored with FinBERT (a financial NLP model)

Each factor is then normalized across it's respective ticker's sector. If there is <30 companies in a given sector in the universe, then it will take the market-wide z score for that factor and blend it with the given ticker's z-score. If there is >30 companies in a given sector, then it will z-score on a sectoral basis to account for natural differences in company types: e.g. tech stocks typically have higher volatility than staples, but that doesn't explicitly mean it is a bad investment decision. It's all relative. So this is how we are getting around that.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

python main.py
```

Outputs `results.csv` with the full ranked list and prints the top 20 to the terminal. Expect it to take a few minutes — fetching headlines for 500 stocks is the slow part.

## Structure

```
src/stockpicker/
  data/         universe, prices, P/E ratios
  nlp/          headline fetching + FinBERT scoring
  scoring/      combines factors into final rankings
main.py
```

## Notes

- Sentiment is headline-only — full article parsing would probably help but it's much slower
- FinBERT works better here than general-purpose sentiment models since it's trained on financial text
- Google News rate limits occasionally, so a handful of tickers may come back with no headlines

## License

MIT
