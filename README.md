# NLP Stock Picker

Building a monthly stock ranking system that combines momentum/volatility signals with news sentiment from FinBERT. Started this to see if NLP could actually add alpha over basic quant factors—spoiler: results are mixed but interesting.

## What this does

Takes ~50 large-cap stocks, ranks them monthly based on:
- **Momentum** (3 & 6 month returns)
- **Volatility penalty** (3 month realized vol)
- **News sentiment** (optional) - headlines scraped from Google News, scored with FinBERT

The idea: combine what's working (momentum) with what people are saying (sentiment). In practice, sentiment helps sometimes but adds noise other times. More on that below.

## Quick start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# Run a basic backtest (no sentiment)
python -m stockpicker.scripts.run_backtest --start 2017-01-31 --top-n 20 --cost-rate 0.001

# Build sentiment cache (takes a while, caches to disk)
python -m stockpicker.scripts.build_sentiment --start 2017-01-31 --max-items 30

# Backtest with sentiment (lambda-sent controls weight, 0.25 = 25% sentiment / 75% quant)
python -m stockpicker.scripts.run_backtest --start 2017-01-31 --top-n 20 --cost-rate 0.001 --lambda-sent 0.25

# Generate trade sheet for next rebalance
python -m stockpicker.scripts.make_trades --capital 10000 --top-n 20 --lambda-sent 0.25
```

## What I learned

**The good:**
- Caching sentiment to parquet made backtesting way faster (~80% speedup)
- Sector-relative momentum actually matters—comparing tech stocks to tech stocks works better than raw rankings
- FinBERT is surprisingly good at financial headlines vs general sentiment models

**The mixed:**
- Sentiment helps in some market regimes, hurts in others. During 2020 volatility, headlines were just noise
- Google News RSS rate limits aggressively. Had to add retry logic and local caching
- Monthly rebalancing misses intramonth moves but keeps transaction costs reasonable

**What doesn't work:**
- High sentiment weight (>0.3) seems to add more noise than signal
- Using more than ~30 headlines per stock per month hits diminishing returns
- RSS headlines aren't truly point-in-time (Google's index timing is fuzzy)

## Structure

```
src/stockpicker/
  data/        - ticker universe + yfinance price downloads
  features/    - momentum, volatility calcs
  nlp/         - RSS scraping, FinBERT scoring, monthly aggregation
  models/      - combines quant + sentiment into final rankings
  portfolio/   - position sizing, trade generation
  backtest/    - sim engine with transaction costs
  scripts/     - CLI tools
```

## Caveats

- This is **research code**, not production. Don't trade real money on this
- Google News RSS can go down or change format—there's basic error handling but it's not bulletproof
- Sentiment scores are **headline-only**. Full article parsing would be better but way slower
- Transaction costs are simplified (fixed % of turnover). Real slippage varies
- No shorting, just long-only top N stocks

## Ideas for improvement

Things I'd try if I had more time:
- Weight by inverse volatility instead of equal weight
- Try different sentiment aggregation (weighted by source credibility?)
- Add fundamental factors (P/E, P/B) to the mix
- Test different rebalance frequencies
- Better handling of corporate actions

## License

MIT - do whatever you want with it