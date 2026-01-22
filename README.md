HEAD
# nlp-stock-picker
Using a mix of qualitative and quantitative factors to influence a large-cap stock picker. 

# NLP Stock Picker

A research-to-production style codebase for a monthly-rebalanced equity ranking strategy that blends:
- **sector-relative momentum + volatility penalty** (quant baseline)
- **headline sentiment** (FinBERT) aggregated monthly (optional NLP overlay)

This repo is a cleaned-up, modular refactor of your original notebook-style script. fileciteturn0file0

## What’s included

- **Universe**: a curated, sector-balanced ticker list (editable)
- **Data**: price download via `yfinance`
- **Features**: 3M/6M momentum + 3M volatility
- **NLP**: Google News RSS headlines → FinBERT sentiment → monthly parquet cache
- **Backtest**: monthly rebalance, equal-weight or inverse-vol weights, turnover-based transaction costs
- **Trade sheet**: portfolio table + trade blotter for a given rebalance date

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

### 2) Run a backtest (quant-only)
```bash
python -m stockpicker.scripts.run_backtest --start 2017-01-31 --top-n 20 --cost-rate 0.001
```

### 3) Build monthly sentiment cache (optional, slower)
```bash
python -m stockpicker.scripts.build_sentiment --start 2017-01-31 --max-items 30
```

### 4) Run a backtest with sentiment
```bash
python -m stockpicker.scripts.run_backtest --start 2017-01-31 --top-n 20 --cost-rate 0.001 --lambda-sent 0.25
```

### 5) Generate a “trade blotter” for the latest month
```bash
python -m stockpicker.scripts.make_trades --capital 10000 --top-n 20 --lambda-sent 0.25
```

## Notes / caveats

- Google News RSS can rate-limit; there’s a lightweight on-disk cache under `nlp_cache/`.
- Sentiment is **headline-only** and not point-in-time perfect (RSS availability changes). Treat results as research, not trading advice.
- Transaction costs are modeled as `cost_rate * turnover` per rebalance.

## Repo layout

```text
src/stockpicker/
  data/        # universe + prices
  features/    # technical features
  nlp/         # rss + finbert + monthly sentiment store
  models/      # scoring (quant + sentiment)
  portfolio/   # weights + trade blotter
  backtest/    # engine + metrics
  scripts/     # CLI entrypoints
```

## License

MIT (see `LICENSE`).
e7f3c69 (Initial commit: refactor NLP stock picker into modular repo)
