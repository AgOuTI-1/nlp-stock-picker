"""Full 5-factor backtest: momentum + volatility + P/E + sentiment (GDELT).

First run: ~30–60 minutes (fetches and caches prices, P/E, sentiment).
Subsequent runs: ~5 minutes (reads from cached CSVs).

To force a full refresh, delete:
  backtest_sentiment_cache.csv
  backtest_pe_cache.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from stockpicker.backtest.engine import run_backtest
from stockpicker.backtest.report import compute_stats, plot_results, print_stats

BENCHMARK  = "SPY"
START_DATE = "2015-01-01"
END_DATE   = "2025-04-01"
TOP_N      = 30


def main():
    results = run_backtest(
        start_date=START_DATE,
        end_date=END_DATE,
        top_n=TOP_N,
        benchmark=BENCHMARK,
    )

    strategy_stats = compute_stats(results["strategy_r"])
    spy_stats      = compute_stats(results["spy_r"])

    print()
    print_stats(strategy_stats, spy_stats, benchmark=BENCHMARK)

    results["monthly_returns"].to_csv("backtest_returns.csv")
    print("\nMonthly returns saved → backtest_returns.csv")

    plot_results(
        results["strategy_cumret"],
        results["spy_cumret"],
        benchmark=BENCHMARK,
        output_path="backtest.png",
    )


if __name__ == "__main__":
    main()
