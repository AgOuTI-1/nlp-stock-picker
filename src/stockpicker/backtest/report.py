import math

import pandas as pd


def compute_stats(r: pd.Series) -> dict:
    """Compute key performance metrics from a monthly return series."""
    n = len(r)
    total = (1 + r).prod() - 1
    annualized = (1 + total) ** (12 / n) - 1 if n > 0 else 0.0
    sharpe = (r.mean() / r.std()) * math.sqrt(12) if r.std() > 0 else 0.0

    cumret = (1 + r).cumprod()
    running_peak = cumret.cummax()
    drawdown = (cumret - running_peak) / running_peak
    max_dd = drawdown.min()

    return {
        "total_return": total,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_months": n,
    }


def print_stats(strategy_stats: dict, spy_stats: dict, benchmark: str = "SPY") -> None:
    def fmt_pct(v): return f"{v * 100:+.1f}%"
    def fmt_ratio(v): return f"{v:.2f}"

    print("=" * 52)
    print(f"{'Backtest Results':^52}")
    print(f"{'(3-factor: momentum + volatility, no P/E or sentiment)':^52}")
    print(f"{'Note: no transaction costs modeled':^52}")
    print("=" * 52)
    print(f"{'Metric':<28} {'Strategy':>10} {benchmark:>10}")
    print("-" * 52)
    print(f"{'Total Return':<28} {fmt_pct(strategy_stats['total_return']):>10} {fmt_pct(spy_stats['total_return']):>10}")
    print(f"{'Annualized Return':<28} {fmt_pct(strategy_stats['annualized_return']):>10} {fmt_pct(spy_stats['annualized_return']):>10}")
    print(f"{'Sharpe Ratio (rf=0)':<28} {fmt_ratio(strategy_stats['sharpe']):>10} {fmt_ratio(spy_stats['sharpe']):>10}")
    print(f"{'Max Drawdown':<28} {fmt_pct(strategy_stats['max_drawdown']):>10} {fmt_pct(spy_stats['max_drawdown']):>10}")
    print(f"{'Months':<28} {strategy_stats['n_months']:>10} {spy_stats['n_months']:>10}")
    print("=" * 52)


def plot_results(
    strategy_cumret: pd.Series,
    spy_cumret: pd.Series,
    benchmark: str = "SPY",
    output_path: str = "backtest.png",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot((strategy_cumret * 100), label="Strategy (mom + vol)", color="#1f77b4", linewidth=2)
    ax.plot((spy_cumret * 100), label=benchmark, color="#d62728", linewidth=2, linestyle="--")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Walk-Forward Backtest: Strategy vs SPY\n(3-factor: 3m momentum, 6m momentum, volatility)", fontsize=13)
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")
    plt.show()
