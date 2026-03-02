import matplotlib.pyplot as plt
import pandas as pd

from data import load_ibov, load_prices
from signals import rolling_vol_signal
from portfolio import momentum_strategy
from pipeline import run_pipeline


def print_metrics(metrics: dict) -> None:
    rows = [
        ("Retorno Total", "strategy_total_return", "benchmark_total_return", "{:.1%}"),
        ("Retorno Anualizado", "strategy_ann_return", "benchmark_ann_return", "{:.2%}"),
        ("Sharpe Ratio", "strategy_sharpe", "benchmark_sharpe", "{:.3f}"),
        ("Max Drawdown", "strategy_max_drawdown", "benchmark_max_drawdown", "{:.1%}"),
    ]
    print(f"\n{'Métrica':<22} {'Estratégia':>12} {'Benchmark':>12}")
    print("-" * 48)
    for label, sk, bk, fmt in rows:
        sv = fmt.format(metrics[sk])
        bv = fmt.format(metrics[bk])
        print(f"{label:<22} {sv:>12} {bv:>12}")
    print()


def main() -> None:
    print("Carregando dados...")
    prices = load_prices()
    ibov = load_ibov()

    regime_signal = rolling_vol_signal(window=21, threshold=0.20)
    strategy = momentum_strategy(n=10, lookback=60)
    risk_filters = []

    print("Executando pipeline...")
    result = run_pipeline(prices, ibov, regime_signal, strategy, risk_filters)

    print_metrics(result["metrics"])

    # Curvas de equity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    eq = result["equity"]
    ibov_eq = result["ibov_equity"]

    ax1.plot(eq.index, eq.values, label="Estratégia Momentum + Regime", color="steelblue")
    ax1.plot(ibov_eq.index, ibov_eq.values, label="IBOV Buy & Hold", color="orange", alpha=0.8)
    ax1.set_ylabel("Equity (base 1)")
    ax1.set_title("Curvas de Equity — Estratégia vs IBOV (2010–2024)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Regime overlay
    regime_aligned = result["regime"].reindex(eq.index)
    ax2.fill_between(
        regime_aligned.index,
        regime_aligned.values,
        alpha=0.4,
        color="red",
        label="Regime alta vol (cash)",
    )
    ax2.set_ylabel("Regime (1=alta vol)")
    ax2.set_xlabel("Data")
    ax2.set_title("Regime de Volatilidade")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
