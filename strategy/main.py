import matplotlib.pyplot as plt
import pandas as pd

from data import load_ibov, load_prices
from signals import hmm_vol_signal, HmmRegime
from portfolio import momentum_strategy, mean_reversion_strategy, cash_strategy, regime_dispatch_strategy
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

    TRAIN_END  = '2017-12-31'
    TEST_START = '2018-01-01'
    regime_signal = hmm_vol_signal(n_states=3, train_end=TRAIN_END)

    strategy = regime_dispatch_strategy({
        HmmRegime.LOW_VOL:  momentum_strategy(n=10, lookback=60),
        HmmRegime.MID_VOL:  mean_reversion_strategy(n=10, lookback=60),
        HmmRegime.HIGH_VOL: cash_strategy(),
    })
    risk_filters = []

    prices = prices.loc[TEST_START:]

    print("Executando pipeline...")
    result = run_pipeline(prices, ibov, regime_signal, strategy, risk_filters)

    print_metrics(result["metrics"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    eq = result["equity"]
    ibov_eq = result["ibov_equity"]

    ax1.plot(eq.index, eq.values, label="Estratégia Momentum/MeanRev + Regime", color="steelblue")
    ax1.plot(ibov_eq.index, ibov_eq.values, label="IBOV Buy & Hold", color="orange", alpha=0.8)
    ax1.set_ylabel("Equity (base 1)")
    ax1.set_title("Curvas de Equity — Estratégia vs IBOV (2018–2024, fora da amostra)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 3 estados de regime com cores distintas
    regime_aligned = result["regime"].reindex(eq.index)
    regime_colors = {HmmRegime.LOW_VOL: "green", HmmRegime.MID_VOL: "gold", HmmRegime.HIGH_VOL: "red"}
    regime_labels = {HmmRegime.LOW_VOL: "Low vol (momentum)", HmmRegime.MID_VOL: "Mid vol (mean reversion)", HmmRegime.HIGH_VOL: "High vol (cash)"}

    for state, color in regime_colors.items():
        mask = (regime_aligned == state).astype(float)
        ax2.fill_between(
            regime_aligned.index,
            mask,
            alpha=0.4,
            color=color,
            label=regime_labels[state],
        )

    ax2.set_ylabel("Regime")
    ax2.set_xlabel("Data")
    ax2.set_title("Regime de Volatilidade (3 estados HMM)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
