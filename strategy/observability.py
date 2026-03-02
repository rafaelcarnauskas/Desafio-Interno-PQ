import os

import pandas as pd

from signals import HmmRegime


_REGIME_NAMES = {
    HmmRegime.LOW_VOL: "Low vol",
    HmmRegime.MID_VOL: "Mid vol",
    HmmRegime.HIGH_VOL: "High vol",
}


def inspect_date(result: dict, date_str: str) -> None:
    """Imprime posições, contribuições e regime para uma data específica."""
    date = pd.Timestamp(date_str)

    regime = result["regime"]
    weights = result["weights"]
    contributions = result["contributions"]
    port_ret = result["port_ret"]

    if date not in weights.index:
        print(f"Data {date_str} não encontrada no período do backtest.")
        return

    regime_val = regime.reindex(weights.index).loc[date]
    regime_name = _REGIME_NAMES.get(HmmRegime(int(regime_val)), str(regime_val))

    w = weights.loc[date]
    c = contributions.loc[date]
    held = w[w != 0].sort_values(ascending=False)

    print(f"\n{'='*60}")
    print(f"  Data: {date_str}  |  Regime: {regime_name}")
    print(f"{'='*60}")

    if held.empty:
        print("  Sem posições (cash)")
    else:
        print(f"  {'Ativo':<14} {'Peso':>10} {'Contribuição':>14}")
        print(f"  {'-'*38}")
        for ticker in held.index:
            print(f"  {ticker:<14} {held[ticker]:>10.2%} {c[ticker]:>14.4%}")

    total = port_ret.loc[date]
    print(f"\n  Retorno total do dia: {total:.4%}")
    print()


def export_csv(result: dict, output_dir: str = "output") -> None:
    """Exporta weights, contributions e regime para CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    result["weights"].to_csv(os.path.join(output_dir, "weights.csv"))
    result["contributions"].to_csv(os.path.join(output_dir, "contributions.csv"))

    regime = result["regime"]
    regime.to_csv(os.path.join(output_dir, "regime.csv"), header=True)

    print(f"CSVs exportados em {output_dir}/")
