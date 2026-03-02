import pandas as pd

import backtest
from interfaces import RegimeSignal, RiskFilter, Strategy


def run_pipeline(
    prices: pd.DataFrame,
    ibov: pd.Series,
    regime_signal: RegimeSignal,
    strategy: Strategy,
    risk_filters: list[RiskFilter] | None = None,
) -> dict:
    """Executa o pipeline completo e retorna o resultado do backtest com 'regime' incluído."""
    regime = regime_signal(ibov)
    weights = strategy(prices, regime)
    for f in risk_filters or []:
        weights = f(weights, ibov)
    result = backtest.run(weights, prices, ibov)
    result["regime"] = regime
    return result
