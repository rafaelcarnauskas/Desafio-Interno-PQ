import pandas as pd

from interfaces import SubStrategy
from portfolio.common import monthly_rebalance


def momentum_strategy(n: int = 10, lookback: int = 60) -> SubStrategy:
    """
    Top-n por momentum trailing `lookback` dias, equal weight.
    Pesos gerados no último dia útil do mês, forward-fill + shift(1) para eliminar lookahead.
    """
    def strategy(prices: pd.DataFrame) -> pd.DataFrame:
        scores = prices.pct_change(lookback)
        return monthly_rebalance(prices, scores, n, ascending=False)

    return strategy
