import pandas as pd

from interfaces import SubStrategy


def cash_strategy() -> SubStrategy:
    """Retorna pesos zero (cash) para todos os dias."""
    def strategy(prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    return strategy
