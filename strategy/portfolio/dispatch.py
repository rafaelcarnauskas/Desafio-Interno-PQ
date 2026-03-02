import pandas as pd

from interfaces import Strategy, SubStrategy


def regime_dispatch_strategy(mapping: dict[int, SubStrategy]) -> Strategy:
    """
    Dispatcher: para cada dia, seleciona pesos da sub-estratégia correspondente ao regime.
    mapping: {regime_int: SubStrategy}, ex: {0: momentum, 1: momentum, 2: cash}
    """
    def strategy(prices: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
        # Pré-computa pesos de cada sub-estratégia
        sub_weights = {r: sub(prices) for r, sub in mapping.items()}

        # Para cada dia, pega os pesos da sub correspondente ao regime
        combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date in prices.index:
            r = regime.get(date, max(mapping.keys()))  # conservador: assume maior regime
            if r in sub_weights:
                combined.loc[date] = sub_weights[r].loc[date]

        return combined

    return strategy
