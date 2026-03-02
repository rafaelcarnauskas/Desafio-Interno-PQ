import numpy as np
import pandas as pd

from interfaces import Strategy, SubStrategy


def momentum_strategy(n: int = 10, lookback: int = 60) -> SubStrategy:
    """
    Top-n por momentum trailing `lookback` dias, equal weight.
    Pesos gerados no último dia útil do mês, forward-fill + shift(1) para eliminar lookahead.
    """
    def strategy(prices: pd.DataFrame) -> pd.DataFrame:
        momentum = prices.pct_change(lookback)

        month_ends = pd.DatetimeIndex(
            [grp.index.max() for _, grp in prices.groupby(prices.index.to_period("M"))]
        )

        monthly_weights = []
        for date in month_ends:
            scores = momentum.loc[date].dropna()
            if len(scores) >= n:
                top = scores.nlargest(n).index
                row = pd.Series(0.0, index=prices.columns)
                row[top] = 1.0 / n
            else:
                row = pd.Series(0.0, index=prices.columns)
            monthly_weights.append((date, row))

        weights = pd.DataFrame(
            {d: w for d, w in monthly_weights},
        ).T
        weights.index = pd.DatetimeIndex([d for d, _ in monthly_weights])

        weights = weights.reindex(prices.index).ffill().shift(1).fillna(0.0)
        return weights

    return strategy


def cash_strategy() -> SubStrategy:
    """Retorna pesos zero (cash) para todos os dias."""
    def strategy(prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    return strategy


def mean_reversion_strategy(n: int = 10, lookback: int = 60) -> SubStrategy:
    """
    Top-n mais oversold por z-score de retornos trailing, equal weight.
    Z-score calculado contra janela lookback*4 para baseline estável.
    Rebalanceamento mensal + ffill + shift(1).
    """
    def strategy(prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change(lookback)
        zscore_window = lookback * 4
        rolling_mean = returns.rolling(zscore_window).mean()
        rolling_std = returns.rolling(zscore_window).std()
        zscore = (returns - rolling_mean) / rolling_std.replace(0, np.nan)

        month_ends = pd.DatetimeIndex(
            [grp.index.max() for _, grp in prices.groupby(prices.index.to_period("M"))]
        )

        monthly_weights = []
        for date in month_ends:
            scores = zscore.loc[date].dropna()
            if len(scores) >= n:
                bottom = scores.nsmallest(n).index
                row = pd.Series(0.0, index=prices.columns)
                row[bottom] = 1.0 / n
            else:
                row = pd.Series(0.0, index=prices.columns)
            monthly_weights.append((date, row))

        weights = pd.DataFrame(
            {d: w for d, w in monthly_weights},
        ).T
        weights.index = pd.DatetimeIndex([d for d, _ in monthly_weights])

        weights = weights.reindex(prices.index).ffill().shift(1).fillna(0.0)
        return weights

    return strategy


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
