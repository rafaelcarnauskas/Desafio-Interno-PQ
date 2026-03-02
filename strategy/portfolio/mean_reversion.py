import numpy as np
import pandas as pd

from interfaces import SubStrategy
from portfolio.common import monthly_rebalance


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
        scores = (returns - rolling_mean) / rolling_std.replace(0, np.nan)
        return monthly_rebalance(prices, scores, n, ascending=True)

    return strategy
