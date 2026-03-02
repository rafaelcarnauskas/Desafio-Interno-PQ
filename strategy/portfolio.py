import pandas as pd

from interfaces import Strategy


def momentum_strategy(n: int = 10, lookback: int = 60) -> Strategy:
    """
    Top-n por momentum trailing `lookback` dias, igual weight em regime normal.
    Regime 1 (alta vol): cash (pesos = 0). Pesos gerados no último dia útil do mês,
    forward-fill + shift(1) para eliminar lookahead.
    """
    def strategy(prices: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
        momentum = prices.pct_change(lookback)

        month_ends = pd.DatetimeIndex(
            [grp.index.max() for _, grp in prices.groupby(prices.index.to_period("M"))]
        )

        monthly_weights = []
        for date in month_ends:
            r = regime.get(date, 1)  # conservador: assume alta vol se não tiver dado
            if r == 1:
                row = pd.Series(0.0, index=prices.columns)
            else:
                scores = momentum.loc[date].dropna()
                top = scores.nlargest(n).index
                row = pd.Series(0.0, index=prices.columns)
                row[top] = 1.0 / n
            monthly_weights.append((date, row))

        weights = pd.DataFrame(
            {d: w for d, w in monthly_weights},
        ).T
        weights.index = pd.DatetimeIndex([d for d, _ in monthly_weights])

        weights = weights.reindex(prices.index).ffill().shift(1).fillna(0.0)
        return weights

    return strategy
