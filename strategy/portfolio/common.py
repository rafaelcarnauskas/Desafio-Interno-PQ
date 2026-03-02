import pandas as pd


def monthly_rebalance(prices: pd.DataFrame, scores: pd.DataFrame, n: int, ascending: bool = False) -> pd.DataFrame:
    """
    Scaffold de rebalanceamento mensal: seleciona top-n ativos por score,
    equal weight 1/n, forward-fill + shift(1) anti-lookahead.

    ascending=False → nlargest (momentum), ascending=True → nsmallest (mean reversion).
    """
    month_ends = pd.DatetimeIndex(
        [grp.index.max() for _, grp in prices.groupby(prices.index.to_period("M"))]
    )

    monthly_weights = []
    for date in month_ends:
        s = scores.loc[date].dropna()
        if len(s) >= n:
            selected = s.nsmallest(n).index if ascending else s.nlargest(n).index
            row = pd.Series(0.0, index=prices.columns)
            row[selected] = 1.0 / n
        else:
            row = pd.Series(0.0, index=prices.columns)
        monthly_weights.append((date, row))

    weights = pd.DataFrame(
        {d: w for d, w in monthly_weights},
    ).T
    weights.index = pd.DatetimeIndex([d for d, _ in monthly_weights])

    weights = weights.reindex(prices.index).ffill().shift(1).fillna(0.0)
    return weights
