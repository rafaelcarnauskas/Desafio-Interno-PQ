import pandas as pd


def get_weights(
    prices: pd.DataFrame,
    regime: pd.Series,
    n: int = 10,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Pesos mensais baseados em momentum cross-sectional e regime de volatilidade.

    Regime 0 (normal): top-n ações por retorno trailing `lookback` dias, igual weight.
    Regime 1 (alta vol): cash (todos os pesos = 0).

    Os pesos são gerados no último dia útil de cada mês, depois:
    - forward-fill para dias intermediários
    - shift(1) para eliminar lookahead
    """
    momentum = prices.pct_change(lookback)

    # Último dia de negociação de cada mês (via groupby em período mensal)
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

    # Alinhar com o índice de preços, forward-fill e shift(1)
    weights = weights.reindex(prices.index).ffill().shift(1).fillna(0.0)
    return weights
