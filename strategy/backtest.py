import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def run(weights: pd.DataFrame, prices: pd.DataFrame, ibov: pd.Series) -> dict:
    """
    Executa backtest e retorna métricas + curva de equity.

    Parâmetros
    ----------
    weights : pd.DataFrame
        Pesos diários (já deslocados 1 dia, sem lookahead).
    prices : pd.DataFrame
        Preços ajustados das ações.
    ibov : pd.Series
        Série do IBOV para benchmark.

    Retorno
    -------
    dict com:
        equity       - curva de equity normalizada (começa em 1)
        ibov_equity  - curva do IBOV normalizada
        metrics      - dict com métricas de ambos
    """
    rets = prices.pct_change()

    # Contribuição de cada ativo e retorno diário da estratégia
    contributions = weights * rets
    port_ret = contributions.sum(axis=1)

    # Alinhar com período comum
    common = port_ret.index.intersection(ibov.index)
    port_ret = port_ret.loc[common]
    ibov_ret = ibov.loc[common].pct_change()

    equity = (1 + port_ret).cumprod()
    ibov_equity = (1 + ibov_ret).cumprod()

    def metrics(ret: pd.Series, eq: pd.Series, name: str) -> dict:
        n_years = len(ret) / 252
        total = float(eq.iloc[-1] - 1)
        ann = float((1 + total) ** (1 / n_years) - 1)
        sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
        mdd = _max_drawdown(eq)
        return {
            f"{name}_total_return": total,
            f"{name}_ann_return": ann,
            f"{name}_sharpe": sharpe,
            f"{name}_max_drawdown": mdd,
        }

    m = {}
    m.update(metrics(port_ret, equity, "strategy"))
    m.update(metrics(ibov_ret.dropna(), ibov_equity.dropna(), "benchmark"))

    return {
        "equity": equity,
        "ibov_equity": ibov_equity,
        "port_ret": port_ret,
        "metrics": m,
        "weights": weights,
        "contributions": contributions,
    }
