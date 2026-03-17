from __future__ import annotations

import numpy as np
import pandas as pd

from interfaces import RiskFilter
from signals.common import log_returns


def vov_filter(
    vol_window: int = 20,
    vov_window: int = 60,
    train_end: str = "2017-12-31",
    threshold_quantile: float = 0.80,
    reduction: float = 0.5,
    annualize_vol: bool = True,
) -> RiskFilter:
    """
    Risk filter baseado em Volatility of Volatility (VoV) histórica do IBOV.

    Ideia:
    1) calcula log-retornos do IBOV
    2) calcula volatilidade rolling
    3) calcula a volatilidade dessa volatilidade (VoV)
    4) se VoV > threshold (estimado no treino), reduz a exposição da carteira

    Parâmetros
    ----------
    vol_window : int
        Janela da volatilidade rolling (ex.: 20 dias).
    vov_window : int
        Janela da volatilidade da volatilidade (ex.: 60 dias).
    train_end : str
        Data final do período de treino usada para calibrar o threshold
        sem lookahead.
    threshold_quantile : float
        Quantil da VoV no treino usado como threshold.
        Ex.: 0.80 = percentil 80.
    reduction : float
        Multiplicador aplicado aos pesos quando VoV está alta.
        Ex.: 0.5 reduz a exposição pela metade.
    annualize_vol : bool
        Se True, anualiza a volatilidade rolling por sqrt(252).
        Não muda a lógica do filtro, apenas a escala da série.

    Retorno
    -------
    RiskFilter
        Função compatível com a interface do projeto:
        (weights: pd.DataFrame, ibov: pd.Series) -> pd.DataFrame
    """
    if vol_window < 2:
        raise ValueError("vol_window deve ser >= 2")
    if vov_window < 2:
        raise ValueError("vov_window deve ser >= 2")
    if not (0.0 < threshold_quantile < 1.0):
        raise ValueError("threshold_quantile deve estar entre 0 e 1")
    if not (0.0 <= reduction <= 1.0):
        raise ValueError("reduction deve estar entre 0 e 1")

    def filter_fn(weights: pd.DataFrame, ibov: pd.Series) -> pd.DataFrame:
        # 1) log-retornos do IBOV
        ret = log_returns(ibov).dropna()

        # 2) volatilidade rolling histórica
        vol = ret.rolling(vol_window, min_periods=vol_window).std()
        if annualize_vol:
            vol = vol * np.sqrt(252)

        # 3) volatility of volatility
        vov = vol.rolling(vov_window, min_periods=vov_window).std()

        # 4) threshold calibrado APENAS no treino
        vov_train = vov.loc[:train_end].dropna()
        if vov_train.empty:
            raise ValueError(
                "Não há dados suficientes no período de treino para calibrar o threshold do VoV. "
                "Verifique train_end, vol_window e vov_window."
            )

        threshold = float(vov_train.quantile(threshold_quantile))

        # 5) multiplicador diário de risco
        #    VoV acima do threshold -> reduz exposição
        risk_mult = pd.Series(
            np.where(vov > threshold, reduction, 1.0),
            index=vov.index,
            name="vov_multiplier",
        )

        # Alinhar ao índice dos weights
        risk_mult = (
            risk_mult.reindex(weights.index)
            .ffill()      # carrega o último estado conhecido
            .fillna(1.0)  # no começo da amostra, não reduz
        )

        # 6) aplicar multiplicador em todos os ativos da carteira
        adjusted = weights.mul(risk_mult, axis=0)

        return adjusted

    return filter_fn
