import numpy as np
import pandas as pd


def get_regime(ibov: pd.Series, window: int = 21, threshold: float = 0.20) -> pd.Series:
    """
    Retorna série de regimes com base na volatilidade realizada do IBOV.
    0 = normal, 1 = alta volatilidade.
    """
    log_ret = np.log(ibov / ibov.shift(1))
    vol = log_ret.rolling(window).std() * np.sqrt(252)
    regime = (vol > threshold).astype(int)
    regime.name = "regime"
    return regime
