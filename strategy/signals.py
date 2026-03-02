import numpy as np
import pandas as pd

from interfaces import RegimeSignal


def rolling_vol_signal(window: int = 21, threshold: float = 0.20) -> RegimeSignal:
    """Sinal de regime baseado em volatilidade realizada do IBOV (0=normal, 1=alta vol)."""
    def signal(ibov: pd.Series) -> pd.Series:
        log_ret = np.log(ibov / ibov.shift(1))
        vol = log_ret.rolling(window).std() * np.sqrt(252)
        regime = (vol > threshold).astype(int)
        regime.name = "regime"
        return regime

    return signal
