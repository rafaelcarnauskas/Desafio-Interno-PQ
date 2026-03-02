from enum import IntEnum

import numpy as np
import pandas as pd

from interfaces import RegimeSignal
from signals.common import log_returns


class RollingVolRegime(IntEnum):
    NORMAL = 0
    HIGH_VOL = 1


def rolling_vol_signal(window: int = 21, threshold: float = 0.20) -> RegimeSignal:
    """Sinal de regime baseado em volatilidade realizada do IBOV (0=normal, 1=alta vol)."""
    def signal(ibov: pd.Series) -> pd.Series:
        log_ret = log_returns(ibov)
        vol = log_ret.rolling(window).std() * np.sqrt(252)
        regime = pd.Series(
            np.where(vol > threshold, RollingVolRegime.HIGH_VOL, RollingVolRegime.NORMAL),
            index=ibov.index,
            name="regime",
        )
        return regime

    return signal
