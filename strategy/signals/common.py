import numpy as np
import pandas as pd


def log_returns(series: pd.Series) -> pd.Series:
    """Retornos logarítmicos da série."""
    return np.log(series / series.shift(1))
