from typing import Callable

import pandas as pd

RegimeSignal = Callable[[pd.Series], pd.Series]
Strategy = Callable[[pd.DataFrame, pd.Series], pd.DataFrame]
RiskFilter = Callable[[pd.DataFrame, pd.Series], pd.DataFrame]
