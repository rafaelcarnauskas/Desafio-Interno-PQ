import pandas as pd

from interfaces import RiskFilter


def liquidity_filter(volume: pd.DataFrame, window: int = 20, min_adv: float = 500_000) -> RiskFilter:
    adv = volume.rolling(window, min_periods=1).mean()

    def _filter(weights: pd.DataFrame, ibov: pd.Series) -> pd.DataFrame:
        common_tickers = weights.columns.intersection(adv.columns)
        mask = adv.reindex(index=weights.index, columns=common_tickers) >= min_adv
        mask = mask.reindex(columns=weights.columns, fill_value=False)
        return weights.where(mask, 0.0)

    return _filter
