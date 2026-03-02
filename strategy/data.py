import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PRICES_FILE = DATA_DIR / "precos_b3_202010-2024_adjclose.csv"
IBOV_FILE = DATA_DIR / "ibov_2010_2024.csv"
OHLCV_FILE = DATA_DIR / "b3-2010-2025.csv"

MAX_NAN_FRAC = 0.20


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_FILE, index_col="Date", parse_dates=True)
    df = df.sort_index()
    # Drop tickers com mais de 20% de NaN
    nan_frac = df.isna().mean()
    df = df.loc[:, nan_frac <= MAX_NAN_FRAC]
    return df


def load_volume() -> pd.DataFrame:
    df = pd.read_csv(OHLCV_FILE, header=[0, 1], index_col=0, parse_dates=True)
    vol = df.xs("Volume", axis=1, level=1)
    vol = vol.sort_index()
    return vol


def load_ibov() -> pd.Series:
    # Arquivo tem 2 linhas de cabeçalho: ignorar a segunda (linha 1, com ",^BVSP")
    df = pd.read_csv(IBOV_FILE, skiprows=[1], index_col="Date", parse_dates=True)
    s = df["Close"].sort_index()
    return s
