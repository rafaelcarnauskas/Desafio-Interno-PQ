import pandas as pd
import yfinance as yf

tickers = pd.read_csv('tickers.csv')['ticker'].dropna().unique().tolist()
tickers = [t for t in tickers if t not in {'^BVSP', 'IBOV', 'IBOV11.SA'}]

raw = yf.download(
    tickers,
    start='2010-01-01',
    end='2025-12-31',
    auto_adjust=False
)

data = raw['Adj Close'].copy()

for col in data.columns:

    s = data[col].copy()

    ret = s.pct_change(fill_method=None)

    s = s.mask(ret.abs() > 10)

    median = s.rolling(60, min_periods=20).median()

    ratio = s / median

    s = s.mask((ratio < 0.1) | (ratio > 10))

    data[col] = s

data.reset_index().to_csv("b3-2010-2025.csv", index=False)