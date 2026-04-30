import pandas as pd
import yfinance as yf


ibov = yf.download('^BVSP', start='2010-01-01', end='2025-12-31', auto_adjust=False)

if isinstance(ibov.columns, pd.MultiIndex):
    ibov.columns = ibov.columns.get_level_values(0)  # remove o nível do ticker

ibov = ibov.reset_index()[['Date', 'Adj Close']]
ibov.to_csv('ibov_2010_2025.csv', index=False)