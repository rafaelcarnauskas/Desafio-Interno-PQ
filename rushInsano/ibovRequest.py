import pandas as pd
import yfinance as yf

ibov = yf.download('^BVSP', start='2010-01-01', end='2025-12-31', auto_adjust=False)

ibov.to_csv('ibov_2010_2025.csv')