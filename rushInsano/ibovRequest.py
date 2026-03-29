import pandas as pd
import yfinance as yf

# Baixar dados do índice IBOVESPA
ibov = yf.download('^BVSP', start='2010-01-01', end='2025-12-31', auto_adjust=False)

# Resetar índice para salvar Date como coluna
ibov.reset_index(inplace=True)

# Salvar em CSV
ibov.to_csv('ibov_2010_2025.csv', index=False)