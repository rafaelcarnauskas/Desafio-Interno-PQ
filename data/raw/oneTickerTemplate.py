import yfinance as yf

dados = yf.download('BPAC3.SA', start='2010-01-01', end='2025-12-31', group_by='ticker', threads=False, auto_adjust=False)

import pandas as pd

df = pd.DataFrame(dados)
# Remove timezone information from datetime index for Excel compatibility
df.index = df.index.tz_localize(None)
df.to_excel('teste.xlsx', index=True)  # Keep index (Date)