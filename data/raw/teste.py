import yfinance as yf

dados = yf.download(['AAPL'], start='2020-01-01')

print(dados['Close'].head())

import pandas as pd

df = pd.DataFrame(dados)
df.to_excel('teste.xlsx', index=True)  # Keep index (Date)