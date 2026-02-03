import pandas as pd
import yfinance as yf

df = pd.read_csv("tickers.csv")

tickers = df["ticker"].tolist()# salva os nomes dos ativos em uma lista

df = pd.DataFrame(tickers)
df = yf.download(tickers, start='2010-01-01', end='2025-12-31', group_by='ticker', threads=False, auto_adjust=False)
# obs: os tickers são as colunas e as datas são as linhas
#Aqui nem todos os downloads foram feitos, talvez refazer o loop para os valores vazios arrume isso
"""
incomplete_data = df.columns[df.isna().all()].tolist()
df_novos = yf.download(incomplete_data, start='2010-01-01', end='2025-12-31', group_by='ticker', threads=False, auto_adjust=False)

df.update(df_novos)"""

df.to_csv('b3-2010-2025.csv')