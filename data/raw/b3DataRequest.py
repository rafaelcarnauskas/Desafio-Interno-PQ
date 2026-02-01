import pandas as pd
import yfinance as yf

df = pd.read_csv("precos_b3_202010-2024_adjclose.csv")

tickers = df.columns.tolist()# salva os nomes dos ativos em uma lista
tickers.pop(0) # remove a coluna data

df = pd.DataFrame(tickers)
df = yf.download(tickers, start='2010-01-01', end='2025-12-31', group_by='ticker', threads=False, auto_adjust=False)

df.to_excel('b3-2010-2025.xlsx')

#Aqui nem todos os downloads foram feitos, talvez refazer o loop para os valores vazios arrume isso