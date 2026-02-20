import pandas as pd

df = pd.read_csv("ibov_2010_2024.csv")

returns = df.pct_change().dropna()

RoR = returns.pct_change() # Return of return

