from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carregar CSV ignorando a linha 1 (onde estão os textos '^BVSP')
ibov = pd.read_csv("ibov_2010_2025.csv", skiprows=[1], parse_dates=['Date'])

# Definir como index e garantir que está ordenado (monotônico)
ibov.set_index('Date', inplace=True)
ibov.sort_index(inplace=True) # <- Isso previne o KeyError


# Filtrar dados: treino até 2015, teste a partir de 2016
ibov_train = ibov.loc[:'2015-12-31']
ibov_test = ibov.loc['2016-01-01':]

# Calcular retornos
returns_train = ibov_train['Adj Close'].pct_change()
#np.log(df['Adj Close'] / df['Adj Close'].shift(1)) se quiser log retorno
returns_train = returns_train.dropna()

returns_test = ibov_test['Adj Close'].pct_change()
#np.log(df['Adj Close'] / df['Adj Close'].shift(1)) se quiser log retorno
returns_test = returns_test.dropna()

# extrair média SIMPLES de retorno mensais
monthly_returns = returns_test.resample('M').mean()
print("\n=== Média de Retornos Mensais ===")
print(monthly_returns)
print(f"\nTotal de meses: {len(monthly_returns)}")

# Treinar modelo HMM
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
model.fit(returns_train.values.reshape(-1, 1))

# Prever estados no treinamento
states_train = model.predict(monthly_returns.values.reshape(-1, 1))

# Prever estados no teste
states_test = model.predict(returns_test.values.reshape(-1, 1))

print(model.monitor_.converged)
print(model.monitor_.iter) #para ver em quantas iterações convergiu

print(states_train)


#//////////////////////////////////// Área de plot do treino

# Criar gráfico dos estados no período de treino
fig, ax = plt.subplots(figsize=(14, 6))

# Plot dos retornos
ax.plot(monthly_returns.index, monthly_returns.values, 'gray', alpha=0.5, label='Retornos')

# Colorir os states
colors = ['red', 'green', 'blue']
state_labels = {0: 'Estado 0', 1: 'Estado 1', 2: 'Estado 2'}

for state in range(model.n_components):
    mask = states_train == state
    ax.scatter(monthly_returns.index[mask], monthly_returns.values[mask], 
              c=colors[state], label=state_labels[state], s=20, alpha=0.6)

ax.set_xlabel('Data')
ax.set_ylabel('Retornos')
ax.set_title('Estados HMM - IBOV (2016-2025)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hmm_states.png', dpi=300)
plt.show()