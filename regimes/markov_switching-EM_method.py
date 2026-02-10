# Implement 3-state HMM from scratch
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import yfinance as yf

# Baixa os dados completos
data = yf.download("PETR3.SA", start="2010-01-01", end="2025-12-31")
# Extrai apenas os preços de fechamento
df = pd.DataFrame(data['Close'])

# Vou estudar primeiro os estados das ações ordinárias da petrobrás

np.random.seed(42)

# Converte a coluna Close para numérico (remove erros de formatação)
#df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

df['Returns'] = df.pct_change()

# Prepare the data
X = df['Returns'].values

# Initialize parameters
N = 3  # number of states
T = len(X)  # number of observations

# Random initialization
pi = np.ones(N) / N  # initial state probabilities
A = np.random.rand(N, N)  # transition matrix 
#OBS: Aqui a matriz de transição começa aleatória, mas um jeito de deixar isso mais rebuscado é usar informação de treino para modelar melhor essa matriz e começar mais avançado
A = A / A.sum(axis=1)[:, np.newaxis] # Normalização da coluna (probabilidades devem somar 100%, então divide a célula pela soma da coluna)

# Initialize emission parameters (mean and std for each state)
mu = np.array([-0.01, 0, 0.01])  # initial guesses for means
sigma = np.array([0.01, 0.02, 0.03])  # initial guesses for standard deviations

def forward_pass(X, pi, A, mu, sigma):
    alpha = np.zeros((T, N))
    # Initialize first timestep
    for j in range(N):
        alpha[0,j] = pi[j] * norm.pdf(X[0], mu[j], sigma[j]) 
        #Atualiza a  inicial do estado multiplicando pela densidade de probabilidade da sua curva normal (considerando media e desv atuais)
    
    # Forward recursion: propaga probabilidades para frente no tempo
    # Para cada momento t, calcula a probabilidade de estar em cada estado j
    for t in range(1, T):
        for j in range(N):
            # alpha[t,j] = P(observar X[0]...X[t] E estar no estado j no tempo t)
            # = P(emitir X[t] | estado j) × P(chegar no estado j vindo de qualquer estado anterior)
            # = norm.pdf(...) × soma[P(estar em i no t-1) × P(transição de i para j)]
            alpha[t,j] = norm.pdf(X[t], mu[j], sigma[j]) * np.sum(alpha[t-1,:] * A[:,j])
        # Atualiza o resto das probabilidades ao longo do tempo
        # Acumula a soma dos produtos das probabilidades de transicionar para o estado anteriormente (alpha[t-1, :]) com a probabilidade de transicionar pro estado (A[:, j])
            
    return alpha

def backward_pass(X, A, mu, sigma):
    beta = np.zeros((T, N))
    # Inicializa o último momento: não há observações futuras, então probabilidade = 1
    beta[T-1,:] = 1
    
    # Backward recursion: propaga probabilidades de trás para frente no tempo
    # Para cada momento t (do penúltimo até o primeiro), calcula a prob. das observações futuras
    for t in range(T-2, -1, -1):  # De T-2 até 0, indo de trás pra frente
        for i in range(N):
            # beta[t,i] = P(observar X[t+1]...X[T-1] | estar no estado i no tempo t)
            # = soma sobre todos estados futuros j de:
            #   [P(transição i→j) × P(emitir X[t+1] | estado j) × P(observações após t+1 | estado j)]
            beta[t,i] = np.sum([A[i,j] * norm.pdf(X[t+1], mu[j], sigma[j]) * beta[t+1,j] for j in range(N)])
            
    return beta

# Run EM algorithm
max_iter = 100
prev_log_likelihood = -np.inf

for iteration in range(max_iter):
    # E-step
    alpha = forward_pass(X, pi, A, mu, sigma)
    beta = backward_pass(X, A, mu, sigma)
    
    # Calculate gamma (state probabilities)
    gamma = alpha * beta
    gamma = gamma / gamma.sum(axis=1)[:,np.newaxis]
    
    # Calculate xi (transition probabilities)
    xi = np.zeros((T-1, N, N))
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                xi[t,i,j] = alpha[t,i] * A[i,j] * norm.pdf(X[t+1], mu[j], sigma[j]) * beta[t+1,j]
        xi[t] = xi[t] / xi[t].sum()
    
    # M-step
    # Update initial probabilities
    pi = gamma[0]
    
    # Update transition matrix
    A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:,np.newaxis]
    
    # Update emission parameters
    for j in range(N):
        mu[j] = np.sum(gamma[:,j] * X) / np.sum(gamma[:,j])
        sigma[j] = np.sqrt(np.sum(gamma[:,j] * (X - mu[j])**2) / np.sum(gamma[:,j]))
    
    # Check convergence
    log_likelihood = np.sum(np.log(np.sum(alpha * beta, axis=1)))
    if abs(log_likelihood - prev_log_likelihood) < 1e-6:
        break
    prev_log_likelihood = log_likelihood

# Get the hidden states
hidden_states = np.argmax(gamma, axis=1)

# Map numeric states to latent state labels
state_volatilities = np.array([np.std(X[hidden_states == i]) for i in range(N)])
state_order = np.argsort(state_volatilities)
regime_map = {
    state_order[0]: 'Latent State 1',
    state_order[1]: 'Latent State 2',
    state_order[2]: 'Latent State 3'
}
regimes = [regime_map[state] for state in hidden_states]

# Calculate regime statistics
regime_stats = {}
for regime in ['Latent State 1', 'Latent State 2', 'Latent State 3']:
    mask = [r == regime for r in regimes]
    returns = df.loc[mask, 'Returns']
    
    regime_stats[regime] = {
        'mean': returns.mean(),
        'std': returns.std(),
        'count': len(returns)
    }

# Print results
print("\nTransition Probabilities Matrix:")
print("From/To      Latent State 1    Latent State 2    Latent State 3")
states = ['Latent State 1', 'Latent State 2', 'Latent State 3']
for i, state in enumerate(states):
    print(f"{state:15} {A[i,0]:.3f}         {A[i,1]:.3f}          {A[i,2]:.3f}")

print("\nLatent State Statistics (Daily):")
for regime, stats in regime_stats.items():
    print(f"\n{regime}:")
    print(f"Mean Return: {stats['mean']*100:.2f}%")
    print(f"Std Dev: {stats['std']*100:.2f}%")
    print(f"Number of Days: {stats['count']}")

# Plot regime distributions
x = np.linspace(-0.1, 0.1, 1000)
fig = go.Figure()

for regime in ['Latent State 1', 'Latent State 2', 'Latent State 3']:
    mu_r = regime_stats[regime]['mean']
    sigma_r = regime_stats[regime]['std']
    y = 1/(sigma_r * np.sqrt(2 * np.pi)) * np.exp(-(x - mu_r)**2 / (2 * sigma_r**2))
    
    fig.add_trace(
        go.Scatter(
            x=x*100,
            y=y,
            name=regime,
            line=dict(color=f'rgb({50 + int(regime[-1])*50}, {100 + int(regime[-1])*50}, {150 + int(regime[-1])*50})')
        )
    )

fig.update_layout(
    title='Return Distributions by Latent State (HMM)',
    xaxis_title='Daily Return (%)',
    yaxis_title='Density',
    height=500,
    width=900,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    )
)

fig.show()
