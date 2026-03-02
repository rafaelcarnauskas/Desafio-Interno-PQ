from enum import IntEnum

import numpy as np
import pandas as pd
from scipy.stats import norm

from interfaces import RegimeSignal


class HmmRegime(IntEnum):
    LOW_VOL = 0
    MID_VOL = 1
    HIGH_VOL = 2


def rolling_vol_signal(window: int = 21, threshold: float = 0.20) -> RegimeSignal:
    """Sinal de regime baseado em volatilidade realizada do IBOV (0=normal, 1=alta vol)."""
    def signal(ibov: pd.Series) -> pd.Series:
        log_ret = np.log(ibov / ibov.shift(1))
        vol = log_ret.rolling(window).std() * np.sqrt(252)
        regime = pd.Series(
            np.where(vol > threshold, HmmRegime.HIGH_VOL, HmmRegime.LOW_VOL),
            index=ibov.index,
            name="regime",
        )
        return regime

    return signal


def _scaled_forward(X, pi, A, mu, sigma):
    T = len(X)
    N = len(pi)
    alpha_hat = np.zeros((T, N))
    scales = np.zeros(T)

    B0 = norm.pdf(X[0], mu, sigma)
    alpha_hat[0] = pi * B0
    scales[0] = alpha_hat[0].sum()
    alpha_hat[0] /= scales[0]

    for t in range(1, T):
        Bt = norm.pdf(X[t], mu, sigma)
        alpha_hat[t] = Bt * (alpha_hat[t - 1] @ A)
        scales[t] = alpha_hat[t].sum()
        alpha_hat[t] /= scales[t]

    return alpha_hat, scales


def _scaled_backward(X, A, mu, sigma, scales):
    T = len(X)
    N = A.shape[0]
    beta_hat = np.zeros((T, N))
    beta_hat[T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        Bt1 = norm.pdf(X[t + 1], mu, sigma)
        beta_hat[t] = (A * Bt1) @ beta_hat[t + 1] / scales[t + 1]

    return beta_hat


def _fit_hmm(X, n_states, random_state):
    rng = np.random.default_rng(random_state)
    N = n_states

    pi = np.ones(N) / N
    A = rng.random((N, N))
    A /= A.sum(axis=1, keepdims=True)
    mu = np.percentile(X, np.linspace(20, 80, N))
    sigma = np.full(N, X.std())

    prev_log_lik = -np.inf

    for _ in range(100):
        alpha_hat, scales = _scaled_forward(X, pi, A, mu, sigma)
        beta_hat = _scaled_backward(X, A, mu, sigma, scales)

        gamma = alpha_hat * beta_hat
        gamma /= gamma.sum(axis=1, keepdims=True)

        T = len(X)
        xi_sum = np.zeros((N, N))
        for t in range(T - 1):
            Bt1 = norm.pdf(X[t + 1], mu, sigma)
            xi_t = alpha_hat[t, :, None] * A * Bt1[None, :] * beta_hat[t + 1, None, :]
            xi_sum += xi_t / scales[t + 1]

        pi = gamma[0]
        A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        gamma_sum = gamma.sum(axis=0)
        mu = (gamma * X[:, None]).sum(axis=0) / gamma_sum
        sigma = np.sqrt((gamma * (X[:, None] - mu[None, :]) ** 2).sum(axis=0) / gamma_sum)
        sigma = np.maximum(sigma, 1e-6)

        log_lik = np.log(scales).sum()
        if abs(log_lik - prev_log_lik) < 1e-6:
            break
        prev_log_lik = log_lik

    return pi, A, mu, sigma


def hmm_vol_signal(n_states: int = 3, train_end: str = '2017-12-31', random_state: int = 42) -> RegimeSignal:
    """Sinal de regime 3 estados (0=low, 1=mid, 2=high vol) via HMM; decodificação causal."""
    def signal(ibov: pd.Series) -> pd.Series:
        log_ret = np.log(ibov / ibov.shift(1)).dropna()

        X_train = log_ret.loc[:train_end].values
        pi, A, mu, sigma = _fit_hmm(X_train, n_states, random_state)

        X_full = log_ret.values
        alpha_hat, _ = _scaled_forward(X_full, pi, A, mu, sigma)
        states = np.argmax(alpha_hat, axis=1)

        # Remapear estados para ordem crescente de sigma: 0=low, 1=mid, 2=high vol
        order = np.argsort(sigma)
        state_map = np.zeros(n_states, dtype=int)
        for new_label, old_label in enumerate(order):
            state_map[old_label] = new_label
        regime = state_map[states]

        return (
            pd.Series(regime, index=log_ret.index, name='regime')
            .reindex(ibov.index)
            .ffill()
            .fillna(0)
            .astype(int)
        )

    return signal
