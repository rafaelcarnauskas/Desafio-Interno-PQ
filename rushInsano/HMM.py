from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import entropy as scipy_entropy

# ── 1. Carregar dados ────────────────────────────────────────────────────────
ibov = pd.read_csv("ibov_2010_2025.csv", skiprows=[1], parse_dates=["Date"])
ibov.set_index("Date", inplace=True)
ibov.sort_index(inplace=True)

daily_returns = ibov["Adj Close"].pct_change().dropna()

# ── 2. Features semanais ─────────────────────────────────────────────────────
def ewm_weekly_mean(series: pd.Series, span: float = 3.0) -> pd.Series:
    def _ewm(group):
        w = np.array([(1 - 1/span) ** (len(group) - 1 - i) for i in range(len(group))])
        w /= w.sum()
        return (group.values * w).sum()
    return series.groupby(pd.Grouper(freq="W")).apply(_ewm).dropna()

def build_features(daily: pd.Series, span: float = 3.0) -> pd.DataFrame:
    weekly_mean = ewm_weekly_mean(daily, span=span)
    weekly_std  = daily.groupby(pd.Grouper(freq="W")).std().dropna()
    idx = weekly_mean.index.intersection(weekly_std.index)
    df = pd.DataFrame({"ret_mean": weekly_mean[idx], "ret_std": weekly_std[idx]})
    return df.replace([np.inf, -np.inf], np.nan).dropna()

features_train = build_features(daily_returns.loc[:"2015-12-31"])
features_test  = build_features(daily_returns.loc["2016-01-01":])

scaler  = StandardScaler()
X_train = scaler.fit_transform(features_train.values)
X_test  = scaler.transform(features_test.values)

# ── 3. Treinar HMM ───────────────────────────────────────────────────────────
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
model.fit(X_train)
print(f"Convergiu: {model.monitor_.converged}  |  Iterações: {model.monitor_.iter}")

# ── 4. Probabilidades posteriores ────────────────────────────────────────────
# predict_proba retorna P(estado_k | observações até t) para cada semana
# Shape: (n_semanas, n_estados)
proba_train = model.predict_proba(X_train)
proba_test  = model.predict_proba(X_test)

states_train = model.predict(X_train)
states_test  = model.predict(X_test)

# ── 5. Métricas de incerteza ──────────────────────────────────────────────────

def uncertainty_metrics(proba: np.ndarray, n_states: int = 3) -> pd.DataFrame:
    """
    Calcula três métricas de incerteza a partir das probabilidades posteriores.

    - entropy_norm : entropia normalizada [0, 1]
        0 = certeza absoluta (ex: [1, 0, 0])
        1 = máxima incerteza (ex: [0.33, 0.33, 0.33])
        Fórmula: H = -Σ p_i * log(p_i) / log(n_estados)

    - margin : diferença entre o 1º e 2º estado mais prováveis [0, 1]
        1 = decisão clara   |   0 = empate perfeito entre dois estados
        Útil para detectar "dúvida entre dois estados específicos"

    - max_prob : probabilidade do estado vencedor [1/n, 1]
        Mais direto: abaixo de 0.5~0.6 o modelo está genuinamente indeciso
    """
    max_prob  = proba.max(axis=1)
    sorted_p  = np.sort(proba, axis=1)[:, ::-1]
    margin    = sorted_p[:, 0] - sorted_p[:, 1]

    # scipy_entropy usa base e por padrão; dividir por log(n) normaliza para [0,1]
    raw_entropy  = np.array([scipy_entropy(p) for p in proba])
    entropy_norm = raw_entropy / np.log(n_states)

    return pd.DataFrame({
        "entropy_norm": entropy_norm,
        "margin":       margin,
        "max_prob":     max_prob,
    })

uncertainty_test = uncertainty_metrics(proba_test, n_states=3)

# ── 6. Definir zona de incerteza ─────────────────────────────────────────────
# Limiar: entropy_norm > 0.5 significa que o modelo está "dividido".
# Equivale a max_prob < ~0.65 na prática. Ajuste conforme seu apetite de risco.
ENTROPY_THRESHOLD = 0.5   # <-- mexa aqui para calibrar

is_uncertain = uncertainty_test["entropy_norm"] > ENTROPY_THRESHOLD

print(f"\nSemanas com incerteza alta (entropia > {ENTROPY_THRESHOLD}): "
      f"{is_uncertain.sum()} de {len(is_uncertain)} "
      f"({is_uncertain.mean():.1%})")

# ── 7. Identificar estados por volatilidade ──────────────────────────────────
state_vol = {s: features_train["ret_std"].values[states_train == s].mean() for s in range(3)}
vol_order = np.argsort([state_vol[s] for s in range(3)])

color_map = {vol_order[0]: "#2ecc71", vol_order[1]: "#f39c12", vol_order[2]: "#e74c3c"}
label_map = {vol_order[0]: "Vol. Baixa",
             vol_order[1]: "Vol. Média",
             vol_order[2]: "Vol. Alta (crise)"}

# ── 8. Diagnóstico por estado ─────────────────────────────────────────────────
print("\n=== Diagnóstico por estado (treino) ===")
for s in range(3):
    mask = states_train == s
    print(f"  Estado {s} [{label_map[s]}]: "
          f"retorno={features_train['ret_mean'].values[mask].mean():.4%}  "
          f"vol={features_train['ret_std'].values[mask].mean():.4%}  "
          f"n={mask.sum()}")

# ── 9. Figura principal ───────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.2, 1.2, 0.8]})

idx = features_test.index

# --- subplot 1: retornos com estado + destaque de incerteza ---
ax1 = axes[0]
# Faixas cinzas onde o modelo está indeciso
for i, unc in enumerate(is_uncertain):
    if unc and i < len(idx) - 1:
        ax1.axvspan(idx[i], idx[i+1], color="silver", alpha=0.5, zorder=1)

ax1.plot(idx, features_test["ret_mean"].values, color="black", alpha=0.2, linewidth=0.8)
for s in range(3):
    mask = (states_test == s) & ~is_uncertain
    ax1.scatter(idx[mask], features_test["ret_mean"].values[mask],
                c=color_map[s], label=label_map[s], s=30, alpha=0.85, zorder=3)
# Pontos incertos em cinza escuro
ax1.scatter(idx[is_uncertain.values], features_test["ret_mean"].values[is_uncertain.values],
            c="dimgray", label="Incerto (não operar)", s=30, marker="x", zorder=4)

ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax1.set_ylabel("Retorno semanal (EWM)")
ax1.set_title(f"HMM — IBOV 2016-2025  |  Zona de incerteza: entropia normalizada > {ENTROPY_THRESHOLD}")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

# --- subplot 2: entropia normalizada ao longo do tempo ---
ax2 = axes[1]
ax2.fill_between(idx, uncertainty_test["entropy_norm"].values,
                 alpha=0.3, color="steelblue", label="Entropia norm.")
ax2.plot(idx, uncertainty_test["entropy_norm"].values,
         color="steelblue", linewidth=0.9)
ax2.axhline(ENTROPY_THRESHOLD, color="red", linewidth=1.2, linestyle="--",
            label=f"Limiar = {ENTROPY_THRESHOLD}")
ax2.set_ylabel("Entropia norm. [0–1]")
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.3)

# --- subplot 3: probabilidades empilhadas por estado ---
ax3 = axes[2]
bottom = np.zeros(len(proba_test))
for s in range(3):
    ax3.bar(idx, proba_test[:, s], bottom=bottom,
            color=color_map[s], label=label_map[s],
            width=6, alpha=0.85)
    bottom += proba_test[:, s]
ax3.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
ax3.set_ylabel("P(estado | dados)")
ax3.set_ylim(0, 1)
ax3.legend(fontsize=8, loc="lower left", ncol=3)
ax3.grid(True, alpha=0.2)

# --- subplot 4: margem (p1 - p2) ---
ax4 = axes[3]
ax4.fill_between(idx, uncertainty_test["margin"].values,
                 alpha=0.35, color="darkorange")
ax4.plot(idx, uncertainty_test["margin"].values,
         color="darkorange", linewidth=0.9, label="Margem (p₁ − p₂)")
ax4.axhline(0.2, color="red", linewidth=1, linestyle="--", label="Margem mínima sugerida")
ax4.set_ylabel("Margem")
ax4.set_ylim(0, 1.05)
ax4.set_xlabel("Data")
ax4.legend(fontsize=9, loc="upper right")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hmm_uncertainty.png", dpi=300)
plt.show()

# ── 10. Tabela resumo: distribuição de incerteza por estado ──────────────────
print("\n=== Incerteza por estado (teste) ===")
unc_df = pd.DataFrame({
    "estado":       states_test,
    "label":        [label_map[s] for s in states_test],
    "entropy_norm": uncertainty_test["entropy_norm"].values,
    "margin":       uncertainty_test["margin"].values,
    "max_prob":     uncertainty_test["max_prob"].values,
    "incerto":      is_uncertain.values,
}, index=idx)

for s in range(3):
    sub = unc_df[unc_df["estado"] == s]
    print(f"\n  {label_map[s]}  (n={len(sub)})")
    print(f"    Entropia média  : {sub['entropy_norm'].mean():.3f}")
    print(f"    Margem média    : {sub['margin'].mean():.3f}")
    print(f"    % semanas incertas: {sub['incerto'].mean():.1%}")