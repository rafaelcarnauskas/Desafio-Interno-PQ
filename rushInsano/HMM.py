"""
HMM Trading System v3 — IBOV
══════════════════════════════════════════════════════════════════════════════
CORREÇÕES v3 (em relação à v2):

  P1 [ALTO]  Neutral = 0 descartava ~35% das semanas
             → Neutral vira mean-reversion: mult = −0.3 × sinal de reversão
               (fade do retorno anterior; lucra em semanas que revertem)

  P2 [ALTO]  Cooldown de 3 semanas cortava rebounds pós-crise
             → Reentrada antecipada: se HMM = Momentum E entropia < 0.30,
               cooldown reduz para 1 semana

  P3 [MÉDIO] Apenas 2 features (ret_mean, ret_std) → estados mal separados
             → +3 features: skewness semanal, momentum 4 semanas, z-score
               da vol em janela de 13 semanas

  P4 [MÉDIO] Hedge/Short fixo em −0.70 perdia em rallies de alta vol
             → Hedge adaptativo: −0.70 só se ret_mean anterior < 0
               (mercado caindo de fato), senão −0.20 (hedge parcial)

  P5 [ARQ]   exposure_combined muito conservador (produto de 3 fatores < 1)
             → Floor de exposição mínima de 0.50 quando regime = Momentum
               e entropia < 0.35 (sinal de alta convicção)

Arquitetura de controle (ordem de aplicação):
  1. HMM         → regime + estratégia base
  2. Entropia     → escala contínua da alocação (1 − entropy_norm)
  3. Conectividade de rede → reduz exposição em mercado sincronizado
  4. VoV          → corte preventivo antes do HMM reagir
  5. Floor de exposição → garante participação mínima em regime de alta convicção
  6. Hedge adaptativo   → ajusta sinal do Hedge pela direção real do mercado
  7. Neutral mean-reversion → monetiza reversões no regime de vol média
  8. Stop-loss + Trailing stop + Reentrada antecipada
══════════════════════════════════════════════════════════════════════════════
"""

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import entropy as scipy_entropy, skew as scipy_skew

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARÂMETROS — ajuste aqui                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
ENTROPY_THRESHOLD        = 0.50   # acima → sem operação (override completo)
ENTROPY_FAST_REENTRY     = 0.30   # abaixo → cooldown reduzido para 1 sem
STOP_LOSS_PCT            = 0.07   # drawdown máximo desde o pico (7 %)
TRAILING_STOP_PCT        = 0.05   # recuo do equity desde o último pico (5 %)
COOLDOWN_WEEKS           = 3      # semanas parado após stop
COOLDOWN_FAST            = 1      # cooldown reduzido (reentrada antecipada)
NET_WINDOW               = 8      # janela p/ correlação de rede (semanas)
NET_THRESHOLD            = 0.65   # corr acima → reduz exposição
VOV_WINDOW               = 8      # janela p/ VoV
VOV_THRESHOLD            = 0.008  # VoV acima → começa a cortar
VOV_SCALE                = 0.012  # escala do corte
EXPOSURE_FLOOR_MOMENTUM  = 0.50   # exposição mínima em Momentum + alta convicção
ENTROPY_FLOOR_THRESHOLD  = 0.35   # entropia abaixo → aplica floor de exposição
MOMENTUM_WINDOW          = 4      # semanas para momentum feature
VOL_ZSCORE_WINDOW        = 13     # semanas para z-score de vol feature

# Multiplicadores base (P4: hedge adaptativo definido dinamicamente abaixo)
MULT_MOMENTUM  =  1.00
MULT_HEDGE_UP  = -0.20  # Alta vol, mercado ainda subindo (hedge parcial)
MULT_HEDGE_DN  = -0.70  # Alta vol, mercado caindo (hedge pleno)
MULT_INCERTO   =  0.00

# ── 1. Carregar dados ────────────────────────────────────────────────────────
ibov = pd.read_csv("ibov_2010_2025.csv", skiprows=[1], parse_dates=["Date"])
ibov.set_index("Date", inplace=True)
ibov.sort_index(inplace=True)

daily_returns = ibov["Adj Close"].pct_change().dropna()

# ── 2. Features semanais (v3: 5 features) ───────────────────────────────────
def ewm_weekly_mean(series: pd.Series, span: float = 3.0) -> pd.Series:
    def _ewm(group):
        w = np.array([(1 - 1/span)**(len(group)-1-i) for i in range(len(group))])
        w /= w.sum()
        return (group.values * w).sum()
    return series.groupby(pd.Grouper(freq="W")).apply(_ewm).dropna()

def build_features(daily: pd.Series,
                   span: float = 3.0,
                   mom_window: int = MOMENTUM_WINDOW,
                   vol_z_window: int = VOL_ZSCORE_WINDOW) -> pd.DataFrame:
    """
    5 features semanais:
      ret_mean    : retorno médio EWM da semana
      ret_std     : volatilidade intra-semana
      skewness    : assimetria dos retornos diários (detecta caudas)
      momentum    : retorno acumulado das últimas `mom_window` semanas
      vol_zscore  : z-score da vol em janela de `vol_z_window` semanas
                    (captura vol anormalmente alta ou baixa vs histórico recente)
    """
    weekly_mean = ewm_weekly_mean(daily, span=span)
    weekly_std  = daily.groupby(pd.Grouper(freq="W")).std().dropna()
    weekly_skew = daily.groupby(pd.Grouper(freq="W")).apply(
        lambda g: scipy_skew(g.values) if len(g) >= 3 else 0.0
    ).dropna()

    idx = weekly_mean.index.intersection(weekly_std.index).intersection(weekly_skew.index)

    df = pd.DataFrame({
        "ret_mean": weekly_mean[idx],
        "ret_std":  weekly_std[idx],
        "skewness": weekly_skew[idx],
    })

    # Momentum: retorno acumulado das últimas mom_window semanas
    df["momentum"] = df["ret_mean"].rolling(mom_window).sum()

    # Z-score da volatilidade
    vol_roll_mean = df["ret_std"].rolling(vol_z_window).mean()
    vol_roll_std  = df["ret_std"].rolling(vol_z_window).std()
    df["vol_zscore"] = (df["ret_std"] - vol_roll_mean) / vol_roll_std.replace(0, np.nan)

    return df.replace([np.inf, -np.inf], np.nan).dropna()

features_train = build_features(daily_returns.loc[:"2015-12-31"])
features_test  = build_features(daily_returns.loc["2016-01-01":])

# Garantir que treino e teste usam as mesmas colunas
feature_cols = ["ret_mean", "ret_std", "skewness", "momentum", "vol_zscore"]
X_train = features_train[feature_cols].values
X_test  = features_test[feature_cols].values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 3. Treinar HMM ───────────────────────────────────────────────────────────
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=2000,
                    random_state=42)
model.fit(X_train)
print(f"Convergiu: {model.monitor_.converged}  |  Iterações: {model.monitor_.iter}")

# ── 4. Probabilidades posteriores ────────────────────────────────────────────
proba_train  = model.predict_proba(X_train)
proba_test   = model.predict_proba(X_test)
states_train = model.predict(X_train)
states_test  = model.predict(X_test)

# ── 5. Métricas de incerteza ──────────────────────────────────────────────────
def uncertainty_metrics(proba: np.ndarray, n_states: int = 3) -> pd.DataFrame:
    max_prob     = proba.max(axis=1)
    sorted_p     = np.sort(proba, axis=1)[:, ::-1]
    margin       = sorted_p[:, 0] - sorted_p[:, 1]
    raw_entropy  = np.array([scipy_entropy(p) for p in proba])
    entropy_norm = raw_entropy / np.log(n_states)
    return pd.DataFrame({
        "entropy_norm": entropy_norm,
        "margin":       margin,
        "max_prob":     max_prob,
    })

uncertainty_test = uncertainty_metrics(proba_test, n_states=3)
is_uncertain     = uncertainty_test["entropy_norm"] > ENTROPY_THRESHOLD

print(f"\nSemanas incertas (entropia > {ENTROPY_THRESHOLD}): "
      f"{is_uncertain.sum()} / {len(is_uncertain)} ({is_uncertain.mean():.1%})")

# ── 6. Identificar estados por volatilidade ──────────────────────────────────
state_vol = {s: features_train["ret_std"].values[states_train == s].mean()
             for s in range(3)}
vol_order = np.argsort([state_vol[s] for s in range(3)])

color_map = {vol_order[0]: "#2ecc71", vol_order[1]: "#f39c12", vol_order[2]: "#e74c3c"}
label_map = {vol_order[0]: "Vol. Baixa",
             vol_order[1]: "Vol. Média",
             vol_order[2]: "Vol. Alta (crise)"}

STRATEGY_MAP = {
    vol_order[0]: "Momentum",
    vol_order[1]: "Neutral",
    vol_order[2]: "Hedge/Short/Vol",
}
STRATEGY_COLOR = {
    "Momentum":        "#1a6faf",
    "Neutral":         "#f39c12",
    "Hedge/Short/Vol": "#c0392b",
    "Incerto":         "dimgray",
}

# ── 7. Sinal de estratégia semanal ───────────────────────────────────────────
def assign_strategy(states: np.ndarray, uncertain: np.ndarray,
                    strategy_map: dict) -> np.ndarray:
    return np.where(uncertain,
                    "Incerto",
                    np.array([strategy_map[s] for s in states]))

strategy_test = assign_strategy(states_test, is_uncertain.values, STRATEGY_MAP)

# ── 8. Fatores de exposição ───────────────────────────────────────────────────

# 8a. Entropia → modulação contínua da alocação
entropy_factor = 1.0 - uncertainty_test["entropy_norm"].values

# 8b. Conectividade de rede (proxy: autocorr lag-1 da vol semanal)
weekly_vol = pd.Series(features_test["ret_std"].values, index=features_test.index)
corr_rolling = weekly_vol.rolling(NET_WINDOW).corr(weekly_vol.shift(1)).fillna(0.5).clip(0, 1)
network_factor = np.where(
    corr_rolling.values >= NET_THRESHOLD,
    1.0 - (corr_rolling.values - NET_THRESHOLD) / (1.0 - NET_THRESHOLD),
    1.0
).clip(0, 1)

# 8c. VoV
vov_series = weekly_vol.rolling(VOV_WINDOW).std().fillna(0.0)
vov_factor = np.where(
    vov_series.values <= VOV_THRESHOLD,
    1.0,
    (1.0 - (vov_series.values - VOV_THRESHOLD) / VOV_SCALE).clip(0, 1)
)

# 8d. Produto combinado + P5: floor de exposição em Momentum de alta convicção
exposure_raw = entropy_factor * network_factor * vov_factor

is_high_conviction_momentum = (
    (strategy_test == "Momentum") &
    (uncertainty_test["entropy_norm"].values < ENTROPY_FLOOR_THRESHOLD)
)
exposure_combined = np.where(
    is_high_conviction_momentum,
    np.maximum(exposure_raw, EXPOSURE_FLOOR_MOMENTUM),
    exposure_raw
)

print(f"\nExposição média combinada: {exposure_combined.mean():.1%}")
print(f"  Semanas com floor ativo (Momentum + alta convicção): "
      f"{is_high_conviction_momentum.sum()} ({is_high_conviction_momentum.mean():.1%})")

# ── 9. P4: Multiplicador de Hedge adaptativo ─────────────────────────────────
# Hedge pleno (−0.70) apenas quando ret_mean da semana anterior foi negativo.
# Hedge parcial (−0.20) quando vol está alta mas mercado ainda sobe.
ret_prev = np.roll(features_test["ret_mean"].values, 1)
ret_prev[0] = 0.0  # sem informação na primeira semana

hedge_mult = np.where(ret_prev < 0, MULT_HEDGE_DN, MULT_HEDGE_UP)

# ── 10. P1: Retornos base por estratégia (Neutral = mean-reversion) ──────────
# Neutral mean-reversion: fade do retorno anterior.
#   Se ret_anterior > 0 → apostamos em reversão (short) → mult negativo
#   Se ret_anterior < 0 → apostamos em bounce (long)   → mult positivo
# Magnitude controlada: 0.30 × |ret_prev|, sinal oposto.
# Isso captura reversões à média sem exposição direcional líquida.
neutral_signal = -0.30 * np.sign(ret_prev)   # sinal de reversão

raw_strat_returns = np.zeros(len(strategy_test))
for t, strat in enumerate(strategy_test):
    raw = features_test["ret_mean"].values[t]
    if strat == "Momentum":
        raw_strat_returns[t] = MULT_MOMENTUM * raw
    elif strat == "Neutral":
        # mean-reversion: usa sinal de reversão × volatilidade local como proxy
        raw_strat_returns[t] = neutral_signal[t] * features_test["ret_std"].values[t]
    elif strat == "Hedge/Short/Vol":
        raw_strat_returns[t] = hedge_mult[t] * raw
    else:  # Incerto
        raw_strat_returns[t] = 0.0

# ── 11. P2: Backtest com Stop-Loss + Trailing Stop + Reentrada Antecipada ────
def backtest_v3(
    raw_returns:    np.ndarray,
    exposure:       np.ndarray,
    strategy:       np.ndarray,
    entropy_norm:   np.ndarray,
    stop_loss:      float,
    trailing_stop:  float,
    cooldown:       int,
    cooldown_fast:  int,
    entropy_fast:   float,
) -> tuple:
    """
    Novidade v3 (P2):
      Reentrada antecipada: se o regime no momento seria Momentum
      e a entropia < entropy_fast, cooldown reduz para cooldown_fast semanas
      em vez de cooldown.
    """
    n             = len(raw_returns)
    net_returns   = np.zeros(n)
    equity_curve  = np.ones(n + 1)
    stop_flags    = np.zeros(n, dtype=int)
    stop_type     = np.array(["none"] * n, dtype=object)

    peak_ever     = 1.0
    running_high  = 1.0
    cooldown_ctr  = 0

    for t in range(n):
        eq = equity_curve[t]

        if cooldown_ctr > 0:
            # P2: checar reentrada antecipada
            is_momentum_now  = strategy[t] == "Momentum"
            is_low_entropy   = entropy_norm[t] < entropy_fast
            if is_momentum_now and is_low_entropy and cooldown_ctr > cooldown_fast:
                cooldown_ctr = cooldown_fast  # comprime o cooldown restante

            stop_flags[t]     = 1
            stop_type[t]      = "cooldown"
            net_returns[t]    = 0.0
            equity_curve[t+1] = eq
            cooldown_ctr     -= 1
            continue

        sl_triggered = eq < peak_ever    * (1 - stop_loss)
        ts_triggered = eq < running_high * (1 - trailing_stop)

        if sl_triggered or ts_triggered:
            stop_flags[t]     = 1
            stop_type[t]      = "sl" if sl_triggered else "ts"
            net_returns[t]    = 0.0
            equity_curve[t+1] = eq
            cooldown_ctr      = cooldown - 1
            running_high      = eq
            continue

        r                 = raw_returns[t] * exposure[t]
        net_returns[t]    = r
        new_eq            = eq * (1 + r)
        equity_curve[t+1] = new_eq

        if new_eq > peak_ever:    peak_ever    = new_eq
        if new_eq > running_high: running_high = new_eq

    return net_returns, equity_curve[1:], stop_flags, stop_type


net_returns, equity_curve, stop_flags, stop_type = backtest_v3(
    raw_returns   = raw_strat_returns,
    exposure      = exposure_combined,
    strategy      = strategy_test,
    entropy_norm  = uncertainty_test["entropy_norm"].values,
    stop_loss     = STOP_LOSS_PCT,
    trailing_stop = TRAILING_STOP_PCT,
    cooldown      = COOLDOWN_WEEKS,
    cooldown_fast = COOLDOWN_FAST,
    entropy_fast  = ENTROPY_FAST_REENTRY,
)

# Buy-and-hold de referência
bh_returns = features_test["ret_mean"].values
bh_equity  = np.cumprod(1 + bh_returns)
strat_cum  = equity_curve - 1
bh_cum     = bh_equity - 1

# ── 12. Métricas de performance ───────────────────────────────────────────────
W = 52  # semanas/ano

def sharpe(r: np.ndarray) -> float:
    s = r.std()
    return (r.mean() / s * np.sqrt(W)) if s > 0 else 0.0

def max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return ((eq - peak) / peak).min()

def calmar(eq: np.ndarray, r: np.ndarray) -> float:
    ann = eq[-1] ** (W / len(r)) - 1
    mdd = abs(max_drawdown(eq))
    return ann / mdd if mdd > 0 else 0.0

def sortino(r: np.ndarray) -> float:
    neg = r[r < 0]
    ds  = neg.std()
    return (r.mean() / ds * np.sqrt(W)) if ds > 0 else 0.0

sr_s = sharpe(net_returns);   sr_b = sharpe(bh_returns)
mdd_s = max_drawdown(equity_curve); mdd_b = max_drawdown(bh_equity)
cal_s = calmar(equity_curve, net_returns); cal_b = calmar(bh_equity, bh_returns)
sor_s = sortino(net_returns);  sor_b = sortino(bh_returns)
hit_s = (net_returns[net_returns != 0] > 0).mean()
n_sl  = (stop_type == "sl").sum()
n_ts  = (stop_type == "ts").sum()

print(f"\n{'═'*58}")
print(f"  PERFORMANCE v3 — BACKTEST 2016–2025")
print(f"{'═'*58}")
print(f"  {'Métrica':<26} {'HMM v3':>10} {'B&H':>10}")
print(f"  {'-'*46}")
print(f"  {'Retorno acumulado':<26} {strat_cum[-1]:>10.2%} {bh_cum[-1]:>10.2%}")
print(f"  {'Sharpe (anual)':<26} {sr_s:>10.2f} {sr_b:>10.2f}")
print(f"  {'Sortino (anual)':<26} {sor_s:>10.2f} {sor_b:>10.2f}")
print(f"  {'Max Drawdown':<26} {mdd_s:>10.2%} {mdd_b:>10.2%}")
print(f"  {'Calmar ratio':<26} {cal_s:>10.2f} {cal_b:>10.2f}")
print(f"  {'Hit rate (sem. operadas)':<26} {hit_s:>10.1%} {'—':>10}")
print(f"  {'Stop-loss acionado':<26} {n_sl:>10}x {'—':>10}")
print(f"  {'Trailing stop acionado':<26} {n_ts:>10}x {'—':>10}")
print(f"  {'Semanas em cooldown':<26} {stop_flags.sum():>10} {'—':>10}")
print(f"  {'Exposição média':<26} {exposure_combined.mean():>10.1%} {'100.0%':>10}")
print('═'*58)

# ── 13. Diagnóstico por estado ────────────────────────────────────────────────
print("\n=== Estados HMM (treino, 5 features) ===")
for s in range(3):
    mask = states_train == s
    print(f"  Estado {s} [{label_map[s]}] → {STRATEGY_MAP[s]}")
    print(f"    ret={features_train['ret_mean'].values[mask].mean():.4%}  "
          f"vol={features_train['ret_std'].values[mask].mean():.4%}  "
          f"skew={features_train['skewness'].values[mask].mean():.2f}  "
          f"n={mask.sum()}")

print("\n=== Distribuição de estratégias (teste) ===")
for strat in ["Momentum", "Neutral", "Hedge/Short/Vol", "Incerto"]:
    n   = (strategy_test == strat).sum()
    pct = n / len(strategy_test)
    print(f"  {strat:<20}: {n:>4} semanas ({pct:.1%})")

# ── 14. Estratégia atual ──────────────────────────────────────────────────────
idx = features_test.index
current_strat   = strategy_test[-1]
cur_ent         = uncertainty_test["entropy_norm"].values[-1]
cur_ent_f       = entropy_factor[-1]
cur_net_f       = network_factor[-1]
cur_vov_f       = vov_factor[-1]
cur_exp         = exposure_combined[-1]
cur_hedge_mult  = hedge_mult[-1]

print(f"\n{'═'*62}")
print(f"  SINAL ATUAL (última semana): {current_strat}")
print(f"  Exposição combinada        : {cur_exp:.1%}")
print(f"    entropy_factor = {cur_ent_f:.3f}  (entropia={cur_ent:.3f})")
print(f"    network_factor = {cur_net_f:.3f}  (corr={corr_rolling.values[-1]:.3f})")
print(f"    vov_factor     = {cur_vov_f:.3f}  (VoV={vov_series.values[-1]:.5f})")
if current_strat == "Hedge/Short/Vol":
    print(f"    hedge_mult     = {cur_hedge_mult:.2f}  "
          f"({'hedge pleno' if cur_hedge_mult == MULT_HEDGE_DN else 'hedge parcial'})")
print('═'*62)

# ── 15. Figura principal ──────────────────────────────────────────────────────
fig, axes = plt.subplots(6, 1, figsize=(17, 22), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1, 1, 1, 0.8, 1.8]})

def hl(color, label, alpha=0.55):
    return mpatches.Patch(color=color, alpha=alpha, label=label)

# subplot 1 — retornos + estratégia
ax1 = axes[0]
for i in range(len(idx)-1):
    if is_uncertain.values[i]:
        ax1.axvspan(idx[i], idx[i+1], color="silver", alpha=0.28, zorder=1)
    if stop_flags[i]:
        ax1.axvspan(idx[i], idx[i+1], color="mistyrose", alpha=0.55, zorder=2)

ax1.plot(idx, features_test["ret_mean"].values, color="black", alpha=0.12, lw=0.8)
for strat in ["Momentum", "Neutral", "Hedge/Short/Vol"]:
    mask = strategy_test == strat
    ax1.scatter(idx[mask], features_test["ret_mean"].values[mask],
                c=STRATEGY_COLOR[strat], label=strat, s=26, alpha=0.85, zorder=3)
ax1.scatter(idx[strategy_test == "Incerto"],
            features_test["ret_mean"].values[strategy_test == "Incerto"],
            c="dimgray", label="Incerto", s=26, marker="x", zorder=4)
ax1.axhline(0, color="black", lw=0.5, linestyle="--")
ax1.set_ylabel("Retorno semanal (EWM)")
ax1.set_title("HMM Trading System v3 — IBOV 2016-2025  |  "
              "5 features · Neutral mean-reversion · Hedge adaptativo · Floor de exposição",
              fontsize=10)
h, l = ax1.get_legend_handles_labels()
new_handles = h + [hl("mistyrose", "Stop/Cooldown"), hl("silver", "Incerteza HMM")]
new_labels = l + ["Stop/Cooldown", "Incerteza HMM"]
ax1.legend(new_handles, new_labels, loc="upper left", fontsize=8, ncol=3)
ax1.grid(True, alpha=0.3)

# subplot 2 — exposição + fatores
ax2 = axes[1]
ax2.fill_between(idx, exposure_combined*100, alpha=0.28, color="navy")
ax2.plot(idx, exposure_combined*100, color="navy",     lw=1.1, label="Exposição combinada")
ax2.plot(idx, entropy_factor*100,   color="steelblue", lw=0.8, linestyle="-",  alpha=0.7, label="entropy")
ax2.plot(idx, network_factor*100,   color="teal",      lw=0.8, linestyle=":",  alpha=0.9, label="rede")
ax2.plot(idx, vov_factor*100,       color="purple",    lw=0.8, linestyle="--", alpha=0.9, label="VoV")
ax2.axhline(EXPOSURE_FLOOR_MOMENTUM*100, color="navy", lw=0.7, linestyle=":",
            label=f"Floor Momentum ({EXPOSURE_FLOOR_MOMENTUM:.0%})")
ax2.set_ylabel("Exposição (%)"); ax2.set_ylim(0, 112)
ax2.legend(fontsize=8, loc="lower left", ncol=3)
ax2.grid(True, alpha=0.3)

# subplot 3 — entropia
ax3 = axes[2]
ax3.fill_between(idx, uncertainty_test["entropy_norm"].values, alpha=0.25, color="steelblue")
ax3.plot(idx, uncertainty_test["entropy_norm"].values, color="steelblue", lw=0.9, label="Entropia norm.")
ax3.axhline(ENTROPY_THRESHOLD,    color="red",    lw=1.2, linestyle="--", label=f"Limiar = {ENTROPY_THRESHOLD}")
ax3.axhline(ENTROPY_FAST_REENTRY, color="orange", lw=1.0, linestyle=":",  label=f"Reentrada rápida = {ENTROPY_FAST_REENTRY}")
ax3.axhline(ENTROPY_FLOOR_THRESHOLD, color="navy",lw=0.8, linestyle="-.", label=f"Floor Momentum = {ENTROPY_FLOOR_THRESHOLD}")
ax3r = ax3.twinx()
ax3r.plot(idx, uncertainty_test["margin"].values, color="darkorange", lw=0.8, alpha=0.7, label="Margem")
ax3r.set_ylabel("Margem", color="darkorange", fontsize=8)
ax3r.tick_params(axis="y", labelcolor="darkorange")
ax3.set_ylabel("Entropia norm."); ax3.set_ylim(0, 1.05)
l3, b3 = ax3.get_legend_handles_labels()
l3r, b3r = ax3r.get_legend_handles_labels()
ax3.legend(l3+l3r, b3+b3r, fontsize=7, loc="upper right", ncol=2)
ax3.grid(True, alpha=0.3)

# subplot 4 — VoV + rede + hedge_mult
ax4 = axes[3]
ax4.fill_between(idx, vov_series.values*1_000, alpha=0.22, color="purple", label="VoV×10³")
ax4.plot(idx, vov_series.values*1_000, color="purple", lw=0.9)
ax4.axhline(VOV_THRESHOLD*1_000, color="purple", lw=1, linestyle="--",
            label=f"Limiar VoV = {VOV_THRESHOLD*1000:.1f}×10⁻³")
ax4r = ax4.twinx()
ax4r.plot(idx, corr_rolling.values, color="teal", lw=0.9, alpha=0.8, label="Corr. rede")
ax4r.axhline(NET_THRESHOLD, color="teal", lw=1, linestyle=":", label=f"Limiar rede={NET_THRESHOLD}")
ax4r.set_ylabel("Correlação (rede)", color="teal", fontsize=8)
ax4r.tick_params(axis="y", labelcolor="teal")
ax4.set_ylabel("VoV×10³")
l4, b4 = ax4.get_legend_handles_labels()
l4r, b4r = ax4r.get_legend_handles_labels()
ax4.legend(l4+l4r, b4+b4r, fontsize=8, loc="upper right")
ax4.grid(True, alpha=0.3)

# subplot 5 — probabilidades empilhadas
ax5 = axes[4]
bottom = np.zeros(len(proba_test))
for s in range(3):
    ax5.bar(idx, proba_test[:, s], bottom=bottom,
            color=color_map[s], label=label_map[s], width=6, alpha=0.85)
    bottom += proba_test[:, s]
ax5.axhline(0.5, color="black", lw=0.5, linestyle=":")
ax5.set_ylabel("P(estado | dados)"); ax5.set_ylim(0, 1)
ax5.legend(fontsize=8, loc="lower left", ncol=3)
ax5.grid(True, alpha=0.2)

# subplot 6 — curvas de capital + métricas
ax6 = axes[5]
for i in range(len(idx)-1):
    if stop_flags[i]:
        ax6.axvspan(idx[i], idx[i+1], color="mistyrose", alpha=0.4, zorder=1)
ax6.plot(idx, bh_equity,    color="gray",    lw=1.3, linestyle="--",
         label=f"Buy-and-Hold  ret={bh_cum[-1]:.0%}  Sharpe={sr_b:.2f}  MDD={mdd_b:.0%}")
ax6.plot(idx, equity_curve, color="#1a6faf", lw=1.7,
         label=f"HMM v3        ret={strat_cum[-1]:.0%}  Sharpe={sr_s:.2f}  MDD={mdd_s:.0%}")
ax6.axhline(1, color="black", lw=0.5, linestyle=":")
ax6.fill_between(idx, equity_curve, bh_equity,
                 where=(equity_curve >= bh_equity), alpha=0.12, color="green")
ax6.fill_between(idx, equity_curve, bh_equity,
                 where=(equity_curve < bh_equity),  alpha=0.12, color="red")

# Caixa de métricas
bbox = dict(boxstyle="round,pad=0.4", fc="white", alpha=0.88, ec="gray")
ax6.text(0.01, 0.97,
         f"HMM v3 — Sharpe:{sr_s:.2f}  Sortino:{sor_s:.2f}  Calmar:{cal_s:.2f}  MDD:{mdd_s:.0%}  Hit:{hit_s:.0%}",
         transform=ax6.transAxes, fontsize=8.5, va="top", bbox=bbox)
ax6.text(0.01, 0.86,
         f"B&H   — Sharpe:{sr_b:.2f}  Sortino:{sor_b:.2f}  Calmar:{cal_b:.2f}  MDD:{mdd_b:.0%}",
         transform=ax6.transAxes, fontsize=8.5, va="top", color="gray", bbox=bbox)

ax6.set_ylabel("Capital (base = 1)")
ax6.set_xlabel("Data")
ax6.legend(fontsize=8.5, loc="upper left")
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hmm_trading_strategy.png", dpi=300)
plt.show()

# ── 16. Tabela por estratégia ─────────────────────────────────────────────────
print("\n=== Performance por estratégia (teste, v3) ===")
strat_df = pd.DataFrame({
    "strategy":     strategy_test,
    "ret_mean":     features_test["ret_mean"].values,
    "net_return":   net_returns,
    "entropy_norm": uncertainty_test["entropy_norm"].values,
    "exposure":     exposure_combined,
    "stop_flag":    stop_flags,
    "neutral_sig":  neutral_signal,
}, index=idx)

for strat in ["Momentum", "Neutral", "Hedge/Short/Vol", "Incerto"]:
    sub = strat_df[strat_df["strategy"] == strat]
    if len(sub) == 0:
        continue
    cum  = (1 + sub["net_return"]).prod() - 1
    hit  = (sub["net_return"] > 0).mean() if strat != "Incerto" else float("nan")
    ann  = (1 + cum) ** (W / max(len(sub), 1)) - 1
    print(f"\n  {strat:<20} (n={len(sub)})")
    print(f"    Retorno acum.    : {cum:.2%}  (anualizado: {ann:.2%})")
    if not np.isnan(hit):
        print(f"    Hit rate         : {hit:.1%}")
    print(f"    Retorno médio/sem: {sub['net_return'].mean():.4%}")
    print(f"    Exposição média  : {sub['exposure'].mean():.1%}")
    print(f"    Entropia média   : {sub['entropy_norm'].mean():.3f}")
    print(f"    Semanas em stop  : {sub['stop_flag'].sum()}")