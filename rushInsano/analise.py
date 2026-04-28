"""
analise.py — Dashboard de Análise do CERBERUS
═══════════════════════════════════════════════════════════════════════
Roda APÓS o checkpoint.py. Importa o pipeline, executa e gera:

  FIG 1  — Estados HMM coloridos sobre o IBOV
  FIG 2  — Entropia ao longo do tempo
  FIG 3  — Distribuição de retornos por regime
  FIG 4  — Density × tempo
  FIG 5  — max_eigen_value × tempo
  FIG 6  — Density vs max_eigen_value (scatter)
  FIG 7  — Equity curve + Drawdown
  FIG 8  — Heatmap de retornos mensais
  FIG 9  — Rolling Sharpe (252 dias)
  FIG 10 — Distribuição estratégia vs IBOV
  TABLE  — Métricas estendidas (Sharpe, Sortino, Calmar, Win Rate, etc.)
═══════════════════════════════════════════════════════════════════════
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from DIPQ import QuantStrategyPipeline, STOP_LOSS_PCT, TRAILING_STOP_PCT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

# ── Paleta e estilo global ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#F8F9FA",
    "axes.facecolor":    "#FFFFFF",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "#E5E5E5",
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "lines.linewidth":   1.6,
})

# Cores dos regimes
COR_MOMENTUM = "#2ECC71"   # verde
COR_NEUTRAL  = "#F39C12"   # âmbar
COR_HEDGE    = "#E74C3C"   # vermelho
COR_STRAT    = "#1A6FAF"   # azul escuro
COR_BENCH    = "#95A5A6"   # cinza

CRISES = {
    "Dilma\nImpeach.":      "2016-04-17",
    "Joesley\nDay":         "2017-05-18",
    "Greve dos\nCaminh.":   "2018-05-21",
    "Brumadinho":           "2019-01-25",
    "COVID-19":             "2020-03-23",
    "Invasão\nUcrânia":     "2022-02-24",
}

# ── Executar pipeline ─────────────────────────────────────────────────────────
print("=" * 60)
print("  CERBERUS — Dashboard de Análise")
print("=" * 60)
print("\nInicializando pipeline...")

pipeline = QuantStrategyPipeline(
    start_date='2010-01-04',
    end_date='2025-12-03',
    split_date='2019-01-01'
)
plt.close("all")   # fecha plots do __init__

print("Calculando rede de conectividade...")
b3_net = pipeline.net_analysis(df=pipeline.b3_returns, window=21)
plt.close("all")

pipeline.ibov['density']         = b3_net['density']
pipeline.ibov['max_eigen_value'] = b3_net['max_eigen_value']

print("Calculando features...")
pipeline.calculate_features(window=21)
plt.close("all")

print("Treinando HMM...")
pipeline.fit_hmm()
plt.close("all")

print("Rodando backtest...")
pipeline.run_backtest(
    initial_capital=100000,
    stop_loss_pct=STOP_LOSS_PCT,
    trailing_stop_pct=TRAILING_STOP_PCT
)

# ── Montar perf_df ────────────────────────────────────────────────────────────
perf_df = pipeline.ibov.join(pipeline.equity_df)
perf_df = perf_df.dropna(subset=['strategy_equity']).copy()
perf_df['strategy_returns']  = perf_df['strategy_equity'].pct_change().fillna(0)
perf_df['benchmark_returns'] = perf_df['returns'].fillna(0)
perf_df['cum_strategy']      = (1 + perf_df['strategy_returns']).cumprod()
perf_df['cum_benchmark']     = (1 + perf_df['benchmark_returns']).cumprod()

split = pd.to_datetime('2019-01-01')
label_map  = {0: "Momentum (baixa vol)", 1: "Neutral (média vol)", 2: "Hedge (alta vol)"}
cor_regime = {0: COR_MOMENTUM, 1: COR_NEUTRAL, 2: COR_HEDGE}

ibov = pipeline.ibov.dropna(subset=['hmm_state'])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 1 — Estados HMM sobre o IBOV                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\nGerando Figure 1 — Estados HMM sobre o IBOV...")

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor("#F8F9FA")

# Fundo colorido por regime
dates = ibov.index
for i in range(len(dates) - 1):
    estado = ibov['hmm_state'].iloc[i]
    if pd.notna(estado):
        ax.axvspan(dates[i], dates[i+1],
                   color=cor_regime[int(estado)], alpha=0.18, linewidth=0)

# Preço do IBOV
ax.plot(ibov.index, ibov['Adj Close'], color="#1C252E", linewidth=1.2,
        label="IBOV (Adj Close)", zorder=3)

# Linha de divisão treino/teste
ax.axvline(split, color="#C0392B", linestyle="--", linewidth=1.5,
           label="Início do Teste (2019)", zorder=4)

# Anotações de crises
for nome, data in CRISES.items():
    d = pd.to_datetime(data)
    if d in ibov.index or ibov.index.searchsorted(d) < len(ibov):
        ax.axvline(d, color="#7F8C8D", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(d, ibov['Adj Close'].max() * 0.97, nome,
                fontsize=7.5, color="#555555", ha='center', va='top',
                bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))

# Legenda de regimes
patches = [mpatches.Patch(color=cor_regime[k], alpha=0.5, label=v)
           for k, v in label_map.items()]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + patches, loc="upper left", framealpha=0.92, ncol=2)

ax.set_title("Regimes de Mercado Detectados pelo HMM — IBOV 2010–2025")
ax.set_xlabel("Data")
ax.set_ylabel("Pontos (IBOV)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig("fig1_regimes_ibov.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 2 — Entropia ao longo do tempo                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 2 — Entropia...")

# Define o novo limiar baseado na entropia normalizada (0 a 1)
THRESHOLD_ENTROPY = 0.6 

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor("#F8F9FA")

ent = ibov.loc[ibov.index >= split, 'hmm_entropy'].dropna()

# --- Fundo colorido por regime (igual à Fig 5) ---
ibov_oos = ibov.loc[ibov.index >= split]
dates = ibov_oos.index
for i in range(len(dates) - 1):
    estado = ibov_oos['hmm_state'].iloc[i]   # ← iloc[i] agora correto
    if pd.notna(estado):
        ax.axvspan(dates[i], dates[i+1],
                   color=cor_regime[int(estado)], alpha=0.18, linewidth=0)
# -------------------------------------------------

ax.fill_between(ent.index, ent.values, alpha=0.25, color="#2980B9")
ax.plot(ent.index, ent.values, color="#2980B9", linewidth=1.0, label="Entropia HMM")

# Threshold de operação (Atualizado para a nova escala)
ax.axhline(THRESHOLD_ENTROPY, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"Threshold (H > {THRESHOLD_ENTROPY})")

# Entropia máxima teórica (Agora é 1.0 por conta da normalização prévia)
ax.axhline(1.0, color="#95A5A6", linestyle=":", linewidth=1.0,
           label="Entropia máxima normalizada = 1.0")

# Crises (Altura do texto ajustada de 1.52 para 0.95)
for nome, data in CRISES.items():
    d = pd.to_datetime(data)
    ax.axvline(d, color="#7F8C8D", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(d, 0.95, nome, fontsize=7.5, color="#555555",
            ha='center', va='top',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))

# Linha de divisão (In-sample / Out-of-sample)
ax.axvline(split, color="#C0392B", linestyle="--", linewidth=1.5, alpha=0.6)

# Limites e Rótulos ajustados para a nova escala de 0 a 1
ax.set_ylim(0, 1.05)
ax.set_title("Entropia de Shannon do HMM ao Longo do Tempo (Normalizada)")
ax.set_xlabel("Data")
ax.set_ylabel("Entropia H (0 a 1)")

# --- Legendas dos regimes HMM (igual à Fig 5) ---
patches = [mpatches.Patch(color=cor_regime[k], alpha=0.5, label=v)
           for k, v in label_map.items()]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + patches, loc="upper right", framealpha=0.92, ncol=3)
# -------------------------------------------------
ax.set_xlim(split, ibov.index.max())
plt.tight_layout()
plt.savefig("fig2_entropia.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 3 — Distribuição de retornos por regime                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 3 — Distribuição de retornos por regime...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.patch.set_facecolor("#F8F9FA")
fig.suptitle("Distribuição de Retornos Diários por Regime de Mercado",
             fontsize=14, fontweight='bold', y=1.01)

x_range = np.linspace(-0.08, 0.08, 300)

for idx, (estado, nome) in enumerate(label_map.items()):
    ax = axes[idx]
    mask = ibov['hmm_state'] == estado
    rets = ibov.loc[mask, 'returns'].dropna()

    cor = cor_regime[estado]

    # Histograma
    ax.hist(rets, bins=60, density=True, color=cor, alpha=0.35,
            edgecolor='white', linewidth=0.4)

    # KDE suavizado
    if len(rets) > 10:
        kde = gaussian_kde(rets, bw_method=0.3)
        ax.plot(x_range, kde(x_range), color=cor, linewidth=2.2)

    # Linha normal de referência
    from scipy.stats import norm
    mu, sigma = rets.mean(), rets.std()
    ax.plot(x_range, norm.pdf(x_range, mu, sigma),
            color="#888888", linewidth=1.2, linestyle="--",
            label="Normal teórica")

    # Linha de média
    ax.axvline(mu, color=cor, linestyle="-.", linewidth=1.2,
               label=f"Média: {mu:.2%}")
    ax.axvline(0, color="#AAAAAA", linestyle=":", linewidth=0.8)

    stats_txt = (f"n = {len(rets):,}\n"
                 f"μ = {mu:.3%}\n"
                 f"σ = {sigma:.3%}\n"
                 f"Kurt = {rets.kurt():.2f}")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='right',
            bbox=dict(fc='white', ec=cor, alpha=0.85, pad=4, lw=1.2))

    ax.set_title(nome, color=cor, fontweight='bold')
    ax.set_xlabel("Retorno Diário")
    ax.set_ylabel("Densidade" if idx == 0 else "")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig3_distribuicao_regimes.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 4 — Density × tempo                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 4 — Density × tempo...")

density_s = ibov['density'].dropna()
p80 = density_s.quantile(0.80)

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor("#F8F9FA")

ax.fill_between(density_s.index, density_s.values, alpha=0.20, color="#8E44AD")
ax.plot(density_s.index, density_s.values, color="#8E44AD", linewidth=1.0)

# Zona de risco sistêmico
ax.fill_between(density_s.index, p80, density_s.values,
                where=(density_s.values > p80),
                alpha=0.40, color="#E74C3C",
                label=f"Zona de risco (acima do P80 = {p80:.2%}) → exposição ÷ 2")

ax.axhline(p80, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"Percentil 80: {p80:.2%}")

for nome, data in CRISES.items():
    d = pd.to_datetime(data)
    ax.axvline(d, color="#7F8C8D", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(d, density_s.max() * 0.97, nome, fontsize=7.5, color="#555555",
            ha='center', va='top',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))

ax.axvline(split, color="#C0392B", linestyle="--", linewidth=1.5, alpha=0.6)
ax.set_title("Densidade da Rede de Correlações da B3 — Conectividade do Mercado")
ax.set_xlabel("Data")
ax.set_ylabel("Densidade da Rede (arestas ativas / total possível)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
ax.legend(loc="upper right", framealpha=0.92)
plt.tight_layout()
plt.savefig("fig4_density.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 5 — max_eigen_value × tempo                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 5 — max_eigen_value × tempo...")

eig_s = ibov['max_eigen_value'].dropna()

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor("#F8F9FA")

# --- ADIÇÃO: Fundo colorido por regime (igual à Fig 1) ---
dates = ibov.index
for i in range(len(dates) - 1):
    estado = ibov['hmm_state'].iloc[i]
    if pd.notna(estado):
        ax.axvspan(dates[i], dates[i+1],
                   color=cor_regime[int(estado)], alpha=0.18, linewidth=0)
# ---------------------------------------------------------

# Plota a linha e o preenchimento do autovalor máximo por cima do fundo
ax.fill_between(eig_s.index, eig_s.values, alpha=0.18, color="#16A085")
ax.plot(eig_s.index, eig_s.values, color="#16A085", linewidth=1.0)

# Linha de média histórica
media_eig = eig_s.mean()
ax.axhline(media_eig, color="#888888", linestyle="--", linewidth=1.2,
           label=f"Média histórica: {media_eig:.1f}")

# Anotações de crises com setas
for nome, data in CRISES.items():
    d = pd.to_datetime(data)
    idx_loc = eig_s.index.searchsorted(d)
    if idx_loc < len(eig_s):
        val = eig_s.iloc[min(idx_loc, len(eig_s)-1)]
        ax.annotate(nome,
                    xy=(d, val),
                    xytext=(d, val + eig_s.max() * 0.12),
                    fontsize=8, ha='center', color="#C0392B",
                    arrowprops=dict(arrowstyle="-|>", color="#C0392B",
                                   lw=1.2, mutation_scale=10),
                    bbox=dict(fc='white', ec='#C0392B', alpha=0.85,
                              pad=3, lw=1.0, boxstyle='round,pad=0.3'))

ax.axvline(split, color="#C0392B", linestyle="--", linewidth=1.5, alpha=0.6,
           label="Início do Teste (2019)")

ax.set_title("Maior Autovalor da Rede (λ_max) — Fragilidade Sistêmica do Mercado")
ax.set_xlabel("Data")
ax.set_ylabel("λ_max (autovalor máximo da matriz de adjacência)")

# --- ADIÇÃO: Adicionar legendas dos regimes HMM ---
patches = [mpatches.Patch(color=cor_regime[k], alpha=0.5, label=v)
           for k, v in label_map.items()]
handles, labels = ax.get_legend_handles_labels()
# Usa ncol=3 para a legenda não ficar alta demais e cobrir o gráfico
ax.legend(handles=handles + patches, loc="upper left", framealpha=0.92, ncol=3) 
# ---------------------------------------------------

plt.tight_layout()
plt.savefig("fig5_eigen.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 6 — Density vs max_eigen_value (scatter)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 6 — Scatter density vs eigen...")

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#F8F9FA")

net_df = ibov[['density', 'max_eigen_value', 'hmm_state']].dropna()

for estado in [0, 1, 2]:
    sub = net_df[net_df['hmm_state'] == estado]
    ax.scatter(sub['density'], sub['max_eigen_value'],
               c=cor_regime[estado], alpha=0.25, s=8,
               label=label_map[estado])

# Linha de tendência
from numpy.polynomial.polynomial import polyfit
x = net_df['density'].values
y = net_df['max_eigen_value'].values
mask_valid = np.isfinite(x) & np.isfinite(y)
if mask_valid.sum() > 2:
    coef = np.polyfit(x[mask_valid], y[mask_valid], 1)
    x_line = np.linspace(x[mask_valid].min(), x[mask_valid].max(), 100)
    ax.plot(x_line, np.polyval(coef, x_line),
            color="#1C252E", linewidth=1.5, linestyle="--",
            label=f"Tendência linear")

corr = pd.Series(x[mask_valid]).corr(pd.Series(y[mask_valid]))
ax.text(0.97, 0.05, f"Correlação: {corr:.3f}",
        transform=ax.transAxes, fontsize=9, ha='right',
        bbox=dict(fc='white', ec='#888', alpha=0.85, pad=4))

ax.set_title("Densidade da Rede vs Maior Autovalor (λ_max)\nValidação dos Dois Indicadores de Risco Sistêmico")
ax.set_xlabel("Densidade da Rede")
ax.set_ylabel("λ_max")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
ax.legend(loc="upper left", framealpha=0.92)
plt.tight_layout()
plt.savefig("fig6_scatter_rede.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 7 — Equity Curve + Drawdown                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 7 — Equity + Drawdown...")

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor("#F8F9FA")
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)

ax_eq  = fig.add_subplot(gs[0])
ax_dd  = fig.add_subplot(gs[1], sharex=ax_eq)

# Filtra apenas o período OOS e rebasa ambas as curvas a 1.0 no split
plot_df = perf_df[perf_df.index >= split].copy()
plot_df['cum_strategy_oos']  = (1 + plot_df['strategy_returns']).cumprod()
plot_df['cum_benchmark_oos'] = (1 + plot_df['benchmark_returns']).cumprod()

# Equity
ax_eq.plot(plot_df.index, plot_df['cum_strategy_oos'],
           color=COR_STRAT, linewidth=2.0,
           label="CERBERUS (HMM Multi-Ativos)")
ax_eq.plot(plot_df.index, plot_df['cum_benchmark_oos'],
           color=COR_BENCH, linewidth=1.4, linestyle="--",
           label="Buy & Hold IBOV")

# Alpha fill
ax_eq.fill_between(plot_df.index,
                   plot_df['cum_strategy_oos'], plot_df['cum_benchmark_oos'],
                   where=(plot_df['cum_strategy_oos'] >= plot_df['cum_benchmark_oos']),
                   alpha=0.12, color="#27AE60", label="Alpha positivo")
ax_eq.fill_between(plot_df.index,
                   plot_df['cum_strategy_oos'], plot_df['cum_benchmark_oos'],
                   where=(plot_df['cum_strategy_oos'] < plot_df['cum_benchmark_oos']),
                   alpha=0.12, color="#E74C3C", label="Abaixo do benchmark")

ax_eq.axvline(split, color="#C0392B", linestyle="--", linewidth=1.5,
              label="Início do Teste (2019)")

# Métricas no gráfico (calculadas sobre o OOS)
ret_total_s = plot_df['cum_strategy_oos'].iloc[-1] - 1
ret_total_b = plot_df['cum_benchmark_oos'].iloc[-1] - 1
ax_eq.text(0.01, 0.97,
           f"CERBERUS: {ret_total_s:.1%} acumulado (OOS)",
           transform=ax_eq.transAxes, fontsize=9.5,
           va='top', color=COR_STRAT,
           bbox=dict(fc='white', ec=COR_STRAT, alpha=0.88, pad=4, lw=1.2))
ax_eq.text(0.01, 0.88,
           f"IBOV B&H:   {ret_total_b:.1%} acumulado (OOS)",
           transform=ax_eq.transAxes, fontsize=9.5,
           va='top', color=COR_BENCH,
           bbox=dict(fc='white', ec=COR_BENCH, alpha=0.88, pad=4, lw=1.2))

ax_eq.set_ylabel("Retorno Acumulado (1.0 = Capital Inicial em Jan/2019)")
ax_eq.set_title("CERBERUS vs Buy & Hold IBOV — Performance Acumulada (Out-of-Sample)")
ax_eq.legend(loc="upper left", framealpha=0.92, ncol=2)
ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
plt.setp(ax_eq.get_xticklabels(), visible=False)

# Drawdown — também apenas OOS
dd_strat = (plot_df['cum_strategy_oos']  / plot_df['cum_strategy_oos'].cummax()  - 1) * 100
dd_bench = (plot_df['cum_benchmark_oos'] / plot_df['cum_benchmark_oos'].cummax() - 1) * 100

ax_dd.fill_between(plot_df.index, dd_strat, 0,
                   alpha=0.45, color=COR_STRAT, label="Drawdown CERBERUS")
ax_dd.plot(plot_df.index, dd_bench, color=COR_BENCH,
           linewidth=1.0, linestyle="--", label="Drawdown IBOV")
ax_dd.axvline(split, color="#C0392B", linestyle="--", linewidth=1.5, alpha=0.6)

ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_xlabel("Data")
ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax_dd.legend(loc="lower left", framealpha=0.92)

plt.savefig("fig7_equity_drawdown.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 8 — Heatmap de retornos mensais                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 8 — Heatmap mensal...")

monthly = perf_df['strategy_returns'].resample('ME').apply(
    lambda r: (1 + r).prod() - 1
)
monthly_bench = perf_df['benchmark_returns'].resample('ME').apply(
    lambda r: (1 + r).prod() - 1
)

pivot = monthly.to_frame('ret')
pivot['year']  = pivot.index.year
pivot['month'] = pivot.index.month
heatmap_data = pivot.pivot(index='year', columns='month', values='ret')
heatmap_data.columns = ['Jan','Fev','Mar','Abr','Mai','Jun',
                        'Jul','Ago','Set','Out','Nov','Dez']

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("#F8F9FA")

vmax = max(abs(heatmap_data.values[np.isfinite(heatmap_data.values)]).max(), 0.01)
cmap = plt.cm.RdYlGn

im = ax.imshow(heatmap_data.values, cmap=cmap, aspect='auto',
               vmin=-vmax, vmax=vmax)

ax.set_xticks(range(12))
ax.set_xticklabels(heatmap_data.columns, fontsize=10)
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index, fontsize=10)

# Valores dentro das células
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data.values[i, j]
        if np.isfinite(val):
            cor_txt = "white" if abs(val) > vmax * 0.5 else "#222222"
            ax.text(j, i, f"{val:.1%}", ha='center', va='center',
                    fontsize=8.0, color=cor_txt, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Retorno Mensal", fontsize=10)
cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

ax.set_title("Retornos Mensais da Estratégia CERBERUS — Calendário de Performance")
ax.set_xlabel("Mês")
ax.set_ylabel("Ano")
plt.tight_layout()
plt.savefig("fig8_heatmap_mensal.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 9 — Rolling Sharpe (252 dias)                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 9 — Rolling Sharpe...")

window_sharpe = 252

# Calcula o rolling sharpe sobre o perf_df completo para ter janela cheia,
# mas plota APENAS o período OOS (evita o trecho vazio pré-2019)
roll_mean = perf_df['strategy_returns'].rolling(window_sharpe).mean()
roll_std  = perf_df['strategy_returns'].rolling(window_sharpe).std()
roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)

roll_mean_b = perf_df['benchmark_returns'].rolling(window_sharpe).mean()
roll_std_b  = perf_df['benchmark_returns'].rolling(window_sharpe).std()
roll_sharpe_b = (roll_mean_b / roll_std_b) * np.sqrt(252)

# Filtra apenas OOS para plotagem
rs_plot   = roll_sharpe[perf_df.index >= split]
rs_b_plot = roll_sharpe_b[perf_df.index >= split]

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor("#F8F9FA")

ax.fill_between(rs_plot.index, rs_plot.values, 0,
                where=(rs_plot.values >= 0),
                alpha=0.20, color="#27AE60")
ax.fill_between(rs_plot.index, rs_plot.values, 0,
                where=(rs_plot.values < 0),
                alpha=0.20, color="#E74C3C")

ax.plot(rs_plot.index, rs_plot.values,
        color=COR_STRAT, linewidth=1.8,
        label="Sharpe Móvel — CERBERUS")
ax.plot(rs_b_plot.index, rs_b_plot.values,
        color=COR_BENCH, linewidth=1.2, linestyle="--",
        label="Sharpe Móvel — IBOV")

ax.axhline(0, color="#888888", linewidth=0.8, linestyle="-")
ax.axhline(1, color="#27AE60", linewidth=0.8, linestyle=":",
           label="Sharpe = 1 (referência)")

for nome, data in CRISES.items():
    d = pd.to_datetime(data)
    if d >= split:
        ax.axvline(d, color="#7F8C8D", linestyle=":", linewidth=1.0, alpha=0.6)

ax.set_title(f"Sharpe Ratio Móvel ({window_sharpe} dias) — Qualidade do Sinal ao Longo do Tempo")
ax.set_xlabel("Data")
ax.set_ylabel("Índice de Sharpe Anualizado")
ax.legend(loc="upper left", framealpha=0.92)
plt.tight_layout()
plt.savefig("fig9_rolling_sharpe.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 10 — Distribuição retornos estratégia vs IBOV                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 10 — Distribuição estratégia vs IBOV...")

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#F8F9FA")

r_s = perf_df['strategy_returns'].dropna()
r_b = perf_df['benchmark_returns'].dropna()

x_range = np.linspace(
    min(r_s.quantile(0.005), r_b.quantile(0.005)),
    max(r_s.quantile(0.995), r_b.quantile(0.995)),
    400
)

# Histogramas
ax.hist(r_b, bins=80, density=True, color=COR_BENCH, alpha=0.30,
        edgecolor='white', linewidth=0.3, label="IBOV (histograma)")
ax.hist(r_s, bins=80, density=True, color=COR_STRAT, alpha=0.30,
        edgecolor='white', linewidth=0.3, label="CERBERUS (histograma)")

# KDEs
kde_s = gaussian_kde(r_s, bw_method=0.3)
kde_b = gaussian_kde(r_b, bw_method=0.3)
ax.plot(x_range, kde_s(x_range), color=COR_STRAT, linewidth=2.2,
        label="CERBERUS (KDE)")
ax.plot(x_range, kde_b(x_range), color=COR_BENCH, linewidth=2.0,
        linestyle="--", label="IBOV (KDE)")

ax.axvline(0, color="#AAAAAA", linewidth=0.8, linestyle=":")

# Box de estatísticas
stats_s = (f"CERBERUS\n"
           f"μ = {r_s.mean():.3%}  σ = {r_s.std():.3%}\n"
           f"Kurt = {r_s.kurt():.2f}  Skew = {r_s.skew():.2f}")
stats_b = (f"IBOV\n"
           f"μ = {r_b.mean():.3%}  σ = {r_b.std():.3%}\n"
           f"Kurt = {r_b.kurt():.2f}  Skew = {r_b.skew():.2f}")

ax.text(0.02, 0.97, stats_s, transform=ax.transAxes, fontsize=8.5,
        va='top', color=COR_STRAT,
        bbox=dict(fc='white', ec=COR_STRAT, alpha=0.88, pad=5, lw=1.2))
ax.text(0.02, 0.74, stats_b, transform=ax.transAxes, fontsize=8.5,
        va='top', color=COR_BENCH,
        bbox=dict(fc='white', ec=COR_BENCH, alpha=0.88, pad=5, lw=1.2))

ax.set_title("Distribuição dos Retornos Diários — CERBERUS vs IBOV")
ax.set_xlabel("Retorno Diário")
ax.set_ylabel("Densidade")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
ax.legend(loc="upper right", framealpha=0.92)
plt.tight_layout()
plt.savefig("fig10_distribuicao.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TABELA DE MÉTRICAS ESTENDIDAS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 72)
print("  MÉTRICAS ESTENDIDAS — CERBERUS".center(72))
print("═" * 72)

def metricas(ret_series, nome_periodo, nome_strat):
    r = ret_series.dropna()

    if len(r) < 2:
        print(f"\n  [{nome_strat}] — {nome_periodo}")
        print(f"  {'─'*66}")
        print(f"  Dados insuficientes para calcular métricas.")
        print(f"  {'─'*66}")
        return

    # Reconstrói o equity a partir dos retornos do período (rebase a 1.0)
    eq = (1 + r).cumprod()

    anos    = len(r) / 252
    cum     = eq.iloc[-1] - 1
    cagr    = eq.iloc[-1] ** (1 / anos) - 1 if anos > 0 else 0
    vol     = r.std() * np.sqrt(252)
    sharpe  = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    neg     = r[r < 0]
    sortino = (r.mean() / neg.std()) * np.sqrt(252) if len(neg) > 1 and neg.std() > 0 else 0
    mdd     = (eq / eq.cummax() - 1).min()
    calmar  = cagr / abs(mdd) if mdd != 0 else 0
    hit     = (r[r != 0] > 0).mean()

    print(f"\n  [{nome_strat}] — {nome_periodo}")
    print(f"  {'─'*66}")
    print(f"  {'Retorno Acumulado':<30} {cum:>12.2%}")
    print(f"  {'CAGR (Retorno Anualizado)':<30} {cagr:>12.2%}")
    print(f"  {'Volatilidade Anualizada':<30} {vol:>12.2%}")
    print(f"  {'Índice de Sharpe':<30} {sharpe:>12.2f}")
    print(f"  {'Índice de Sortino':<30} {sortino:>12.2f}")
    print(f"  {'Calmar Ratio':<30} {calmar:>12.2f}")
    print(f"  {'Max Drawdown':<30} {mdd:>12.2%}")
    print(f"  {'Hit Rate (dias com retorno)':<30} {hit:>12.1%}")
    print(f"  {'Dias totais':<30} {len(r):>12,}")
    print(f"  {'─'*66}")

split_pd = pd.to_datetime('2019-01-01')
in_s  = perf_df[perf_df.index < split_pd]
out_s = perf_df[perf_df.index >= split_pd]

for periodo, df_p in [("TOTAL (2010–2025)", perf_df),
                       ("IN-SAMPLE (2010–2018)", in_s),
                       ("OUT-OF-SAMPLE (2019–2025)", out_s)]:
    metricas(df_p['strategy_returns'],  periodo, "CERBERUS")
    metricas(df_p['benchmark_returns'], periodo, "IBOV B&H")

print("\n" + "═" * 72)
print("  Gráficos salvos: fig1 a fig10 (.png) na pasta CERBERUS")
print("═" * 72)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 11 — Ativação dos Sinais de Controle de Exposição (OOS)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\nGerando Figure 11 — Sinais de Controle de Exposição...")

# ── Reconstrução dos sinais dia a dia (OOS) ───────────────────────────────
# Usa o ibov completo para o rolling, mas plota apenas OOS
ibov_full = pipeline.ibov.copy()

THRESH_ENTROPY  = 0.6   # mesmo valor do backtester
WINDOW_ROLL     = 252   # janela rolling para percentis

# Sinal 1 — Entropia HMM
ibov_full['sig_entropy'] = (ibov_full['hmm_entropy'] > THRESH_ENTROPY).astype(float)
ibov_full['sig_entropy'] = ibov_full['sig_entropy'].where(ibov_full['hmm_entropy'].notna())

# Sinal 2 — Conectividade (density > p80 rolling) — mesmo cálculo do backtester
ibov_full['density_p80'] = ibov_full['density'].rolling(WINDOW_ROLL, min_periods=63).quantile(0.80)
ibov_full['sig_density'] = (ibov_full['density'] > ibov_full['density_p80']).astype(float)
ibov_full['sig_density'] = ibov_full['sig_density'].where(ibov_full['density'].notna())

# Sinal 3 — Vol-of-Vol (p80 rolling — indicador visual; não reduz exposure no backtester atual)
ibov_full['vov_p80']  = ibov_full['vol_of_vol'].rolling(WINDOW_ROLL, min_periods=63).quantile(0.80)
ibov_full['sig_vov']  = (ibov_full['vol_of_vol'] > ibov_full['vov_p80']).astype(float)
ibov_full['sig_vov']  = ibov_full['sig_vov'].where(ibov_full['vol_of_vol'].notna())

# Exposure modifier resultante (replica a lógica do backtester)
ibov_full['exposure'] = 1.0
ibov_full.loc[ibov_full['sig_entropy'] == 1, 'exposure'] *= 0.5
ibov_full.loc[ibov_full['sig_density'] == 1, 'exposure'] *= 0.5

# Filtra OOS para plotagem
oos = ibov_full[ibov_full.index >= split].copy()

# Estatísticas de ativação
n_total   = len(oos.dropna(subset=['hmm_entropy']))
n_entropy = int(oos['sig_entropy'].sum())
n_density = int(oos['sig_density'].sum())
n_vov     = int(oos['sig_vov'].sum())
n_quarter = int((oos['exposure'] == 0.25).sum())
n_half    = int((oos['exposure'] == 0.5).sum())
n_full    = int((oos['exposure'] == 1.0).sum())

print(f"  Entropia ativa:    {n_entropy:>4} dias ({n_entropy/n_total:.1%})")
print(f"  Density ativa:     {n_density:>4} dias ({n_density/n_total:.1%})")
print(f"  VoV ativa:         {n_vov:>4} dias ({n_vov/n_total:.1%})")
print(f"  Exposure 100%:     {n_full:>4} dias | 50%: {n_half} | 25%: {n_quarter}")

# ── Layout: 5 painéis compartilhando eixo X ──────────────────────────────
fig = plt.figure(figsize=(17, 14))
fig.patch.set_facecolor("#F8F9FA")
gs11 = GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 1], hspace=0.06)

ax_exp   = fig.add_subplot(gs11[0])
ax_ent   = fig.add_subplot(gs11[1], sharex=ax_exp)
ax_den   = fig.add_subplot(gs11[2], sharex=ax_exp)
ax_vov   = fig.add_subplot(gs11[3], sharex=ax_exp)
ax_eq_s  = fig.add_subplot(gs11[4], sharex=ax_exp)

# ── Painel 0: Exposure Modifier ──────────────────────────────────────────
# Faixas coloridas por nível de exposure
for i in range(len(oos) - 1):
    exp = oos['exposure'].iloc[i]
    if pd.isna(exp):
        continue
    cor = {1.0: "#27AE60", 0.5: "#F39C12", 0.25: "#E74C3C"}.get(exp, "#AAAAAA")
    ax_exp.axvspan(oos.index[i], oos.index[i+1], color=cor, alpha=0.35, linewidth=0)

# Linha do exposure
ax_exp.step(oos.index, oos['exposure'], where='post',
            color="#1C252E", linewidth=1.4, label="Exposure Modifier")
ax_exp.set_ylim(-0.05, 1.15)
ax_exp.set_yticks([0.25, 0.5, 1.0])
ax_exp.set_yticklabels(["25% (duplo\nbloqueio)", "50% (um\nbloqueio)", "100% (livre)"], fontsize=8)
ax_exp.set_ylabel("Exposição\nao Mercado", fontsize=9)
ax_exp.set_title("Ativação dos Sinais de Controle de Exposição — CERBERUS (Out-of-Sample)",
                 fontsize=13, fontweight='bold', pad=10)

# Legenda de exposição
patches_exp = [
    mpatches.Patch(color="#27AE60", alpha=0.5, label=f"100% — livre ({n_full} dias)"),
    mpatches.Patch(color="#F39C12", alpha=0.5, label=f"50% — 1 sinal ativo ({n_half} dias)"),
    mpatches.Patch(color="#E74C3C", alpha=0.5, label=f"25% — 2 sinais ativos ({n_quarter} dias)"),
]
ax_exp.legend(handles=patches_exp, loc="lower left", fontsize=8.5, framealpha=0.92, ncol=3)
plt.setp(ax_exp.get_xticklabels(), visible=False)

# ── Painel 1: Entropia HMM ────────────────────────────────────────────────
ax_ent.plot(oos.index, oos['hmm_entropy'], color="#8E44AD", linewidth=1.0, alpha=0.85,
            label="Entropia HMM")
ax_ent.axhline(THRESH_ENTROPY, color="#C0392B", linewidth=1.2, linestyle="--",
               label=f"Limiar ({THRESH_ENTROPY})")
ax_ent.fill_between(oos.index, oos['hmm_entropy'], THRESH_ENTROPY,
                    where=(oos['hmm_entropy'] > THRESH_ENTROPY),
                    color="#8E44AD", alpha=0.30, label=f"Ativo ({n_entropy} dias)")
ax_ent.set_ylabel("Entropia\nHMM", fontsize=9)
ax_ent.set_ylim(0, oos['hmm_entropy'].max() * 1.15)
ax_ent.legend(loc="upper right", fontsize=8, framealpha=0.92, ncol=3)
plt.setp(ax_ent.get_xticklabels(), visible=False)

# ── Painel 2: Conectividade (Density) ────────────────────────────────────
ax_den.plot(oos.index, oos['density'],     color="#2980B9", linewidth=1.0, alpha=0.85,
            label="Density")
ax_den.plot(oos.index, oos['density_p80'], color="#C0392B", linewidth=1.2, linestyle="--",
            label="P80 rolling 252d")
ax_den.fill_between(oos.index, oos['density'], oos['density_p80'],
                    where=(oos['density'] > oos['density_p80']),
                    color="#2980B9", alpha=0.30, label=f"Ativo ({n_density} dias)")
ax_den.set_ylabel("Conectividade\n(Density)", fontsize=9)
ax_den.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax_den.legend(loc="upper right", fontsize=8, framealpha=0.92, ncol=3)
plt.setp(ax_den.get_xticklabels(), visible=False)

# ── Painel 3: Vol-of-Vol ─────────────────────────────────────────────────
ax_vov.plot(oos.index, oos['vol_of_vol'], color="#E67E22", linewidth=1.0, alpha=0.85,
            label="Vol-of-Vol")
ax_vov.plot(oos.index, oos['vov_p80'],   color="#C0392B", linewidth=1.2, linestyle="--",
            label="P80 rolling 252d")
ax_vov.fill_between(oos.index, oos['vol_of_vol'], oos['vov_p80'],
                    where=(oos['vol_of_vol'] > oos['vov_p80']),
                    color="#E67E22", alpha=0.30,
                    label=f"Acima P80 ({n_vov} dias) [visual]")
ax_vov.set_ylabel("Vol-of-Vol", fontsize=9)
ax_vov.legend(loc="upper right", fontsize=8, framealpha=0.92, ncol=3)
plt.setp(ax_vov.get_xticklabels(), visible=False)

# ── Painel 4: Equity CERBERUS (contexto) ─────────────────────────────────
oos_perf = perf_df[perf_df.index >= split].copy()
cum_s = (1 + oos_perf['strategy_returns']).cumprod()
cum_b = (1 + oos_perf['benchmark_returns']).cumprod()
ax_eq_s.plot(oos_perf.index, cum_s, color=COR_STRAT, linewidth=1.4, label="CERBERUS")
ax_eq_s.plot(oos_perf.index, cum_b, color=COR_BENCH, linewidth=1.0,
             linestyle="--", alpha=0.7, label="IBOV B&H")
ax_eq_s.set_ylabel("Retorno\nAcumulado", fontsize=9)
ax_eq_s.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
ax_eq_s.set_xlabel("Data", fontsize=10)
ax_eq_s.legend(loc="upper left", fontsize=8, framealpha=0.92, ncol=2)

# Crises em todos os painéis
for ax in [ax_exp, ax_ent, ax_den, ax_vov, ax_eq_s]:
    for nome, data in CRISES.items():
        d = pd.to_datetime(data)
        if d >= split:
            ax.axvline(d, color="#7F8C8D", linestyle=":", linewidth=0.9, alpha=0.6)

plt.savefig("fig11_sinais_controle.png", dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG 12 — Desenvolvimento da Vol-of-Vol ao Longo do Tempo                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 12 — Vol-of-Vol ao longo do tempo...")

vov_full = ibov_full[['vol_of_vol', 'volatility', 'vov_p80', 'hmm_state']].dropna(subset=['vol_of_vol'])

fig, axes = plt.subplots(3, 1, figsize=(17, 11),
                         gridspec_kw={'height_ratios': [2.5, 1.2, 1.2]},
                         sharex=True)
fig.patch.set_facecolor("#F8F9FA")
ax_vov_main, ax_vol, ax_reg = axes

# ── Painel 0: Vol-of-Vol principal + bandas por regime ───────────────────
for estado, cor in cor_regime.items():
    mask = vov_full['hmm_state'] == estado
    ax_vov_main.fill_between(vov_full.index, 0, vov_full['vol_of_vol'],
                             where=mask, color=cor, alpha=0.22, linewidth=0)

ax_vov_main.plot(vov_full.index, vov_full['vol_of_vol'],
                 color="#1C252E", linewidth=1.3, alpha=0.9, label="Vol-of-Vol (diária)")

# Média móvel 63 dias (suavização trimestral)
vov_ma63 = vov_full['vol_of_vol'].rolling(63).mean()
ax_vov_main.plot(vov_full.index, vov_ma63,
                 color="#E74C3C", linewidth=1.8, linestyle="-",
                 label="Média móvel 63d (trimestral)")

# P80 rolling
ax_vov_main.plot(vov_full.index, vov_full['vov_p80'],
                 color="#C0392B", linewidth=1.2, linestyle="--",
                 label="P80 rolling 252d (limiar de alerta)")

# Linha do split
ax_vov_main.axvline(split, color="#C0392B", linewidth=1.5, linestyle="--",
                    alpha=0.7, label="Início do Teste (2019)")

# Anotações de crises
for nome, data in CRISES.items():
    d = pd.to_datetime(data)
    if d in vov_full.index or vov_full.index.searchsorted(d) < len(vov_full):
        ymax = vov_full['vol_of_vol'].max()
        ax_vov_main.axvline(d, color="#7F8C8D", linestyle=":", linewidth=0.9, alpha=0.7)
        ax_vov_main.text(d, ymax * 0.97, nome, fontsize=7, color="#555",
                         ha='center', va='top',
                         bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))

# Legenda de regimes
patches_reg = [mpatches.Patch(color=cor_regime[k], alpha=0.4, label=label_map[k])
               for k in cor_regime]
handles_main, labels_main = ax_vov_main.get_legend_handles_labels()
ax_vov_main.legend(handles=handles_main + patches_reg,
                   loc="upper left", fontsize=8.5, framealpha=0.92, ncol=3)

ax_vov_main.set_ylabel("Vol-of-Vol (σ da volatilidade)", fontsize=10)
ax_vov_main.set_title("Desenvolvimento da Volatilidade da Volatilidade (Vol-of-Vol) — 2010 a 2025",
                      fontsize=13, fontweight='bold', pad=10)
plt.setp(ax_vov_main.get_xticklabels(), visible=False)

# ── Painel 1: Volatilidade base (contexto) ────────────────────────────────
ax_vol.fill_between(vov_full.index, vov_full['volatility'],
                    color="#2980B9", alpha=0.30)
ax_vol.plot(vov_full.index, vov_full['volatility'],
            color="#2980B9", linewidth=1.2, label="Volatilidade anualizada (21d)")
ax_vol.axhline(vov_full['volatility'].mean(), color="#C0392B",
               linewidth=1.0, linestyle="--",
               label=f"Média histórica ({vov_full['volatility'].mean():.1%})")
ax_vol.axvline(split, color="#C0392B", linewidth=1.2, linestyle="--", alpha=0.6)
ax_vol.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax_vol.set_ylabel("Volatilidade\n(anualizada)", fontsize=9)
ax_vol.legend(loc="upper left", fontsize=8.5, framealpha=0.92, ncol=2)
for d_str in CRISES.values():
    ax_vol.axvline(pd.to_datetime(d_str), color="#7F8C8D",
                   linestyle=":", linewidth=0.9, alpha=0.6)
plt.setp(ax_vol.get_xticklabels(), visible=False)

# ── Painel 2: Razão VoV / Volatilidade (normalização) ─────────────────────
# Quanto a VoV representa da volatilidade base — alto = regime instável
ratio = (vov_full['vol_of_vol'] / vov_full['volatility']).replace([np.inf, -np.inf], np.nan)
ratio_ma = ratio.rolling(21).mean()

ax_reg.plot(ratio.index, ratio, color="#95A5A6", linewidth=0.8, alpha=0.5)
ax_reg.plot(ratio_ma.index, ratio_ma, color="#8E44AD", linewidth=1.6,
            label="Razão VoV / Volatilidade (MA21)")
ax_reg.axhline(ratio.mean(), color="#C0392B", linewidth=1.0, linestyle="--",
               label=f"Média ({ratio.mean():.2f})")
ax_reg.axhline(ratio.quantile(0.80), color="#E67E22", linewidth=1.0, linestyle=":",
               label=f"P80 ({ratio.quantile(0.80):.2f})")
ax_reg.axvline(split, color="#C0392B", linewidth=1.2, linestyle="--", alpha=0.6)
ax_reg.set_ylabel("VoV / Vol\n(instabilidade)", fontsize=9)
ax_reg.set_xlabel("Data", fontsize=10)
ax_reg.legend(loc="upper left", fontsize=8.5, framealpha=0.92, ncol=3)
for d_str in CRISES.values():
    ax_reg.axvline(pd.to_datetime(d_str), color="#7F8C8D",
                   linestyle=":", linewidth=0.9, alpha=0.6)

plt.savefig("fig12_vol_of_vol.png", dpi=180, bbox_inches='tight')
plt.show()

print("\n" + "═" * 72)
print("  Gráficos salvos: fig1 a fig12 (.png) na pasta CERBERUS")
print("═" * 72)