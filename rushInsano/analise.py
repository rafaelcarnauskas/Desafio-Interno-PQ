"""
analise.py — Dashboard de Análise do rushInsano
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

from checkpoint import QuantStrategyPipeline, STOP_LOSS_PCT, TRAILING_STOP_PCT

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
    "Dilma\nImpeach.": "2016-04-17",
    "Joesley\nDay":    "2017-05-18",
    "COVID-19":        "2020-03-23",
}

# ── Executar pipeline ─────────────────────────────────────────────────────────
print("=" * 60)
print("  rushInsano — Dashboard de Análise")
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
# ║  FIG 2 — Entropia ao longo do tempo                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("Gerando Figure 2 — Entropia...")

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor("#F8F9FA")

ent = ibov['hmm_entropy'].dropna()

ax.fill_between(ent.index, ent.values, alpha=0.25, color="#2980B9")
ax.plot(ent.index, ent.values, color="#2980B9", linewidth=1.0)

# Threshold de operação
ax.axhline(1.2, color="#E74C3C", linestyle="--", linewidth=1.5,
           label="Threshold: H > 1.2 → exposição ÷ 2")

# Entropia máxima teórica
ax.axhline(np.log2(3), color="#95A5A6", linestyle=":", linewidth=1.0,
           label=f"Entropia máxima log₂(3) ≈ {np.log2(3):.2f}")

# Crises
for nome, data in CRISES.items():
    d = pd.to_datetime(data)
    ax.axvline(d, color="#7F8C8D", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(d, 1.52, nome, fontsize=7.5, color="#555555",
            ha='center', va='top',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))

# Área de alta incerteza
ax.fill_between(ent.index, 1.2, ent.values,
                where=(ent.values > 1.2),
                alpha=0.35, color="#E74C3C", label="Zona de alta incerteza")

ax.axvline(split, color="#C0392B", linestyle="--", linewidth=1.5, alpha=0.6)
ax.set_ylim(0, np.log2(3) + 0.1)
ax.set_title("Entropia de Shannon do HMM ao Longo do Tempo")
ax.set_xlabel("Data")
ax.set_ylabel("Entropia H (base 2)")
ax.legend(loc="upper right", framealpha=0.92)
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
ax.legend(loc="upper left", framealpha=0.92)
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

# Equity
ax_eq.plot(perf_df.index, perf_df['cum_strategy'],
           color=COR_STRAT, linewidth=2.0,
           label="rushInsano (HMM Multi-Ativos)")
ax_eq.plot(perf_df.index, perf_df['cum_benchmark'],
           color=COR_BENCH, linewidth=1.4, linestyle="--",
           label="Buy & Hold IBOV")

# Alpha fill
ax_eq.fill_between(perf_df.index,
                   perf_df['cum_strategy'], perf_df['cum_benchmark'],
                   where=(perf_df['cum_strategy'] >= perf_df['cum_benchmark']),
                   alpha=0.12, color="#27AE60", label="Alpha positivo")
ax_eq.fill_between(perf_df.index,
                   perf_df['cum_strategy'], perf_df['cum_benchmark'],
                   where=(perf_df['cum_strategy'] < perf_df['cum_benchmark']),
                   alpha=0.12, color="#E74C3C", label="Abaixo do benchmark")

ax_eq.axvline(split, color="#C0392B", linestyle="--", linewidth=1.5,
              label="Início do Teste (2019)")

# Métricas no gráfico
ret_total_s = perf_df['cum_strategy'].iloc[-1] - 1
ret_total_b = perf_df['cum_benchmark'].iloc[-1] - 1
ax_eq.text(0.01, 0.97,
           f"rushInsano: {ret_total_s:.1%} acumulado",
           transform=ax_eq.transAxes, fontsize=9.5,
           va='top', color=COR_STRAT,
           bbox=dict(fc='white', ec=COR_STRAT, alpha=0.88, pad=4, lw=1.2))
ax_eq.text(0.01, 0.88,
           f"IBOV B&H:   {ret_total_b:.1%} acumulado",
           transform=ax_eq.transAxes, fontsize=9.5,
           va='top', color=COR_BENCH,
           bbox=dict(fc='white', ec=COR_BENCH, alpha=0.88, pad=4, lw=1.2))

ax_eq.set_ylabel("Retorno Acumulado (1.0 = Capital Inicial)")
ax_eq.set_title("rushInsano vs Buy & Hold IBOV — Performance Acumulada")
ax_eq.legend(loc="upper left", framealpha=0.92, ncol=2)
ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
plt.setp(ax_eq.get_xticklabels(), visible=False)

# Drawdown
dd_strat = (perf_df['cum_strategy'] / perf_df['cum_strategy'].cummax() - 1) * 100
dd_bench = (perf_df['cum_benchmark'] / perf_df['cum_benchmark'].cummax() - 1) * 100

ax_dd.fill_between(perf_df.index, dd_strat, 0,
                   alpha=0.45, color=COR_STRAT, label="Drawdown rushInsano")
ax_dd.plot(perf_df.index, dd_bench, color=COR_BENCH,
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

ax.set_title("Retornos Mensais da Estratégia rushInsano — Calendário de Performance")
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
roll_mean = perf_df['strategy_returns'].rolling(window_sharpe).mean()
roll_std  = perf_df['strategy_returns'].rolling(window_sharpe).std()
roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)

roll_mean_b = perf_df['benchmark_returns'].rolling(window_sharpe).mean()
roll_std_b  = perf_df['benchmark_returns'].rolling(window_sharpe).std()
roll_sharpe_b = (roll_mean_b / roll_std_b) * np.sqrt(252)

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor("#F8F9FA")

ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                where=(roll_sharpe.values >= 0),
                alpha=0.20, color="#27AE60")
ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                where=(roll_sharpe.values < 0),
                alpha=0.20, color="#E74C3C")

ax.plot(roll_sharpe.index, roll_sharpe.values,
        color=COR_STRAT, linewidth=1.8,
        label="Sharpe Móvel — rushInsano")
ax.plot(roll_sharpe_b.index, roll_sharpe_b.values,
        color=COR_BENCH, linewidth=1.2, linestyle="--",
        label="Sharpe Móvel — IBOV")

ax.axhline(0, color="#888888", linewidth=0.8, linestyle="-")
ax.axhline(1, color="#27AE60", linewidth=0.8, linestyle=":",
           label="Sharpe = 1 (referência)")
ax.axvline(split, color="#C0392B", linestyle="--", linewidth=1.5, alpha=0.6,
           label="Início do Teste (2019)")

for nome, data in CRISES.items():
    d = pd.to_datetime(data)
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
        edgecolor='white', linewidth=0.3, label="rushInsano (histograma)")

# KDEs
kde_s = gaussian_kde(r_s, bw_method=0.3)
kde_b = gaussian_kde(r_b, bw_method=0.3)
ax.plot(x_range, kde_s(x_range), color=COR_STRAT, linewidth=2.2,
        label="rushInsano (KDE)")
ax.plot(x_range, kde_b(x_range), color=COR_BENCH, linewidth=2.0,
        linestyle="--", label="IBOV (KDE)")

ax.axvline(0, color="#AAAAAA", linewidth=0.8, linestyle=":")

# Box de estatísticas
stats_s = (f"rushInsano\n"
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

ax.set_title("Distribuição dos Retornos Diários — rushInsano vs IBOV")
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
print("  MÉTRICAS ESTENDIDAS — rushInsano".center(72))
print("═" * 72)

def metricas(ret_series, equity_series, nome_periodo, nome_strat):
    r  = ret_series.dropna()
    eq = equity_series.dropna()

    anos    = len(r) / 252
    cum     = eq.iloc[-1] / eq.iloc[0] - 1
    cagr    = (eq.iloc[-1] / eq.iloc[0]) ** (1 / anos) - 1 if anos > 0 else 0
    vol     = r.std() * np.sqrt(252)
    sharpe  = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    neg     = r[r < 0]
    sortino = (r.mean() / neg.std()) * np.sqrt(252) if neg.std() > 0 else 0
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
    metricas(df_p['strategy_returns'],
             df_p['cum_strategy'],
             periodo, "rushInsano")
    metricas(df_p['benchmark_returns'],
             df_p['cum_benchmark'],
             periodo, "IBOV B&H")

print("\n" + "═" * 72)
print("  Gráficos salvos: fig1 a fig10 (.png) na pasta rushInsano")
print("═" * 72)