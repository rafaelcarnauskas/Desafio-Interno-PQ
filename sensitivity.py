"""
Sensitivity Analysis across Three Sub-Periods — DIPQ Strategy

Methodology
-----------
1. Train GMM-HMM once on the full IS period (2010-2019), same as DIPQ.py.
2. Split IS into three equal sub-periods (P1, P2, P3 — ~3 years each).
3. For each sub-period run one-at-a-time (OAT) sensitivity:
     - sweep each non-HMM parameter through 3-5 values
     - all other parameters held at BASE_PARAMS
4. Robustness analysis via a table + score approach:
     Robustness Score = mean(Sharpe) / (std(Sharpe) + 0.01)
     High score ⇒ consistently good across all three periods.
5. Outputs
     - printed tables: value × period (Sharpe / CAGR / MaxDD)
     - summary table: best value per period + most robust value
     - heatmap grid: period (row) × value (col), colored by Sharpe
     - multi-line plot: Sharpe vs value, one line per period
     - robustness bar chart: one bar per parameter (score of best value)
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from DIPQ import QuantStrategyPipeline, CDI_HAIRCUT

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

START_DATE = "2010-01-04"
END_DATE   = "2025-12-03"
SPLIT_DATE = "2019-01-01"

# Three roughly-equal thirds of the IS period
SUB_PERIODS = [
    ("P1 (2010-2013)", "2010-01-04", "2013-01-01"),
    ("P2 (2013-2016)", "2013-01-01", "2016-01-01"),
    ("P3 (2016-2019)", "2016-01-01", "2019-01-01"),
]

BASE_PARAMS = {
    "stop_loss_pct":      0.10,
    "trailing_stop_pct":  0.15,
    "entropy_threshold":  0.40,
    "density_quantile":   0.80,
    "threshold_net":      0.80,
    "exposure_reduction": 0.50,
    "rsi_oversold":       25,
    "rsi_overbought":     65,
    "max_position_pct":   0.10,
    "initial_capital":    100_000,
}

# 3-5 values per parameter — spans meaningful range around the base value
PARAM_GRIDS = {
    "stop_loss_pct":      [0.05, 0.075, 0.10, 0.125, 0.15],
    "trailing_stop_pct":  [0.10, 0.125, 0.15, 0.175, 0.20],
    "entropy_threshold":  [0.30, 0.40, 0.50, 0.60],
    "density_quantile":   [0.60, 0.70, 0.80, 0.90],
    "threshold_net":      [0.50, 0.65, 0.80],
    "exposure_reduction": [0.40, 0.50, 0.60],
    "rsi_oversold":       [15, 20, 25, 30],
    "rsi_overbought":     [55, 60, 65, 70],
    "max_position_pct":   [0.025, 0.05, 0.075, 0.10],
}

PARAM_HUMAN = {
    "stop_loss_pct":      "Stop Loss",
    "trailing_stop_pct":  "Trailing Stop",
    "entropy_threshold":  "Limiar Entropia",
    "density_quantile":   "Quantil Densidade",
    "threshold_net":      "Limiar Correlação (ρ)",
    "exposure_reduction": "Fator Redução Exposição",
    "rsi_oversold":       "RSI Sobrevendido",
    "rsi_overbought":     "RSI Sobrecomprado",
    "max_position_pct":   "Posição Máx. (%)",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_network_density(b3_returns, window=21, threshold=0.70):
    """Replicates DIPQ net_analysis with a configurable correlation threshold."""
    df        = b3_returns.copy()
    cols      = df.columns.tolist()
    n         = len(cols)
    max_edges = comb(n, 2, exact=True)
    densities = [np.nan] * window

    for i in range(window, len(df)):
        corr  = df.iloc[i - window:i].corr().values
        edges = sum(
            1 for r in range(n) for c in range(r)
            if abs(corr[r, c]) >= threshold
        )
        densities.append(edges / max_edges if max_edges > 0 else 0.0)

    return pd.Series(densities, index=df.index, name="density")


def compute_metrics(equity_series):
    if len(equity_series) < 10:
        return {"sharpe": np.nan, "sortino": np.nan, "max_dd": np.nan}
    rets     = equity_series.pct_change().fillna(0)
    cum      = (1 + rets).cumprod()
    anos     = len(equity_series) / 252
    std      = rets.std()
    downside = rets[rets < 0].std()
    return {
        "sharpe":  (rets.mean() / std      * np.sqrt(252)) if std      > 0 else 0.0,
        "sortino": (rets.mean() / downside * np.sqrt(252)) if downside > 0 else 0.0,
        "max_dd":  (cum / cum.cummax() - 1).min() * 100,
    }


def robustness_score(sharpes):
    """mean(Sharpe) / (std(Sharpe) + 0.01) — rewards stable and high Sharpe."""
    arr = np.array([s for s in sharpes if not np.isnan(s)])
    if len(arr) == 0:
        return np.nan
    return arr.mean() / (arr.std() + 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ON A SUB-PERIOD
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_on_subperiod(ibov_full_is, b3_is, b3_sma_fast, b3_sma_slow,
                               b3_rsi, params, density_full_is,
                               period_start, period_end):
    """
    Runs trading logic on [period_start, period_end].

    ibov_full_is: full IS IBOV dataframe (contains hmm_state / hmm_entropy
                  computed by pipeline.fit_hmm() on the entire IS window).
    density_full_is: density Series for the full IS period — rolling quantile
                     is computed here (not just the sub-period) to avoid
                     losing warmup data at the start of each sub-window.
    Capital resets to initial_capital at the beginning of each sub-period.
    """
    stop_loss_pct     = params["stop_loss_pct"]
    trailing_stop_pct = params["trailing_stop_pct"]
    entropy_threshold = params["entropy_threshold"]
    density_quantile  = params["density_quantile"]
    exposure_reduction = params["exposure_reduction"]
    rsi_oversold      = int(params["rsi_oversold"])
    rsi_overbought    = int(params["rsi_overbought"])
    max_position_pct  = params["max_position_pct"]
    initial_capital   = params["initial_capital"]

    # Rolling density quantile on full IS → no NaN at sub-period start
    dens_aligned   = density_full_is.reindex(ibov_full_is.index)
    dens_rolling_q = dens_aligned.rolling(window=252).quantile(density_quantile)

    ibov_sub = ibov_full_is.loc[period_start:period_end]

    capital   = float(initial_capital)
    tickers   = b3_is.columns.tolist()
    positions = {t: {"qty": 0.0, "entry_price": 0.0, "highest_price": 0.0} for t in tickers}
    equity_dates, equity_curve = [], []

    for date, row in ibov_sub.iterrows():
        hmm_state = row.get("hmm_state")
        if pd.isna(hmm_state) or date not in b3_is.index:
            continue

        prices = b3_is.loc[date]

        # ── Exposure modifier ─────────────────────────────────────────
        exp_mod = 1.0
        if row["hmm_entropy"] > entropy_threshold:
            exp_mod *= exposure_reduction
        dens_val    = dens_aligned.get(date, np.nan)
        dens_thresh = dens_rolling_q.get(date, np.nan)
        if pd.notna(dens_val) and pd.notna(dens_thresh) and dens_val > dens_thresh:
            exp_mod *= exposure_reduction

        regime      = int(hmm_state)
        buy_signals = []

        for ticker in tickers:
            price = prices.get(ticker, np.nan)
            if pd.isna(price) or price <= 0:
                continue
            pos = positions[ticker]

            if pos["qty"] > 0:
                pos["highest_price"] = max(pos["highest_price"], price)
                if (price < pos["entry_price"]   * (1 - stop_loss_pct) or
                        price < pos["highest_price"] * (1 - trailing_stop_pct) or
                        regime == 2):
                    capital += pos["qty"] * price
                    positions[ticker] = {"qty": 0.0, "entry_price": 0.0, "highest_price": 0.0}
                    continue

            if pos["qty"] == 0:
                if regime == 0:
                    fast = b3_sma_fast.loc[date, ticker] if date in b3_sma_fast.index else np.nan
                    slow = b3_sma_slow.loc[date, ticker] if date in b3_sma_slow.index else np.nan
                    if pd.notna(fast) and pd.notna(slow) and fast > slow:
                        buy_signals.append(ticker)
                elif regime == 1:
                    rsi_val = b3_rsi.loc[date, ticker] if date in b3_rsi.index else np.nan
                    if pd.notna(rsi_val) and rsi_val < rsi_oversold:
                        buy_signals.append(ticker)
            else:
                if regime == 1:
                    rsi_val = b3_rsi.loc[date, ticker] if date in b3_rsi.index else np.nan
                    if pd.notna(rsi_val) and rsi_val > rsi_overbought:
                        capital += pos["qty"] * price
                        positions[ticker] = {"qty": 0.0, "entry_price": 0.0, "highest_price": 0.0}

        if buy_signals:
            deploy    = capital * exp_mod
            per_trade = min(deploy / len(buy_signals), initial_capital * max_position_pct)
            for ticker in buy_signals:
                if capital < per_trade:
                    break
                price = prices.get(ticker, np.nan)
                if pd.isna(price) or price <= 0:
                    continue
                positions[ticker] = {
                    "qty": per_trade / price,
                    "entry_price": price,
                    "highest_price": price,
                }
                capital -= per_trade

        cdi = row.get("cdi_retorno", 0.0)
        if pd.isna(cdi):
            cdi = 0.0
        capital *= (1 + cdi)

        pos_value = sum(
            pos["qty"] * prices.get(t, 0.0)
            for t, pos in positions.items()
            if not pd.isna(prices.get(t, np.nan))
        )
        equity_dates.append(date)
        equity_curve.append(capital + pos_value)

    return pd.Series(equity_curve, index=equity_dates, name="equity")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SWEEP — all parameters × all sub-periods
# ─────────────────────────────────────────────────────────────────────────────

def run_all_periods(pipeline, ibov_full_is, b3_is, b3_returns_full_is):
    """
    Returns nested dict:
        results[param_name][value][period_label] = metrics_dict
    """
    base_density = compute_network_density(
        b3_returns_full_is, window=21, threshold=BASE_PARAMS["threshold_net"]
    )

    results = {p: {v: {} for v in vals} for p, vals in PARAM_GRIDS.items()}
    total   = len(SUB_PERIODS) * sum(len(v) for v in PARAM_GRIDS.values())
    count   = 0

    for period_label, p_start, p_end in SUB_PERIODS:
        print(f"\n{'─'*64}")
        print(f"  Sub-período: {period_label}  ({p_start} → {p_end})")
        print(f"{'─'*64}")

        for param_name, values in PARAM_GRIDS.items():
            for val in values:
                params = BASE_PARAMS.copy()
                params[param_name] = val

                density_s = (
                    compute_network_density(b3_returns_full_is, window=21, threshold=val)
                    if param_name == "threshold_net"
                    else base_density
                )

                equity  = run_backtest_on_subperiod(
                    ibov_full_is, b3_is,
                    pipeline.b3_sma_fast, pipeline.b3_sma_slow, pipeline.b3_rsi,
                    params, density_s,
                    period_start=p_start, period_end=p_end,
                )
                metrics = compute_metrics(equity)
                results[param_name][val][period_label] = metrics

                count += 1
                if count % 5 == 0 or count == total:
                    print(
                        f"  [{count:>3}/{total}] {param_name}={str(val):<8}  "
                        f"Sharpe={metrics['sharpe']:+.2f}  "
                        f"Sortino={metrics['sortino']:+.2f}  "
                        f"MaxDD={metrics['max_dd']:.1f}%"
                    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# COMPILE SUMMARY MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(results):
    """
    Returns per-parameter summary with matrices and robustness scores.
    summary[param_name] = {
        "values", "period_labels",
        "sharpe_matrix"  (n_periods × n_values),
        "cagr_matrix", "dd_matrix",
        "robustness"     (n_values,),
        "best_per_period" (n_periods,),
        "most_robust_value", "most_robust_score",
    }
    """
    period_labels = [pl for pl, _, _ in SUB_PERIODS]
    summary = {}

    for param_name, val_dict in results.items():
        values = list(val_dict.keys())
        n_v    = len(values)
        n_p    = len(period_labels)

        shm  = np.full((n_p, n_v), np.nan)
        sotm = np.full_like(shm, np.nan)
        ddm  = np.full_like(shm, np.nan)

        for j, val in enumerate(values):
            for i, pl in enumerate(period_labels):
                m = val_dict[val].get(pl, {})
                shm[i, j]  = m.get("sharpe",  np.nan)
                sotm[i, j] = m.get("sortino", np.nan)
                ddm[i, j]  = m.get("max_dd",  np.nan)

        rob     = [robustness_score(shm[:, j])  for j in range(n_v)]
        rob_sot = [robustness_score(sotm[:, j]) for j in range(n_v)]

        best_per_period = []
        for i in range(n_p):
            row = shm[i, :]
            best_per_period.append(
                values[int(np.nanargmax(row))] if not np.all(np.isnan(row)) else None
            )

        best_idx = int(np.nanargmax(rob)) if not np.all(np.isnan(rob)) else 0

        summary[param_name] = {
            "values":             values,
            "period_labels":      period_labels,
            "sharpe_matrix":      shm,
            "sortino_matrix":     sotm,
            "dd_matrix":          ddm,
            "robustness":         rob,
            "robustness_sortino": rob_sot,
            "best_per_period":    best_per_period,
            "most_robust_value":  values[best_idx],
            "most_robust_score":  rob[best_idx],
        }

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# PRINTED TABLES
# ─────────────────────────────────────────────────────────────────────────────

def _metric_section(values, matrix, rob, base_val, most_robust_val,
                    label, period_labels, W, show_score=True, lower_is_better=False):
    """Prints one metric section (Sharpe, Sortino, or Max DD) inside a parameter block."""
    n_p = len(period_labels)
    score_col = f"  {'Score':>{6}}" if show_score else ""
    hdr = f"    {label}\n"
    hdr += f"  {'Valor':<{W}}"
    for pl in period_labels:
        hdr += f"  {pl:>{W+1}}"
    hdr += f"  {'Média':>{W}}  {'CV%':>{5}}" + score_col
    print(hdr)
    print(f"  {'─'*W}" + f"  {'─'*(W+1)}" * n_p + f"  {'─'*W}  {'─'*5}"
          + (f"  {'─'*6}" if show_score else ""))

    for j, val in enumerate(values):
        col_vals = matrix[:, j]
        mean_v   = np.nanmean(col_vals)
        std_v    = np.nanstd(col_vals)
        cv       = (std_v / abs(mean_v) * 100) if abs(mean_v) > 0.005 else np.inf
        marker   = " ◄ base"    if val == base_val         else ""
        marker2  = " ★ ROBUSTO" if val == most_robust_val  else ""

        row = f"  {str(val):<{W}}"
        for v in col_vals:
            row += f"  {v:>{W+1}.3f}"
        cv_str = f"{cv:>5.0f}" if np.isfinite(cv) else "   ∞  "
        row += f"  {mean_v:>{W}.3f}  {cv_str}"
        if show_score:
            row += f"  {rob[j]:>6.2f}"
        print(row + marker + marker2)
    print()


def print_tables(summary):
    period_labels = [pl for pl, _, _ in SUB_PERIODS]
    W = 9  # column width

    print("\n" + "=" * 86)
    print("  SENSIBILIDADE POR SUB-PERÍODO — TABELAS DETALHADAS  ".center(86, "="))
    print("=" * 86)

    for param_name, s in summary.items():
        values   = s["values"]
        shm      = s["sharpe_matrix"]
        sotm     = s["sortino_matrix"]
        ddm      = s["dd_matrix"]
        rob      = s["robustness"]
        rob_sot  = s["robustness_sortino"]
        base_val = BASE_PARAMS.get(param_name)
        human    = PARAM_HUMAN.get(param_name, param_name)

        print(f"\n  ╔{'═'*82}╗")
        print(f"  ║  {human:<45}  (base = {base_val}){'':>15}║")
        print(f"  ╚{'═'*82}╝")

        # ── Sharpe ───────────────────────────────────────────────────
        _metric_section(values, shm, rob, base_val, s["most_robust_value"],
                        "SHARPE (anualizado)", period_labels, W, show_score=True)

        # ── Sortino ──────────────────────────────────────────────────
        _metric_section(values, sotm, rob_sot, base_val, s["most_robust_value"],
                        "SORTINO (anualizado)", period_labels, W, show_score=True)

        # ── Max Drawdown ─────────────────────────────────────────────
        # No score column for DD (lower magnitude = better — reader interprets directly)
        _metric_section(values, ddm, None, base_val, s["most_robust_value"],
                        "MAX DRAWDOWN (%)", period_labels, W, show_score=False,
                        lower_is_better=True)

        best_row = f"  {'P-melhor (Sharpe)':<{W+8}}"
        for bv in s["best_per_period"]:
            best_row += f"  {str(bv):>{W+1}}"
        print(best_row)
        print(f"  → Valor mais robusto (Sharpe): {s['most_robust_value']}  "
              f"(Score = {s['most_robust_score']:.2f})")

    # ── Cross-parameter summary ────────────────────────────────────────────
    print("\n\n" + "=" * 82)
    print("  RESUMO FINAL — MELHOR VALOR POR PERÍODO + MAIS ROBUSTO  ".center(82, "="))
    print("=" * 82)

    col_p = 22
    col_v = 11
    hdr = f"  {'Parâmetro':<{col_p}}"
    for pl in period_labels:
        hdr += f"  {pl[:12]:>{col_v}}"
    hdr += f"  {'Robusto':>{col_v}}  {'Score':>6}"
    print(hdr)
    print("  " + "─" * (col_p + (col_v + 2) * (len(period_labels) + 1) + 9))

    for param_name, s in summary.items():
        row = f"  {PARAM_HUMAN.get(param_name, param_name):<{col_p}}"
        for bv in s["best_per_period"]:
            row += f"  {str(bv):>{col_v}}"
        row += f"  {str(s['most_robust_value']):>{col_v}}  {s['most_robust_score']:>6.2f}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

PERIOD_COLORS = ["#1f77b4", "#E74C3C", "#2ECC71"]


def plot_heatmap_grid(summary):
    """
    One row per parameter, three columns (Sharpe / CAGR / MaxDD).
    Each cell = sub-period × parameter value.
    """
    params     = list(summary.keys())
    n_params   = len(params)
    fig_height = max(5, n_params * 3.2)

    fig, axes = plt.subplots(
        n_params, 3,
        figsize=(18, fig_height),
        facecolor="#F0F2F5",
    )
    if n_params == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Sharpe (anualizado)", "Sortino (anualizado)", "Max Drawdown (%)"]
    data_keys  = ["sharpe_matrix",      "sortino_matrix",       "dd_matrix"]
    cmaps      = ["RdYlGn",             "RdYlGn",               "RdYlGn_r"]

    for row_idx, param_name in enumerate(params):
        s      = summary[param_name]
        values = s["values"]
        period_labels = s["period_labels"]

        for col_idx, (dkey, cmap, ctitle) in enumerate(zip(data_keys, cmaps, col_titles)):
            ax   = axes[row_idx, col_idx]
            data = s[dkey]
            ax.set_facecolor("#FAFBFC")

            vmin, vmax = np.nanmin(data), np.nanmax(data)
            im = ax.imshow(data, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.80, pad=0.02)

            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([str(v) for v in values], fontsize=8, rotation=30, ha="right")
            ax.set_yticks(range(len(period_labels)))
            ax.set_yticklabels(period_labels, fontsize=8)

            if col_idx == 0:
                ax.set_ylabel(PARAM_HUMAN.get(param_name, param_name), fontsize=9, fontweight="bold")
            if row_idx == 0:
                ax.set_title(ctitle, fontsize=10, fontweight="bold")

            for (r, c), val in np.ndenumerate(data):
                if not np.isnan(val):
                    # mark most-robust column with bold text
                    bv = values.index(s["most_robust_value"])
                    fw = "bold" if c == bv else "normal"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=7.5, fontweight=fw, color="#111",
                            bbox=dict(fc="white", alpha=0.38, pad=1, lw=0))

    fig.suptitle(
        "Heatmaps de Sensibilidade por Sub-Período\n"
        "Texto em negrito = valor mais robusto (★) do parâmetro",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fname = "sens_heatmap_grid.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"[Salvo] {fname}")
    plt.show()


def plot_multiline_sharpe(summary):
    """
    For each parameter: Sharpe vs value, one line per sub-period.
    Vertical dashed line = most robust value; dotted = base value.
    """
    params  = list(summary.keys())
    n       = len(params)
    n_cols  = 3
    n_rows  = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), facecolor="#F0F2F5")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, param_name in zip(axes_flat, params):
        ax.set_facecolor("#FAFBFC")
        s      = summary[param_name]
        values = s["values"]
        xs     = list(range(len(values)))   # integer positions for clean spacing

        for i, (pl, color) in enumerate(zip(s["period_labels"], PERIOD_COLORS)):
            ys = s["sharpe_matrix"][i, :]
            ax.plot(xs, ys, marker="o", color=color, lw=2.2, ms=7,
                    label=pl, zorder=3)

        # Most robust value — dashed vertical line
        rv = s["most_robust_value"]
        if rv in values:
            rx = values.index(rv)
            ax.axvline(rx, color="#555", lw=1.4, linestyle="--", alpha=0.7, zorder=1)
            ax.text(rx + 0.08, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.1,
                    f"★{rv}", fontsize=7.5, color="#333", va="bottom")

        # Base value — dotted vertical line
        bv = BASE_PARAMS.get(param_name)
        if bv in values and bv != rv:
            bx = values.index(bv)
            ax.axvline(bx, color="#F39C12", lw=1.1, linestyle=":", alpha=0.8, zorder=1)

        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in values], fontsize=9)
        ax.set_xlabel(PARAM_HUMAN.get(param_name, param_name), fontsize=9)
        ax.set_ylabel("Sharpe", fontsize=9)
        ax.set_title(PARAM_HUMAN.get(param_name, param_name), fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.88)
        ax.grid(True, linestyle=":", alpha=0.5)

    for ax in axes_flat[len(params):]:
        ax.set_visible(False)

    fig.suptitle(
        "Sharpe vs Valor do Parâmetro — por Sub-Período\n"
        "-- = mais robusto  ··· = base",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fname = "sens_multiline_sharpe.png"
    plt.savefig(fname, dpi=155, bbox_inches="tight")
    print(f"[Salvo] {fname}")
    plt.show()


def plot_robustness_summary(summary):
    """
    Horizontal bar chart: robustness score of the most robust value,
    one bar per parameter — sorted descending.
    Color: green > 2, orange 1-2, red < 1.
    """
    params  = list(summary.keys())
    scores  = [summary[p]["most_robust_score"] for p in params]
    best_v  = [summary[p]["most_robust_value"]  for p in params]
    labels  = [PARAM_HUMAN.get(p, p) for p in params]

    order  = sorted(range(len(params)), key=lambda i: scores[i], reverse=True)
    labels = [labels[i] for i in order]
    scores = [scores[i] for i in order]
    best_v = [best_v[i] for i in order]

    colors = [
        "#2ECC71" if s > 2.0 else "#F39C12" if s > 1.0 else "#E74C3C"
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(11, max(4, len(params) * 0.75 + 2)), facecolor="#F0F2F5")
    ax.set_facecolor("#F8F9FA")

    bars = ax.barh(labels, scores, color=colors, edgecolor="white", linewidth=0.8, alpha=0.85)
    for bar, bv, sc in zip(bars, best_v, scores):
        ax.text(
            sc + max(scores) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"valor={bv}   score={sc:.2f}",
            va="center", fontsize=9, color="#222",
        )

    ax.set_xlabel("Robustness Score  =  média(Sharpe)  /  (desvio(Sharpe) + 0.01)", fontsize=10)
    ax.set_title(
        "Robustez dos Parâmetros — Valor Mais Estável Across Sub-Períodos\n"
        "Verde > 2.0  ·  Laranja 1–2  ·  Vermelho < 1",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_xlim(left=0)

    plt.tight_layout()
    fname = "sens_robustness_summary.png"
    plt.savefig(fname, dpi=155, bbox_inches="tight")
    print(f"[Salvo] {fname}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  DIPQ — SENSIBILIDADE POR SUB-PERÍODO".center(64, "="))
    print("=" * 64)

    # ── 1. Build pipeline and compute features ────────────────────────────────
    pipeline = QuantStrategyPipeline(
        start_date=START_DATE,
        end_date=END_DATE,
        split_date=SPLIT_DATE,
    )

    # Silence GMM-HMM diagnostic plots
    pipeline.plot_gmm_hmm_states      = lambda model, df: None
    pipeline.plot_gmm_hmm_3d_animated = lambda model, df: None

    b3_net = pipeline.net_analysis(df=pipeline.b3_returns, window=21)
    pipeline.ibov["density"]         = b3_net["density"]
    pipeline.ibov["max_eigen_value"] = b3_net["max_eigen_value"]
    pipeline.calculate_features(window=21)

    # ── 2. Train GMM-HMM on full IS period (same as DIPQ.py) ─────────────────
    print("\n[HMM] Treinando GMM-HMM no período IS completo (2010→2019)...")
    pipeline.fit_hmm()

    # ── 3. Slice to IS ────────────────────────────────────────────────────────
    split_pd = pd.to_datetime(SPLIT_DATE)

    ibov_full_is       = pipeline.ibov.loc[:split_pd].copy()
    b3_is              = pipeline.b3.loc[:split_pd].copy()
    # Drop density/max_eigen_value that net_analysis added in-place to b3_returns —
    # those columns would corrupt the correlation matrix in compute_network_density.
    b3_returns_full_is = (
        pipeline.b3_returns
        .drop(columns=["density", "max_eigen_value"], errors="ignore")
        .loc[:split_pd]
        .copy()
    )

    print(f"\n[IS] {ibov_full_is.index[0].date()} → {ibov_full_is.index[-1].date()}"
          f"  ({len(ibov_full_is)} dias úteis)")
    for lbl, s, e in SUB_PERIODS:
        sub = ibov_full_is.loc[s:e]
        print(f"      {lbl}: {len(sub)} dias")

    # ── 4. Run sweep ──────────────────────────────────────────────────────────
    print("\n[Sweep] Iniciando varredura OAT nos 3 sub-períodos...")
    results = run_all_periods(pipeline, ibov_full_is, b3_is, b3_returns_full_is)

    # ── 5. Build summary ──────────────────────────────────────────────────────
    summary = build_summary(results)

    # ── 6. Tables ─────────────────────────────────────────────────────────────
    print_tables(summary)

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("\n[Plot] Gerando visualizações...")
    plot_heatmap_grid(summary)
    plot_multiline_sharpe(summary)
    plot_robustness_summary(summary)

    print("\n✓ Análise concluída.")


if __name__ == "__main__":
    main()
