from hmmlearn.hmm import GaussianHMM, GMMHMM
from bcb import sgs
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import entropy as scipy_entropy, skew as scipy_skew
from scipy.special import comb


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARÂMETROS                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

STOP_LOSS_PCT            = 0.15   # drawdown máximo desde o pico-
TRAILING_STOP_PCT        = 0.12   # recuo do equity desde o último pico 
THRESHOLD_net_analysis = 0.7

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARÂMETROS DO CAIXA (CDI)                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Fração do CDI que o caixa rende enquanto não está investido em ações.
# 0.90 = 90% do CDI — haircut conservador que absorve IOF implícito,
# come-cotas de fundo DI e spread de liquidez sem precisar modelar
# as regras exatas de prazo do IOF (que impactam resgates < 30 dias).
# Ajuste entre 0.85 (muito conservador) e 1.00 (CDI bruto, sem fricção).
CDI_HAIRCUT = 0.90

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARÂMETROS DE WALK-FORWARD                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# A cada RETRAIN_FREQ dias úteis no período de teste, o modelo é re-treinado
# com todos os dados disponíveis até aquele ponto (expanding window).
# Valores típicos: 63 (trimestral), 126 (semestral), 252 (anual).
RETRAIN_FREQ_DAYS = 126  # Semestral — bom equilíbrio entre adaptação e estabilidade


class QuantStrategyPipeline:
    def __init__(self, start_date, end_date, split_date='2019-01-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.split_date = split_date  # Data que divide o In-Sample do Out-of-Sample

        self.ibov = pd.read_csv(
            "ibov_2010_2025.csv",
            parse_dates=["Date"],
            index_col="Date"
        )

        self.ibov["Adj Close"] = pd.to_numeric(
            self.ibov["Adj Close"],
            errors="coerce"
        )

        self.ibov['returns'] = self.ibov['Adj Close'].pct_change(fill_method=None)

        self.b3 = pd.read_csv(
            "b3-2010-2025.csv",
            parse_dates=["Date"],
            index_col="Date"
        )

        self.b3 = self.b3.apply(pd.to_numeric, errors="coerce")

        self.b3_returns = self.b3.pct_change(fill_method=None)

        # --- CDI DIÁRIO (Série 12 do Banco Central) ---
        # O SGS do BCB limita séries diárias a 10 anos por chamada.
        # Quebramos em janelas de 9 anos para ter margem de segurança.
        from bcb import sgs

        start_dt = pd.to_datetime(self.start_date)
        end_dt   = pd.to_datetime(self.end_date)
        chunk_years = 9  # janela segura abaixo do limite de 10 anos

        chunks = []
        chunk_start = start_dt
        while chunk_start <= end_dt:
            chunk_end = min(chunk_start + pd.DateOffset(years=chunk_years), end_dt)
            print(f"[CDI] Baixando {chunk_start.date()} → {chunk_end.date()} ...")
            part = sgs.get({'cdi_pct': 12},
                            start=chunk_start.strftime('%Y-%m-%d'),
                            end=chunk_end.strftime('%Y-%m-%d'))
            chunks.append(part)
            chunk_start = chunk_end + pd.DateOffset(days=1)

        cdi_diario = pd.concat(chunks)
        cdi_diario = cdi_diario[~cdi_diario.index.duplicated(keep='first')]
        cdi_diario['cdi_retorno'] = (cdi_diario['cdi_pct'] / 100) * CDI_HAIRCUT

        self.ibov = self.ibov.join(cdi_diario['cdi_retorno'], how='left')
        self.ibov['cdi_retorno'] = self.ibov['cdi_retorno'].fillna(0)
        print(f"[CDI] Série carregada com haircut de {CDI_HAIRCUT:.0%}. "
                f"CDI médio efetivo: {self.ibov['cdi_retorno'].mean()*252:.2%} a.a.")

        self.b3.plot(title='b3')
        self.ibov.plot(title='ibov')

    #def fetch_data(self): (data é o ibov e net_data é o b3)

# =====================================================================
# ENGENHARIA DE FEATURES (VOV, REDE, ENTROPIA)
# =====================================================================

    def calculate_features(self, window=21):
        # --- FEATURES DE MERCADO (IBOV) ---
        self.ibov['volatility'] = self.ibov['returns'].rolling(window=window).std() * np.sqrt(252)
        self.ibov['vol_of_vol'] = self.ibov['volatility'].rolling(window=window).std()

        # --- FEATURES DOS ATIVOS (B3) ---
        self.b3_sma_fast = self.b3.rolling(window=10).mean()
        self.b3_sma_slow = self.b3.rolling(window=50).mean()

        # RSI Vetorial para o DataFrame inteiro
        delta = self.b3.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.b3_rsi = 100 - (100 / (1 + rs))

        self.ibov.plot()
        self.b3.plot

    # CONECTIVIDADE DE REDE
    def net_analysis(self, df=None, window=21):

        if df is None:
            df = self.b3_returns

        densities = [np.nan] * window
        max_eigen_values = [np.nan] * window

        columns = df.columns.tolist()
        n = len(columns)
        possible_edges = comb(n, 2, exact=True)

        for i in range(window, len(df)):
            corr = df.iloc[i-window:i].corr().values

            A = np.zeros((n, n))
            edges = []

            for r in range(n):
                for c in range(n):
                    if r == c:
                        A[r, c] = 0
                    else:
                        A[r, c] = corr[r, c] if abs(corr[r, c]) >= THRESHOLD_net_analysis else 0

                        if c < r and A[r, c]:
                            edges.append((columns[r], columns[c]))

            density = len(edges) / possible_edges if possible_edges > 0 else 0
            densities.append(density)

            eigen_values, eigen_vector = np.linalg.eigh(A)
            max_eigen_value = eigen_values[-1]
            max_eigen_values.append(max_eigen_value)

        df['density'] = densities
        df['max_eigen_value'] = max_eigen_values

        df.plot()

        return df

    def _calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

# =====================================================================
# SEÇÃO 3: MODELAGEM DE REGIMES (HMM) — SEM LOOK-AHEAD + WALK-FORWARD
# =====================================================================

    def _train_hmm_on_data(self, train_df):
        """
        Treina um GMMHMM nos dados fornecidos e devolve o modelo treinado
        junto com o mapeamento estado → nível de volatilidade (aprendido
        exclusivamente nesses dados).
        """
        X = train_df[['returns', 'volatility']].values
        states_raw = np.arange(len(X))  # placeholder para indexação

        model = GMMHMM(
            n_components=3, n_mix=2,
            covariance_type="full",
            n_iter=1000, random_state=42
        )
        model.fit(X)

        # Aprende o mapeamento estado → vol APENAS no conjunto de treino
        states_train = model.predict(X)
        state_vols = {
            i: train_df.iloc[states_train == i]['volatility'].mean()
            for i in range(3)
        }
        sorted_states = sorted(state_vols, key=state_vols.get)
        # 0 = baixa vol, 1 = média vol, 2 = alta vol
        state_map = {sorted_states[0]: 0, sorted_states[1]: 1, sorted_states[2]: 2}

        return model, state_map

    def _predict_online(self, model, state_map, obs):
        """
        ═══════════════════════════════════════════════════════════════
        CORREÇÃO 1 — ELIMINAÇÃO DO LOOK-AHEAD BIAS
        ═══════════════════════════════════════════════════════════════
        Usa o algoritmo FORWARD (filtragem causal) em vez de Viterbi
        global.  O Viterbi padrão (model.predict) passa pela sequência
        inteira de trás para frente, usando observações futuras para
        decidir o estado de hoje — look-ahead clássico.

        O forward pass calcula, para cada instante t, a distribuição
        p(state_t | obs_1..t) usando SOMENTE dados passados e presentes.
        O estado predito é o argmax dessa distribuição.

        Parâmetros
        ----------
        model    : GMMHMM já treinado
        state_map: dict {estado_raw → 0/1/2} aprendido no treino
        obs      : array (T, n_features) — sequência de observações

        Retorna
        -------
        states   : array (T,) com estados mapeados (0=baixa, 1=média, 2=alta)
        probs    : array (T, 3) probabilidades forward normalizadas por instante
        """
        T = len(obs)
        n_states = model.n_components

        # ── Parâmetros do modelo ──────────────────────────────────────
        startprob = model.startprob_          # (n_states,)
        transmat   = model.transmat_           # (n_states, n_states)

        # Calcula log-verossimilhança de cada observação em cada estado
        # usando a mistura gaussiana de cada estado (GMM).
        # framelogprob[t, j] = log p(obs_t | estado j)
        framelogprob = model._compute_log_likelihood(obs)  # (T, n_states)
        # Subtrai o máximo por linha antes de exp() para evitar overflow/underflow
        framelogprob_stable = framelogprob - framelogprob.max(axis=1, keepdims=True)
        frameprob = np.exp(framelogprob_stable)

        # ── Forward pass ─────────────────────────────────────────────
        alpha = np.zeros((T, n_states))  # alpha[t, j] = p(obs_1..t, state_t=j)

        # Inicialização (t=0)
        alpha[0] = startprob * frameprob[0]
        alpha[0] /= alpha[0].sum() + 1e-300  # normaliza para evitar underflow

        # Recursão (t=1..T-1) — usa APENAS dados passados
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ transmat) * frameprob[t]
            alpha[t] /= alpha[t].sum() + 1e-300

        # Estado mais provável em cada instante (causal)
        raw_states = alpha.argmax(axis=1)

        # Aplica o mapeamento aprendido no treino
        mapped_states = np.array([state_map.get(s, s) for s in raw_states])

        # Entropia de Shannon da distribuição forward (incerteza do modelo)
        probs_mapped = np.zeros((T, 3))
        for t in range(T):
            for raw, mapped in state_map.items():
                probs_mapped[t, mapped] += alpha[t, raw]

        return mapped_states, probs_mapped

    def fit_hmm(self):
        """
        ═══════════════════════════════════════════════════════════════
        CORREÇÃO 2 — WALK-FORWARD COM EXPANDING WINDOW
        ═══════════════════════════════════════════════════════════════
        Problema original: o modelo era treinado UMA VEZ no período
        2010-2019 e nunca re-aprendia.  Isso faz com que crises
        estruturalmente diferentes (ex.: 2024) não sejam reconhecidas,
        pois o modelo só conhece os regimes de volatilidade do treino.

        Solução: a cada RETRAIN_FREQ_DAYS dias úteis no período de
        teste, o modelo é re-treinado com TODOS os dados disponíveis
        até aquele ponto (expanding window).  O mapeamento
        estado→volatilidade é re-derivado no mesmo conjunto, eliminando
        qualquer vazamento de futuro.

        Fluxo
        -----
        1. Treino inicial:  2010 → split_date   (in-sample completo)
        2. Predição in-sample via forward (causal)
        3. No período de teste, a cada RETRAIN_FREQ_DAYS dias:
              a. Inclui os dados até a data atual no conjunto de treino
              b. Re-treina o GMMHMM (expanding window)
              c. Usa o forward pass para predizer o PRÓXIMO bloco
        ═══════════════════════════════════════════════════════════════
        """
        full_data = self.ibov.dropna(subset=['returns', 'volatility']).copy()
        split_date_pd = pd.to_datetime(self.split_date)

        # ── Separação inicial ─────────────────────────────────────────
        train_data = full_data.loc[:split_date_pd]
        test_data  = full_data.loc[split_date_pd:].iloc[1:]  # exclui o split_date em si

        print(f"[HMM] Treino inicial: {train_data.index[0].date()} → {train_data.index[-1].date()} ({len(train_data)} dias)")
        print(f"[HMM] Teste (OOS):    {test_data.index[0].date()} → {test_data.index[-1].date()} ({len(test_data)} dias)")

        # ── Treino inicial e predição in-sample ───────────────────────
        model, state_map = self._train_hmm_on_data(train_data)
        X_train = train_data[['returns', 'volatility']].values
        states_train, probs_train = self._predict_online(model, state_map, X_train)

        # ── Predição out-of-sample com walk-forward ───────────────────
        test_dates  = test_data.index.tolist()
        n_test      = len(test_dates)
        states_test = np.empty(n_test, dtype=int)
        probs_test  = np.zeros((n_test, 3))

        # Ponteiro do próximo re-treino
        next_retrain_idx = 0  # re-treina antes do primeiro bloco de teste

        i = 0
        current_model    = model
        current_state_map = state_map

        while i < n_test:
            # Verifica se é hora de re-treinar
            if i >= next_retrain_idx:
                # Dados disponíveis até test_dates[i] EXCLUSIVE (não vazamos o dia atual)
                cutoff = test_dates[i]
                expanding_train = full_data.loc[:cutoff].iloc[:-1]  # exclui o dia atual

                if len(expanding_train) >= 60:  # mínimo para treinar o GMM-HMM
                    print(f"  [Walk-Forward] Re-treinando em {cutoff.date()} "
                          f"(n={len(expanding_train)} dias)")
                    current_model, current_state_map = self._train_hmm_on_data(expanding_train)

                next_retrain_idx = i + RETRAIN_FREQ_DAYS

            # Define o bloco a predizer (até o próximo re-treino ou fim do teste)
            block_end = min(next_retrain_idx, n_test)
            block_slice = test_data.iloc[i:block_end]
            X_block = block_slice[['returns', 'volatility']].values

            s_block, p_block = self._predict_online(current_model, current_state_map, X_block)

            states_test[i:block_end] = s_block
            probs_test[i:block_end]  = p_block

            i = block_end

        # ── Monta o DataFrame consolidado ────────────────────────────
        max_entropy = np.log2(3) #normaliza a entropia para trabalhar com valores de 0 a 1

        train_result = pd.DataFrame({
            'hmm_state':   states_train,
            'hmm_entropy': [scipy_entropy(p, base=2)/ max_entropy for p in probs_train],
        }, index=train_data.index)

        test_result = pd.DataFrame({
            'hmm_state':   states_test,
            'hmm_entropy': [scipy_entropy(p, base=2)/ max_entropy for p in probs_test],
        }, index=test_data.index)

        all_results = pd.concat([train_result, test_result])

        self.ibov = self.ibov.join(all_results[['hmm_state', 'hmm_entropy']])
        self.ibov[['hmm_state', 'hmm_entropy']].plot()

# =====================================================================
# SEÇÃO 4 E 5: BACKTESTER (GERENCIAMENTO DE RISCO E ESTRATÉGIAS)
# =====================================================================

    def run_backtest(self, initial_capital=100000, stop_loss_pct=0.07, trailing_stop_pct=0.05):
        print("Rodando Backtest Multi-Ativos...")

        capital = initial_capital

        tickers = self.b3.columns
        positions = {ticker: {'qty': 0, 'entry_price': 0, 'highest_price': 0} for ticker in tickers}

        equity_curve = []
        equity_dates = []

        self.ibov['density_rolling_80'] = self.ibov['density'].rolling(window=252).quantile(0.8)

        split_date_pd = pd.to_datetime(self.split_date)
        df_teste = self.ibov.loc[split_date_pd:]

        for date, ibov_row in df_teste.iterrows():  # <--- Altere self.ibov para df_teste
            if pd.isna(ibov_row['hmm_state']) or date not in self.b3.index:
                continue

            current_b3_prices = self.b3.loc[date]

            # --- 1. MODULADOR DE EXPOSIÇÃO DO MERCADO ---
            exposure_modifier = 1.0
            if ibov_row['hmm_entropy'] > 0.6:
                exposure_modifier *= 0.5

            density_threshold = ibov_row.get('density_rolling_80')
            if pd.notna(ibov_row.get('density')) and pd.notna(density_threshold):
                if ibov_row['density'] > density_threshold:
                    exposure_modifier *= 0.5

            regime = ibov_row['hmm_state']
            buy_signals = []

            # --- 2. LOOP POR CADA AÇÃO PARA CHECAR SINAIS E RISCO ---
            for ticker in tickers:
                price = current_b3_prices[ticker]
                if pd.isna(price) or price <= 0:
                    continue

                pos = positions[ticker]

                if pos['qty'] > 0:
                    pos['highest_price'] = max(pos['highest_price'], price)

                    if (price < pos['entry_price'] * (1 - stop_loss_pct)) or \
                       (price < pos['highest_price'] * (1 - trailing_stop_pct)) or \
                       (regime == 2):

                        capital += pos['qty'] * price
                        positions[ticker] = {'qty': 0, 'entry_price': 0, 'highest_price': 0}
                        continue

                if pos['qty'] == 0:
                    if regime == 0:
                        if self.b3_sma_fast.loc[date, ticker] > self.b3_sma_slow.loc[date, ticker]:
                            buy_signals.append(ticker)

                    elif regime == 1:
                        if self.b3_rsi.loc[date, ticker] < 30:
                            buy_signals.append(ticker)
                else:
                    if regime == 1 and self.b3_rsi.loc[date, ticker] > 70:
                        capital += pos['qty'] * price
                        positions[ticker] = {'qty': 0, 'entry_price': 0, 'highest_price': 0}

            # --- 3. EXECUÇÃO DE COMPRAS ---
            if buy_signals:
                capital_to_deploy = capital * exposure_modifier
                per_trade_capital = min(capital_to_deploy / len(buy_signals), initial_capital * 0.10)

                for ticker in buy_signals:
                    if capital >= per_trade_capital:
                        price = current_b3_prices[ticker]
                        qty = per_trade_capital / price

                        positions[ticker]['qty'] = qty
                        positions[ticker]['entry_price'] = price
                        positions[ticker]['highest_price'] = price

                        capital -= per_trade_capital

            # --- 4. REGISTRO DE CAPITAL DIÁRIO ---
            # O CAIXA OCIOSO RENDE CDI (com haircut) sobre o saldo disponível.
            # Isso é feito APÓS as operações do dia para não antecipar o rendimento.
            cdi_do_dia = ibov_row.get('cdi_retorno', 0.0)
            if pd.isna(cdi_do_dia):
                cdi_do_dia = 0.0
            capital *= (1 + cdi_do_dia)

            total_positions_value = sum(
                pos['qty'] * current_b3_prices[t]
                for t, pos in positions.items()
                if not pd.isna(current_b3_prices[t])
            )
            current_equity = capital + total_positions_value

            equity_dates.append(date)
            equity_curve.append(current_equity)

        self.equity_df = pd.DataFrame({'strategy_equity': equity_curve}, index=equity_dates)

# =====================================================================
# SEÇÃO 6: AVALIAÇÃO E BENCHMARK
# =====================================================================

    def evaluate_performance(self):
        print("Calculando métricas de performance e gerando o gráfico...")

        perf_df = self.ibov.join(self.equity_df)
        perf_df = perf_df.dropna(subset=['strategy_equity']).copy()

        perf_df['strategy_returns'] = perf_df['strategy_equity'].pct_change().fillna(0)
        perf_df['cum_strategy'] = (1 + perf_df['strategy_returns']).cumprod()
        perf_df['cum_benchmark'] = (1 + perf_df['returns'].fillna(0)).cumprod()

        # --- PLOTAGEM ---
        plt.figure(figsize=(14, 7))

        plt.plot(perf_df.index, perf_df['cum_strategy'],
                 label='Estratégia Quant (HMM Multi-Ativos)', color='#1f77b4', linewidth=2)
        plt.plot(perf_df.index, perf_df['cum_benchmark'],
                 label='Benchmark (IBOV)', color='gray', alpha=0.7, linewidth=1.5)

        split_date_pd = pd.to_datetime(self.split_date)

        if split_date_pd > perf_df.index.min() and split_date_pd < perf_df.index.max():
            plt.axvline(x=split_date_pd, color='red', linestyle='--', linewidth=2,
                        label='Divisão Treino/Teste')
            ax = plt.gca()
            plt.text(split_date_pd - pd.Timedelta(days=20), 0.95, 'In-Sample\n(Treino)',
                     color='red', horizontalalignment='right',
                     transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
            plt.text(split_date_pd + pd.Timedelta(days=20), 0.95, 'Out-of-Sample\n(Teste)',
                     color='green', horizontalalignment='left',
                     transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')

        plt.title('Performance da Estratégia Quantitativa vs IBOV', fontsize=14, pad=15)
        plt.ylabel('Retorno Acumulado (1.0 = Capital Inicial)', fontsize=12)
        plt.xlabel('Data', fontsize=12)
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

        print("\n" + "="*70)
        print(" RESULTADOS DO BACKTEST (ESTRATÉGIA HMM MULTI-ATIVOS) ".center(70, "="))
        print("="*70 + "\n")

        perf_df['strategy_returns'] = perf_df['strategy_equity'].pct_change().fillna(0)
        perf_df['benchmark_returns'] = perf_df['returns'].fillna(0)

        split_date_pd = pd.to_datetime(self.split_date)
        in_sample  = perf_df[perf_df.index < split_date_pd]
        out_sample = perf_df[perf_df.index >= split_date_pd]

        def calcular_e_imprimir_metricas(df_periodo, nome_periodo):
            if df_periodo.empty:
                print(f"--- Sem dados suficientes para o período: {nome_periodo} ---\n")
                return

            strat_ret = df_periodo['strategy_returns']
            bench_ret = df_periodo['benchmark_returns']

            cum_strat = (1 + strat_ret).cumprod()
            cum_bench = (1 + bench_ret).cumprod()

            tot_strat  = (cum_strat.iloc[-1] - 1) * 100
            tot_bench  = (cum_bench.iloc[-1] - 1) * 100

            anos = len(df_periodo) / 252
            cagr_strat = ((cum_strat.iloc[-1]) ** (1 / anos) - 1) * 100 if anos > 0 else 0
            cagr_bench = ((cum_bench.iloc[-1]) ** (1 / anos) - 1) * 100 if anos > 0 else 0

            vol_strat  = strat_ret.std() * np.sqrt(252) * 100
            vol_bench  = bench_ret.std() * np.sqrt(252) * 100

            sharpe_strat = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() != 0 else 0
            sharpe_bench = (bench_ret.mean() / bench_ret.std()) * np.sqrt(252) if bench_ret.std() != 0 else 0

            dd_strat = (cum_strat / cum_strat.cummax() - 1).min() * 100
            dd_bench = (cum_bench / cum_bench.cummax() - 1).min() * 100

            data_inicio = df_periodo.index.min().strftime('%Y-%m-%d')
            data_fim    = df_periodo.index.max().strftime('%Y-%m-%d')

            print(f"[{nome_periodo}] - Período: {data_inicio} a {data_fim}")
            print("-" * 70)
            print(f"{'Métrica':<25} | {'Estratégia Quant':<20} | {'Benchmark (IBOV)':<20}")
            print("-" * 70)
            print(f"{'Retorno Total':<25} | {tot_strat:>19.2f}% | {tot_bench:>19.2f}%")
            print(f"{'CAGR (Retorno Anual)':<25} | {cagr_strat:>19.2f}% | {cagr_bench:>19.2f}%")
            print(f"{'Volatilidade Anual':<25} | {vol_strat:>19.2f}% | {vol_bench:>19.2f}%")
            print(f"{'Índice Sharpe':<25} | {sharpe_strat:>19.2f}  | {sharpe_bench:>19.2f} ")
            print(f"{'Max Drawdown':<25} | {dd_strat:>19.2f}% | {dd_bench:>19.2f}%")
            print("-" * 70 + "\n")

        calcular_e_imprimir_metricas(perf_df,    "PERFORMANCE TOTAL")
        calcular_e_imprimir_metricas(in_sample,  "IN-SAMPLE (TREINAMENTO)")
        calcular_e_imprimir_metricas(out_sample, "OUT-OF-SAMPLE (TESTE)")


# =====================================================================
# EXECUÇÃO DO PIPELINE
# =====================================================================
if __name__ == "__main__":

    pipeline = QuantStrategyPipeline(
        start_date='2010-01-04',
        end_date='2025-12-03',
        split_date='2019-01-01'
    )

    b3_net = pipeline.net_analysis(df=pipeline.b3_returns, window=21)

    pipeline.ibov['density']        = b3_net['density']
    pipeline.ibov['max_eigen_value'] = b3_net['max_eigen_value']
    pipeline.calculate_features(window=21)

    pipeline.fit_hmm()

    pipeline.run_backtest(
        initial_capital=100000,
        stop_loss_pct=STOP_LOSS_PCT,
        trailing_stop_pct=TRAILING_STOP_PCT
    )

    pipeline.evaluate_performance()