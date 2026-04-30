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
THRESHOLD_net_analysis = 0.7      #filtro de correlação mínima para matriz de adjacencia

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

        #self.b3.plot(title='b3')
        #self.ibov.plot(title='ibov')

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

        #self.ibov.plot()
        #self.b3.plot

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

        #df.plot()

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

        self.plot_gmm_hmm_states(model, train_data)
        self.plot_gmm_hmm_3d_animated(model, train_data)

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
        #self.ibov[['hmm_state', 'hmm_entropy']].plot()

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

        # ── VoV: limiar P80 rolling 252 dias (sem viés — mesmo padrão da density) ──
        # O percentil é calculado com dados até o dia t, nunca à frente.
        # min_periods=63 evita leituras ruidosas no início do histórico.
        self.ibov['vov_rolling_80'] = (
            self.ibov['vol_of_vol']
            .rolling(window=252, min_periods=63)
            .quantile(0.8)
        )

        self.ibov['eigen_rolling_80'] = (
            self.ibov['max_eigen_value']
            .rolling(window=252, min_periods=63)
            .quantile(0.8)
        )

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

            # ── Sinal VoV: reduz exposição quando volatilidade está sendo
            # anormalmente volátil (acima do P80 histórico rolling 252d).
            # Indica regime de transição ou fragilidade estrutural —
            # mesmo limiar sem viés usado na density e na entropia.
            vov_threshold = ibov_row.get('vov_rolling_80')
            if pd.notna(ibov_row.get('vol_of_vol')) and pd.notna(vov_threshold):
                if ibov_row['vol_of_vol'] > vov_threshold:
                    exposure_modifier *= 0.5

            eigen_threshold = ibov_row.get('eigen_rolling_80')
            if pd.notna(ibov_row.get('max_eigen_value')) and pd.notna(eigen_threshold):
                if ibov_row['max_eigen_value'] > eigen_threshold:
                    exposure_modifier *= 0.8


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

    def plot_gmm_hmm_states(self, model, train_df):
        """
        Visualização rica das distribuições GMM-HMM por estado.

        Layout: 4 painéis
          [0,0] Scatter principal — todos os estados sobrepostos com
                elipses de confiança a 1σ / 2σ por mistura
          [0,1] KDE 2-D por estado (densidade real da distribuição)
          [1,0] Distribuição marginal de Returns por estado
          [1,1] Distribuição marginal de Volatility por estado
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Ellipse
        from scipy.stats import gaussian_kde
        import numpy as np

        # ── Dados limpos ──────────────────────────────────────────────────
        if hasattr(train_df, 'dropna'):
            clean = train_df[['returns', 'volatility']].dropna()
            X     = clean.values
        else:
            X = train_df

        states = model.predict(X)

        # Mapeamento estado → nível de vol (para nomear os regimes)
        state_vols = {i: X[states == i, 1].mean() for i in range(model.n_components)}
        sorted_s   = sorted(state_vols, key=state_vols.get)
        nomes      = {sorted_s[0]: "Momentum (baixa vol)",
                      sorted_s[1]: "Neutro (média vol)",
                      sorted_s[2]: "Hedge (alta vol)"}
        CORES      = {sorted_s[0]: "#2ECC71",   # verde
                      sorted_s[1]: "#F39C12",   # laranja
                      sorted_s[2]: "#E74C3C"}   # vermelho

        # ── Helper: elipse de confiança para 1σ e 2σ ─────────────────────
        def _ellipse(mean, cov, n_std, color, lw, ls, alpha_fill):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h   = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
            el_fill = Ellipse(xy=mean, width=w, height=h, angle=angle,
                              facecolor=color, alpha=alpha_fill, linewidth=0)
            el_edge = Ellipse(xy=mean, width=w, height=h, angle=angle,
                              facecolor='none', edgecolor=color, lw=lw,
                              linestyle=ls, alpha=0.95)
            return el_fill, el_edge

        # ── Layout ────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 13))
        fig.patch.set_facecolor("#F0F2F5")
        gs  = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28)

        ax_sc  = fig.add_subplot(gs[0, 0])   # scatter + elipses
        ax_kde = fig.add_subplot(gs[0, 1])   # KDE 2-D
        ax_ret = fig.add_subplot(gs[1, 0])   # marginal returns
        ax_vol = fig.add_subplot(gs[1, 1])   # marginal volatility

        for ax in [ax_sc, ax_kde, ax_ret, ax_vol]:
            ax.set_facecolor("#FAFBFC")

        # ══════════════════════════════════════════════════════════════════
        # PAINEL 0 — Scatter + elipses de confiança por mistura
        # ══════════════════════════════════════════════════════════════════
        for s in range(model.n_components):
            cor   = CORES[s]
            nome  = nomes[s]
            mask  = states == s
            xs, ys = X[mask, 0], X[mask, 1]
            n_pts  = mask.sum()

            # Pontos com transparência proporcional à densidade local
            ax_sc.scatter(xs, ys, s=12, alpha=0.30, color=cor,
                          rasterized=True, zorder=2)

            # Centróide do estado (média ponderada das misturas)
            weights = model.weights_[s]
            centroid = (weights[:, None] * model.means_[s]).sum(axis=0)
            ax_sc.scatter(*centroid, s=120, color=cor, marker='*',
                          edgecolors='white', linewidths=0.8, zorder=5)

            # Elipses por componente de mistura
            for m in range(model.n_mix):
                mean = model.means_[s][m]
                cov  = model.covars_[s][m]
                w    = model.weights_[s][m]

                # 1σ — preenchimento suave
                ef1, ee1 = _ellipse(mean, cov, 1.0, cor, lw=2.0,
                                    ls='-', alpha_fill=0.08 * w * model.n_mix)
                # 2σ — só borda tracejada
                _,   ee2 = _ellipse(mean, cov, 2.0, cor, lw=1.2,
                                    ls='--', alpha_fill=0.0)

                for patch in [ef1, ee1, ee2]:
                    ax_sc.add_patch(patch)

                # Marca o centro de cada mistura com ×
                ax_sc.plot(*mean, marker='x', ms=8, mew=2,
                           color=cor, zorder=4, alpha=0.9)

            # Label manual com contagem
            ax_sc.scatter([], [], s=40, color=cor, alpha=0.7,
                          label=f"{nome}  (n={n_pts})")

        ax_sc.set_xlabel("Returns (diário)", fontsize=10)
        ax_sc.set_ylabel("Volatilidade (anualizada)", fontsize=10)
        ax_sc.set_title("Scatter + Elipses GMM por Estado\n"
                        "★ = centróide do estado  ✕ = centro da mistura  "
                        "—1σ  - - 2σ",
                        fontsize=10, pad=8)
        ax_sc.legend(fontsize=9, framealpha=0.92)
        ax_sc.grid(True, linestyle=':', alpha=0.5)

        # ══════════════════════════════════════════════════════════════════
        # PAINEL 1 — KDE 2-D por estado (densidade real)
        # ══════════════════════════════════════════════════════════════════
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:120j, y_min:y_max:120j]
        grid   = np.vstack([xx.ravel(), yy.ravel()])

        for s in range(model.n_components):
            cor  = CORES[s]
            mask = states == s
            pts  = X[mask].T
            if pts.shape[1] < 4:
                continue
            kde  = gaussian_kde(pts, bw_method='scott')
            zz   = kde(grid).reshape(xx.shape)
            zz   = zz / zz.max()   # normaliza 0-1

            # ── CORREÇÃO: cmap por cor em vez de arrays colors/alpha ──────
            # contourf com colors=lista e alpha=lista falha quando
            # len(colors) != len(levels) - 1.  A solução é criar um
            # LinearSegmentedColormap transparente → cor sólida.
            import matplotlib.colors as mcolors
            r, g, b = mcolors.to_rgb(cor)
            cmap_state = mcolors.LinearSegmentedColormap.from_list(
                f'cmap_{s}',
                [(r, g, b, 0.0),   # alpha=0 em zz=0
                 (r, g, b, 0.38)], # alpha=0.38 em zz=1
                N=256
            )
            # ─────────────────────────────────────────────────────────────

            LEVELS = np.linspace(0.05, 1.0, 9)   # 9 níveis → 8 bandas OK
            ax_kde.contourf(xx, yy, zz, levels=LEVELS, cmap=cmap_state)
            ax_kde.contour(xx, yy, zz,
                           levels=[0.15, 0.50, 0.85],
                           colors=[cor], linewidths=[0.8, 1.4, 2.0],
                           alpha=0.85)

        ax_kde.set_xlabel("Returns (diário)", fontsize=10)
        ax_kde.set_ylabel("Volatilidade (anualizada)", fontsize=10)
        ax_kde.set_title("Densidade KDE 2-D por Estado\n"
                         "contornos em 15% / 50% / 85% da densidade máxima",
                         fontsize=10, pad=8)
        # Mini legenda de cores
        for s in range(model.n_components):
            ax_kde.scatter([], [], color=CORES[s], s=40, label=nomes[s])
        ax_kde.legend(fontsize=9, framealpha=0.92)
        ax_kde.grid(True, linestyle=':', alpha=0.5)

        # ══════════════════════════════════════════════════════════════════
        # PAINEL 2 — Marginal: Returns
        # ══════════════════════════════════════════════════════════════════
        r_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)
        for s in range(model.n_components):
            cor  = CORES[s]
            mask = states == s
            kde  = gaussian_kde(X[mask, 0], bw_method='scott')
            dens = kde(r_range)
            ax_ret.fill_between(r_range, dens, alpha=0.20, color=cor)
            ax_ret.plot(r_range, dens, color=cor, lw=2.0, label=nomes[s])
            # Média e desvio-padrão empírico
            mu  = X[mask, 0].mean()
            sig = X[mask, 0].std()
            ax_ret.axvline(mu, color=cor, lw=1.2, linestyle='--', alpha=0.7)
            ax_ret.text(mu, dens.max() * 0.92,
                        f"μ={mu:.4f}\nσ={sig:.4f}",
                        fontsize=7.5, color=cor, ha='center', va='top',
                        bbox=dict(fc='white', ec=cor, alpha=0.75, pad=2, lw=0.8))

        ax_ret.axvline(0, color='#888', lw=0.8, linestyle=':')
        ax_ret.set_xlabel("Returns (diário)", fontsize=10)
        ax_ret.set_ylabel("Densidade", fontsize=10)
        ax_ret.set_title("Distribuição Marginal — Returns\n-- = média do estado",
                         fontsize=10, pad=8)
        ax_ret.legend(fontsize=9, framealpha=0.92)
        ax_ret.grid(True, linestyle=':', alpha=0.5)

        # ══════════════════════════════════════════════════════════════════
        # PAINEL 3 — Marginal: Volatilidade
        # ══════════════════════════════════════════════════════════════════
        v_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 300)
        for s in range(model.n_components):
            cor  = CORES[s]
            mask = states == s
            kde  = gaussian_kde(X[mask, 1], bw_method='scott')
            dens = kde(v_range)
            ax_vol.fill_between(v_range, dens, alpha=0.20, color=cor)
            ax_vol.plot(v_range, dens, color=cor, lw=2.0, label=nomes[s])
            mu  = X[mask, 1].mean()
            sig = X[mask, 1].std()
            ax_vol.axvline(mu, color=cor, lw=1.2, linestyle='--', alpha=0.7)
            ax_vol.text(mu, dens.max() * 0.92,
                        f"μ={mu:.3f}\nσ={sig:.3f}",
                        fontsize=7.5, color=cor, ha='center', va='top',
                        bbox=dict(fc='white', ec=cor, alpha=0.75, pad=2, lw=0.8))

        ax_vol.set_xlabel("Volatilidade (anualizada)", fontsize=10)
        ax_vol.set_ylabel("Densidade", fontsize=10)
        ax_vol.set_title("Distribuição Marginal — Volatilidade\n-- = média do estado",
                         fontsize=10, pad=8)
        ax_vol.legend(fontsize=9, framealpha=0.92)
        ax_vol.grid(True, linestyle=':', alpha=0.5)

        fig.suptitle("Distribuições GMM-HMM por Estado — Treino In-Sample",
                     fontsize=14, fontweight='bold', y=1.01)
        plt.savefig("fig_gmm_hmm_states.png", dpi=180,
                    bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.show()

    def plot_gmm_hmm_3d_animated(self, model, train_df):
        """
        Superfície 3-D interativa com Plotly (WebGL).
        Gira suave no browser — sem travar.
        """
        import numpy as np
        import plotly.graph_objects as go
        from scipy.stats import multivariate_normal

        # ── Dados limpos ──────────────────────────────────────────────────
        if hasattr(train_df, 'dropna'):
            X = train_df[['returns', 'volatility']].dropna().values
        else:
            X = train_df

        states     = model.predict(X)
        state_vols = {i: X[states == i, 1].mean() for i in range(model.n_components)}
        sorted_s   = sorted(state_vols, key=state_vols.get)
        nomes      = {sorted_s[0]: "Momentum (baixa vol)",
                    sorted_s[1]: "Neutro (média vol)",
                    sorted_s[2]: "Hedge (alta vol)"}
        CORES      = {sorted_s[0]: "#2ECC71",
                    sorted_s[1]: "#F39C12",
                    sorted_s[2]: "#E74C3C"}

        # ── Grid ─────────────────────────────────────────────────────────
        GRID_N   = 80   # pode subir à vontade — Plotly aguenta bem
        x_range  = np.linspace(X[:, 0].min(), X[:, 0].max(), GRID_N)
        y_range  = np.linspace(X[:, 1].min(), X[:, 1].max(), GRID_N)
        XX, YY   = np.meshgrid(x_range, y_range)
        grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

        def gmm_density(s):
            dens = np.zeros(len(grid_pts))
            for m in range(model.n_mix):
                dens += (model.weights_[s][m] *
                        multivariate_normal.pdf(
                            grid_pts,
                            mean=model.means_[s][m],
                            cov=model.covars_[s][m]))
            return dens.reshape(GRID_N, GRID_N)

        prior = {s: (states == s).mean() for s in range(model.n_components)}
        ZZ    = {s: gmm_density(s) * prior[s] for s in range(model.n_components)}
        z_max = max(z.max() for z in ZZ.values())
        ZZ    = {s: z / z_max for s, z in ZZ.items()}

        # ── Constrói as superfícies Plotly ────────────────────────────────
        traces = []
        for s in sorted_s:
            cor  = CORES[s]
            nome = nomes[s]

            # Colorscale monocromático transparente → cor sólida
            r = int(cor[1:3], 16)
            g = int(cor[3:5], 16)
            b = int(cor[5:7], 16)
            colorscale = [
                [0.0, f'rgba({r},{g},{b},0.0)'],
                [1.0, f'rgba({r},{g},{b},0.92)'],
            ]

            traces.append(go.Surface(
                x=XX, y=YY, z=ZZ[s],
                colorscale=colorscale,
                showscale=False,
                name=nome,
                opacity=0.88,
                contours=dict(
                    z=dict(show=True, usecolormap=True,
                        highlightcolor='white', project_z=True)
                ),
                hovertemplate=(
                    "Return: %{x:.4f}<br>"
                    "Vol: %{y:.3f}<br>"
                    "Densidade: %{z:.4f}<extra>" + nome + "</extra>"
                )
            ))

            # Marcadores nos centros das misturas
            mx_list, my_list, mz_list = [], [], []
            for m in range(model.n_mix):
                mx, my = model.means_[s][m]
                ix = np.argmin(np.abs(x_range - mx))
                iy = np.argmin(np.abs(y_range - my))
                mx_list.append(mx)
                my_list.append(my)
                mz_list.append(ZZ[s][iy, ix])

            traces.append(go.Scatter3d(
                x=mx_list, y=my_list, z=mz_list,
                mode='markers',
                marker=dict(size=6, color='white',
                            line=dict(color=cor, width=3)),
                name=f'{nome} — centro mistura',
                showlegend=False,
                hovertemplate="Centro mistura<br>Return: %{x:.4f}<br>Vol: %{y:.3f}<extra></extra>"
            ))

        # ── Layout ────────────────────────────────────────────────────────
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text="GMM-HMM — Retorno × Volatilidade × Probabilidade",
                font=dict(size=14, color='white'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(title='Returns', backgroundcolor='#1C1C2E',
                        gridcolor='#444466', color='white'),
                yaxis=dict(title='Volatilidade', backgroundcolor='#1C1C2E',
                        gridcolor='#444466', color='white'),
                zaxis=dict(title='Densidade (norm.)', backgroundcolor='#1C1C2E',
                        gridcolor='#444466', color='white'),
                bgcolor='#1C1C2E',
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))
            ),
            paper_bgcolor='#141519',
            plot_bgcolor='#141519',
            legend=dict(font=dict(color='white'), bgcolor='rgba(44,44,62,0.6)'),
            margin=dict(l=0, r=0, t=50, b=0),
            height=700
        )

        fig.show()   # abre no browser — gira smooth com WebGL
        fig.write_html("fig_gmm_hmm_3d.html")
        print("[3D] Salvo: fig_gmm_hmm_3d.html")
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