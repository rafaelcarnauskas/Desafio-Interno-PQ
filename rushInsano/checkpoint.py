from hmmlearn.hmm import GaussianHMM, GMMHMM
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

STOP_LOSS_PCT            = 0.07   # drawdown máximo desde o pico (7 %)
TRAILING_STOP_PCT        = 0.05   # recuo do equity desde o último pico (5 %)
THRESHOLD_net_analysis = 0.7      #usar threshold fixo e nao percentil torna a medida de densidade util

class QuantStrategyPipeline:
    def __init__(self, start_date, end_date, split_date='2019-01-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.split_date = split_date # Data que divide o In-Sample do Out-of-Sample

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
        # Calculando as médias e RSI para todas as ações da B3 de uma vez usando operações vetoriais
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

        # CORREÇÃO: self.b3 é DataFrame de preços e não tem coluna 'returns'.
        # O fallback correto é self.b3_returns.
        if df is None:
            df = self.b3_returns


        '''Considera-se uma aresta da rede como uma correlação que passa o THRESHOLD
            A densidade da rede é calculada com número de arestas/máximo de arestas possíveis
            A quantidade máxima de arestas é o coeficiente de newton com n e 2
            A matriz de correlação é calculada a partir dos log-retornos'''

        # O método net_analysis() vai devolver o autovalor máximo e seu autovetor associado para que outro método use essas informações
        
        # Listas auxiliares para otimização de performance (evita lentidão de preencher o df linha a linha)
        densities = [np.nan] * window
        max_eigen_values = [np.nan] * window
        
        # Caso queira armazenar as matrizes e autovetores no futuro, as listas estão preparadas:
        # list_A = [np.nan] * window
        # list_norm_eigen_vectors = [np.nan] * window
        # list_edges = [np.nan] * window

        columns = df.columns.tolist()
        n = len(columns)
        possible_edges = comb(n, 2, exact=True)

        # CORREÇÃO: self.end_date e self.start_date são strings; subtrair strings gera TypeError.
        # Além disso, o loop original usava delta.days (dias corridos), mas df só contém
        # dias úteis — iloc ficaria fora dos limites. Iteramos agora sobre as posições reais do df.
        for i in range(window, len(df)):
            corr = df.iloc[i-window:i].corr().values

            #proximo passo é analisar autovalor (eigenvector e eigenvalue) e degree centrality com eigenvector centrality
            #basicamente analisar se tem algum ativo muito conectado com muita gente

            """A análise do autovalor da matriz de correlação consiste em medir a centralidade da rede
            se o maior autovalor da matriz A crescer muito e de forma rápida quer dizer que um autovetor 
            está ficando concentrado em uma direção e isso indica fragilidade sistêmica"""

            A = np.zeros((n, n))
            edges = []

            # Loop para construir a matriz de adjacência
            for r in range(n):
                for c in range(n):
                    if r == c:
                        A[r, c] = 0
                    else:
                        A[r, c] = corr[r, c] if abs(corr[r, c]) >= THRESHOLD_net_analysis else 0
                        
                        """Da forma que está, o autovetor medirá intensidade das conexões
                            Se for preciso medir quantidade de conexões, faz A[i,j]=1"""
                        
                        if c < r and A[r, c]:
                            edges.append((columns[r], columns[c]))

            density = len(edges) / possible_edges if possible_edges > 0 else 0 
            densities.append(density)

            # Por enquanto não vejo sentido em implementar medida de diâmetro da matriz de adjacência

            """Quando o maior autovalor aumenta:
                -Densidade tende a aumentar
                -Diâmetro tende a diminuir (rede mais compacta)
                -Eigenvector centrality fica mais concentrada"""

            # Seção para o cálculo do autovetor e autovalor
            # A posição i do autovetor associado ao maior autovalor vai indicar o grau de conectividade do ativo i
            # Ao final é interessante normalizar o autovetor para ter uma medida de 0 a 1

            # np.linalg.eigh é específico e mais rápido para matrizes simétricas (como correlação)
            eigen_values, eigen_vector = np.linalg.eigh(A)

            max_eigen_value = eigen_values[-1] # Captura o maior autovalor
            max_eigen_values.append(max_eigen_value)
            
            # max_eigen_vector = eigen_vector[:, -1]
            # norm_eigen_vector = (max_eigen_vector - max_eigen_vector.min()) / (max_eigen_vector.max() - max_eigen_vector.min()) if max_eigen_vector.max() != max_eigen_vector.min() else max_eigen_vector

        # CORREÇÃO: densities e max_eigen_values já têm exatamente len(df) elementos
        # (window NaNs iniciais + len(df)-window valores calculados), então não é necessário
        # padding adicional — o slice extra causava ValueError por tamanhos incompatíveis.
        df['density'] = densities
        df['max_eigen_value'] = max_eigen_values

        df.plot()

        return df    

    def _calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

# =====================================================================
# SEÇÃO 3: MODELAGEM DE REGIMES (HMM)
# =====================================================================

    def fit_hmm(self):
        # Filtra os dados apenas até a data de split para não vazar informações (Data Leakage)
        train_data = self.ibov.loc[self.start_date:self.split_date].dropna(subset=['returns', 'volatility'])
        X_train = train_data[['returns', 'volatility']].values
        
        # 1. Usarei o gmmhmm para criar uma curva normal para cada estado, de forma a preservar as fat tails dos retornos extremos, coisa que não acontece com uma distribuição normal
        model = GMMHMM(n_components=3, n_mix=2, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X_train)
        
        # 2. Isolar dados de Teste (Out-of-Sample) 
        # (pega os dados estritamente após a data de corte)
        test_data = self.ibov.loc[self.split_date:].iloc[1:].dropna(subset=['returns', 'volatility'])
        X_test = test_data[['returns', 'volatility']].values
        
        # 3. Prever separadamente para garantir isolamento
        # Previsão Treino
        states_train = model.predict(X_train)
        probs_train = model.predict_proba(X_train)
        
        # Previsão Teste (O modelo usa as regras do passado para classificar os dados novos)
        states_test = model.predict(X_test)
        probs_test = model.predict_proba(X_test)
        
        # 4. Concatenar os resultados de volta para o DataFrame inteiro
        all_states = np.concatenate([states_train, states_test])
        all_probs = np.vstack([probs_train, probs_test])
        
        # Cria um DataFrame temporário alinhado para jogar de volta no df principal
        valid_data = pd.concat([train_data, test_data]).copy()
        valid_data['hmm_state_raw'] = all_states
        valid_data['hmm_entropy'] = [scipy_entropy(p, base=2) for p in all_probs]
        
        # 5. Mapear os estados (0=Baixa, 1=Média, 2=Alta vol)
        # IMPORTANTE: O mapeamento do que é "alta" ou "baixa" vol DEVE ser aprendido APENAS no treino!
        state_vols_train = {i: train_data.iloc[states_train == i]['volatility'].mean() for i in range(3)}
        sorted_states = sorted(state_vols_train, key=state_vols_train.get)
        state_map = {sorted_states[0]: 0, sorted_states[1]: 1, sorted_states[2]: 2}
        
        # Aplica o mapa (aprendido no treino) na base toda
        valid_data['hmm_state'] = valid_data['hmm_state_raw'].map(state_map)
        
        # Devolve as colunas para o IBOV
        self.ibov = self.ibov.join(valid_data[['hmm_state', 'hmm_entropy']])

        self.ibov[['hmm_state', 'hmm_entropy']].plot()

# =====================================================================
# SEÇÃO 4 E 5: BACKTESTER (GERENCIAMENTO DE RISCO E ESTRATÉGIAS)
# =====================================================================

    def run_backtest(self, initial_capital=100000, stop_loss_pct=0.07, trailing_stop_pct=0.05):
        print("Rodando Backtest Multi-Ativos...")
        
        capital = initial_capital
        
        # Dicionário para controlar as posições de cada ação individualmente
        tickers = self.b3.columns
        positions = {ticker: {'qty': 0, 'entry_price': 0, 'highest_price': 0} for ticker in tickers}
        
        equity_curve = []
        equity_dates = []
        
        # Itera dia a dia
        for date, ibov_row in self.ibov.iterrows():
            # Pula os dias que não temos os indicadores mapeados
            if pd.isna(ibov_row['hmm_state']) or date not in self.b3.index:
                continue
                
            current_b3_prices = self.b3.loc[date]
            
            # --- 1. MODULADOR DE EXPOSIÇÃO DO MERCADO ---
            exposure_modifier = 1.0
            if ibov_row['hmm_entropy'] > 1.2: exposure_modifier *= 0.5
            
            # Checa se vol_of_vol e density existem antes de checar quantil (evita erros nos primeiros dias)
            # CORREÇÃO (comentário atualizado): 'density' e 'max_eigen_value' agora estão
            # disponíveis em ibov_row após o join feito no __main__.
            if pd.notna(ibov_row.get('density')) and ibov_row['density'] > self.ibov['density'].quantile(0.8):
                exposure_modifier *= 0.5
                
            # Estado do mercado de hoje
            regime = ibov_row['hmm_state']
            
            # Ações que deram sinal de compra hoje
            buy_signals = []
            
            # --- 2. LOOP POR CADA AÇÃO PARA CHECAR SINAIS E RISCO ---
            for ticker in tickers:
                price = current_b3_prices[ticker]
                if pd.isna(price) or price <= 0: continue
                
                pos = positions[ticker]
                
                # Checagem de Stop-Loss e Trailing Stop
                if pos['qty'] > 0:
                    pos['highest_price'] = max(pos['highest_price'], price)
                    
                    if (price < pos['entry_price'] * (1 - stop_loss_pct)) or \
                       (price < pos['highest_price'] * (1 - trailing_stop_pct)) or \
                       (regime == 2): # Saída forçada no Estado 2 (Alta Volatilidade/Crash)
                       
                        # Vende tudo
                        capital += pos['qty'] * price
                        positions[ticker] = {'qty': 0, 'entry_price': 0, 'highest_price': 0}
                        continue # Pula para o próximo ticker
                
                # --- Lógica de Entrada Individual por Ação ---
                if pos['qty'] == 0:
                    # Estado 0: Baixa Vol (Tendência)
                    if regime == 0:
                        if self.b3_sma_fast.loc[date, ticker] > self.b3_sma_slow.loc[date, ticker]:
                            buy_signals.append(ticker)
                            
                    # Estado 1: Média Vol (Reversão)
                    elif regime == 1:
                        if self.b3_rsi.loc[date, ticker] < 30:
                            buy_signals.append(ticker)
                else:
                    # Lógica de saída antecipada no Regime 1 se sobrecomprado
                    if regime == 1 and self.b3_rsi.loc[date, ticker] > 70:
                        capital += pos['qty'] * price
                        positions[ticker] = {'qty': 0, 'entry_price': 0, 'highest_price': 0}

            # --- 3. EXECUÇÃO DE COMPRAS ---
            # Se houveram sinais, divide o capital alocável entre eles
            if buy_signals:
                # Usa no máximo o modifier do capital, e no máximo 10% do capital total por trade para diversificar
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
            # Soma o caixa (capital) com o valor de todas as posições abertas
            total_positions_value = sum(pos['qty'] * current_b3_prices[t] for t, pos in positions.items() if not pd.isna(current_b3_prices[t]))
            current_equity = capital + total_positions_value
            
            equity_dates.append(date)
            equity_curve.append(current_equity)

        # Salva o resultado no df do IBOV (pois é a nossa base temporal mestre)
        self.equity_df = pd.DataFrame({'strategy_equity': equity_curve}, index=equity_dates)

# =====================================================================
# SEÇÃO 6: AVALIAÇÃO E BENCHMARK
# =====================================================================

    def evaluate_performance(self):
        print("Calculando métricas de performance e gerando o gráfico...")
        
        perf_df = self.ibov.join(self.equity_df)
        perf_df = perf_df.dropna(subset=['strategy_equity']).copy()
        
        # 3. Calcula os retornos diários da estratégia
        perf_df['strategy_returns'] = perf_df['strategy_equity'].pct_change().fillna(0)
        
        # 4. Calcula os retornos acumulados de ambos partindo do MESMO ponto (base 1.0)
        # Se usarmos o 'returns' inteiro do IBOV, ele acumularia quedas do período de aquecimento
        perf_df['cum_strategy'] = (1 + perf_df['strategy_returns']).cumprod()
        perf_df['cum_benchmark'] = (1 + perf_df['returns'].fillna(0)).cumprod()
        
        # --- PLOTAGEM ---
        plt.figure(figsize=(14, 7))
        
        # Curvas principais
        plt.plot(perf_df.index, perf_df['cum_strategy'], label='Estratégia Quant (HMM Multi-Ativos)', color='#1f77b4', linewidth=2)
        plt.plot(perf_df.index, perf_df['cum_benchmark'], label='Benchmark (IBOV)', color='gray', alpha=0.7, linewidth=1.5)
        
        # Linha de divisão Treino / Teste
        split_date_pd = pd.to_datetime(self.split_date)
        
        # Checa se a data de split cai dentro do período em que operamos
        if split_date_pd > perf_df.index.min() and split_date_pd < perf_df.index.max():
            plt.axvline(x=split_date_pd, color='red', linestyle='--', linewidth=2, label='Divisão Treino/Teste')
            
            # Textos explicativos no topo do gráfico
            ax = plt.gca()
            plt.text(split_date_pd - pd.Timedelta(days=20), 0.95, 'In-Sample\n(Treino)', 
                     color='red', horizontalalignment='right', transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
            plt.text(split_date_pd + pd.Timedelta(days=20), 0.95, 'Out-of-Sample\n(Teste)', 
                     color='green', horizontalalignment='left', transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')

        # Estética do gráfico
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
        
        
        
        # 3. Calcula os retornos diários
        perf_df['strategy_returns'] = perf_df['strategy_equity'].pct_change().fillna(0)
        perf_df['benchmark_returns'] = perf_df['returns'].fillna(0)
        
        # 4. Define a data de corte para In-Sample e Out-of-Sample
        split_date_pd = pd.to_datetime(self.split_date)
        in_sample = perf_df[perf_df.index < split_date_pd]
        out_sample = perf_df[perf_df.index >= split_date_pd]

        def calcular_e_imprimir_metricas(df_periodo, nome_periodo):
            if df_periodo.empty:
                print(f"--- Sem dados suficientes para o período: {nome_periodo} ---\n")
                return

            strat_ret = df_periodo['strategy_returns']
            bench_ret = df_periodo['benchmark_returns']
            
            # Recalcula o acumulado do zero para o período específico
            cum_strat = (1 + strat_ret).cumprod()
            cum_bench = (1 + bench_ret).cumprod()
            
            # 1. Retorno Total (%)
            tot_strat = (cum_strat.iloc[-1] - 1) * 100
            tot_bench = (cum_bench.iloc[-1] - 1) * 100
            
            # 2. CAGR (Retorno Anualizado)
            anos = len(df_periodo) / 252
            cagr_strat = ((cum_strat.iloc[-1]) ** (1 / anos) - 1) * 100 if anos > 0 else 0
            cagr_bench = ((cum_bench.iloc[-1]) ** (1 / anos) - 1) * 100 if anos > 0 else 0
            
            # 3. Volatilidade Anualizada
            vol_strat = strat_ret.std() * np.sqrt(252) * 100
            vol_bench = bench_ret.std() * np.sqrt(252) * 100
            
            # 4. Índice Sharpe (assumindo Risk-Free = 0 para simplificação)
            sharpe_strat = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() != 0 else 0
            sharpe_bench = (bench_ret.mean() / bench_ret.std()) * np.sqrt(252) if bench_ret.std() != 0 else 0
            
            # 5. Maximum Drawdown
            dd_strat = (cum_strat / cum_strat.cummax() - 1).min() * 100
            dd_bench = (cum_bench / cum_bench.cummax() - 1).min() * 100

            # Formatação da Tabela
            data_inicio = df_periodo.index.min().strftime('%Y-%m-%d')
            data_fim = df_periodo.index.max().strftime('%Y-%m-%d')
            
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

        # Roda a avaliação para os 3 cenários
        calcular_e_imprimir_metricas(perf_df, "PERFORMANCE TOTAL")
        calcular_e_imprimir_metricas(in_sample, "IN-SAMPLE (TREINAMENTO)")
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
    
    # Direct assignment bypasses the MultiIndex merge restriction
    # and perfectly aligns the data by the date index.
    pipeline.ibov['density'] = b3_net['density']
    pipeline.ibov['max_eigen_value'] = b3_net['max_eigen_value']
    pipeline.calculate_features(window=21)

    pipeline.fit_hmm()

    pipeline.run_backtest(
        initial_capital=100000, 
        stop_loss_pct=STOP_LOSS_PCT, 
        trailing_stop_pct=TRAILING_STOP_PCT
    )

    pipeline.evaluate_performance()