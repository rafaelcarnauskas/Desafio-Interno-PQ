import pandas as pd
import numpy as np
from scipy.special import comb
#from sklearn.model_selection import train_test_split

"""THRESHOLD = 0.2 # Ao invés de fixar, usa-se o percentil"""
THRESHOLD = 0.7 #usar threshold fixo e nao percentil torna a medida de densidade util


df = pd.read_csv("b3-2010-2025-Close.csv")

df['Log_Close'] = np.log10(df['Close'])

df['Returns'] = df['Log_Close'].diff()

def net_analysis(df, THRESHOLD):

    '''Considera-se uma aresta da rede como uma correlação que passa o THRESHOLD
        A densidade da rede é calculada com número de arestas/máximo de arestas possíveis
        A quantidade máxima de arestas é o coeficiente de newton com n e 2
        A matriz de correlação é calculada a partir dos log-retornos'''
    
    corr = df.corr()
    columns = df.columns.tolist()

    """# Identificação dos maiores valores de correlação
    # Por enquanto considera-se o valor real, independentemente do sinal pois valores positivos medem dependência direta
    corr_values = corr.values[np.triu_indices_from(corr.values, k=1)]
    threshold_value = np.quantile(corr_values, 1 - THRESHOLD)"""


    #proximo passo é analisar autovalor (eigenvector e eigenvalue) e degree centrality com eigenvector centrality
    #basicamente analisar se tem algum ativo muito conectado com muita gente

    """A análise do autovalor da matriz de correlação consiste em medir a centralidade da rede
    se o maior autovalor da matriz A crescer muito e de forma rápida quer dizer que um autovetor 
    está ficando concentrado em uma direção e isso indica fragilidade sistêmica"""

    #O método net_analysis() vai devolver o autovalor máximo e seu autovetor associado para que outro método use essas informações

    n = len(columns)
    A = np.zeros((n,n))
    edges = []

    # Loop para construir a matriz de adjacência
    for i in range(n):
        for j in range(n):
            if i==j:
                A[i, j] = 0
            else:
                A[i, j] = corr[i, j] if corr[i, j] >= THRESHOLD else 0
                """Da forma que está, o autovetor medirá intensidade das conexões
                    Se for preciso medir quantidade de conexões, faz A[i,j]=1"""
                if j < i and A[i, j]:
                    edges.append((columns[i], columns[j]))


    possible_edges = comb(n, 2, exact=True)
    density = len(edges) / possible_edges if possible_edges > 0 else 0 


    # Por enquanto não vejo sentido em implementar medida de diâmetro da matriz de adjacência

    """Quando o maior autovalor aumenta:
        -Densidade tende a aumentar
        -Diâmetro tende a diminuir (rede mais compacta)
        -Eigenvector centrality fica mais concentrada"""

    # Seção para o cálculo do autovetor e autovalor
    # A posição i do autovetor associado ao maior autovalor vai indicar o grau de conectividade do ativo i
    # Ao final é interessante normalizar o autovetor para ter uma medida de 0 a 1

    eigen_values, eigen_vector = np.linalg.eigh(A)

    max_eigen_value = eigen_values[-1] # Captura o maior autovalor
    max_eigen_vector = eigen_vector[:, -1]

    norm_eigen_vector = (max_eigen_vector - max_eigen_vector.min()) / (max_eigen_vector.max() - max_eigen_vector.min()) if max_eigen_vector.max() != max_eigen_vector.min() else max_eigen_vector

    return A, max_eigen_value, norm_eigen_vector, edges, density
