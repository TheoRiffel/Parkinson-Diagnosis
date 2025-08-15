import numpy as np
import pandas as pd

def pearson_correlation(time_series_list: list[pd.DataFrame]) -> list[np.array]:
    """
    Retorna a correlação de Pearson para a lista de séries temporais
    """
    return [ts.corr(method='pearson').to_numpy() for ts in time_series_list]

def spearman_correlation(time_series_list: list[pd.DataFrame]) -> list[np.array]:
    """
    Retorna a correlação de Spearman para a lista de séries temporais
    """
    return [ts.corr(method='spearman').to_numpy() for ts in time_series_list]