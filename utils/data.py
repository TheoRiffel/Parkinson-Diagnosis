import os
import numpy as np
import pandas as pd
from tqdm import tqdm

avaiable_atlas = ['Shen_268', 'atlas', 'AAL3']

def batch_read(path: str) -> list[pd.DataFrame]:
    """Read .csv timeseries from folder"""
    df_list = []
    file_names = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    for file in file_names:
        df = pd.read_csv(f'{path}/{file}')
        df_list.append(df)
    return df_list

def select_atlas_columns(
        data: list[pd.DataFrame],
        atlas_name: str
    ) -> list[pd.DataFrame]:
    """Select atlas columns from df"""

    if atlas_name not in avaiable_atlas:
        raise ValueError(f'Invalid atlas name {atlas_name}. Use {avaiable_atlas}')
    
    all_columns = data[0].columns
    selected_columns = all_columns[np.where([column.split('.')[0] == atlas_name for column in all_columns])[0]]

    selected_data = [df[selected_columns] for df in data]

    return selected_data

def concatenate_data(
        *data_arrays: list[np.array],
    ) -> np.array:
    """
    Stack two or more list[np.array]
    """

    if not data_arrays:
        raise ValueError("At least one list of numpy arrays must be provided for concatenation.")

    all_arrays = [arr for arr in data_arrays]

    return np.concatenate(all_arrays, axis=0)


def remove_nan(X: np.array, y: np.array) -> tuple[np.array, np.array]:
    """
    Substitui NaNs por 0 nas séries
    """
    X = np.nan_to_num(X, nan=0.0)
    return X, y


def get_triu(       
        data: np.array,
        k: int = 0
    ) -> np.array:

    """
    Get upper triangle of a square matrix for every sample in data.
    """
    triu_data = []
    for sample in tqdm(data, desc='Getting upper triangle'):
        triui = np.triu_indices_from(sample, k=k)
        triu_data.append(sample[triui])

    return np.array(triu_data)
