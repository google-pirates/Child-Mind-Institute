# pylint: disable=no-member
from typing import Union, Dict, List
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import pandas as pd
import numpy as np


class ChildInstituteDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'X': torch.from_numpy(self.data[idx][:, 4:].astype(np.float32)),
            'y': torch.from_numpy(self.data[idx][:, 3][-1:].astype(np.float32)),
            'series_id': torch.from_numpy(self.data[idx][:, 0][:1].astype(np.int32)),
            'date': self.data[idx][:, 1][-1:],
            'step': torch.from_numpy(self.data[idx][:, 2][-1:][-1:].astype(np.int32)),
        }


def preprocess(data, key: Union[str, List[str]] = 'series_id', **kwargs) -> pd.DataFrame:
        
    return data


def scale(data, config: Dict[str, str]) -> None:
    excluding_columnns = config.get('general').get('data').get('excluding_columns')
    target_columns = list(set(data.columns) - set(excluding_columnns))
    scaler = getattr(preprocessing, config.get('train').get('data').get('scaler'))()

    data.loc[:, target_columns] = scaler.fit_transform(data.filter(items=target_columns))


def to_list(data, window_size: int, config: Dict[str, str], step: int = 1, key: Union[str, List[str]] = 'series_id') -> List[pd.DataFrame]:
    data = [datum[1] for datum in data.groupby(key)]
    for datum in data:
        scale(datum, config)

    ## dtype=object 
    slided_window = np.array([np.lib.stride_tricks.sliding_window_view(datum, window_size, axis=0)[::step] 
                              for datum in data], dtype=object)

    return np.concatenate(slided_window).swapaxes(2, 1)
