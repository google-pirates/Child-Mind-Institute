# pylint: disable=no-member
from typing import Union, Dict, List
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow as pa


class ChildInstituteDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'X': torch.from_numpy(self.data[idx].get('X')),
            'y': torch.Tensor([self.data[idx].get('event')]),
            'series_id': self.data[idx].get('series_id'),
            'date': self.data[idx].get('date'),
            'step': self.data[idx].get('step'),
        }


def preprocess(data, key: List[str] = ['series_id'], **kwargs) -> pd.DataFrame:
    data.rename(columns={'timestamp': 'date'}, inplace=True)

    if isinstance(data.date[0], (np.int0, np.int8, np.int16, np.int32, np.int64)):
        return data

    if not isinstance(data.date[0], datetime):
        data.date = data.date.astype(str).str.replace(r'[-+]\d{2}00$', '', regex=True)
        data.date = pd.to_datetime(
            data.date,
            format='%Y-%m-%dT%H:%M:%S',
            utc=True)
        data.date = data.date.dt.date
        data.date = data.date.astype('datetime64[ns]')

    if isinstance(data.date[0], datetime):
        data.date = (
            data.date
            .fillna(-1)
            .astype(str)
            .str.replace('-', '')
            .str.replace('^20', '', regex=True)
            .astype(np.int32)
        ) - 100000

    return data


def scale(data, config: Dict[str, str]) -> None:
    excluding_columnns = config.get('general').get('data').get('excluding_columns')
    target_columns = list(set(data.columns) - set(excluding_columnns))
    scaler = getattr(preprocessing, config.get('train').get('data').get('scaler'))()

    data.loc[:, target_columns] = scaler.fit_transform(data.filter(items=target_columns))


def to_list(data, window_size: int, config: Dict[str, str], step: int = 1, key: List[str] = ['series_id']) -> List[pd.DataFrame]:
    data = [datum[1] for datum in data.groupby(key)]
    for datum in data:
        scale(datum, config)

    start_of_feature_index = np.where(data[0].columns.str.find('anglez') == 0)[0].item()
    slided_window = [
            np.lib.stride_tricks.sliding_window_view(
                datum.iloc[:, start_of_feature_index:],
                window_size,
                axis=0)[::step]
            for datum
            in data]

    return np.concatenate(slided_window, dtype=np.float32)


def extract_keys(data, window_size: int, step: int = 1, key: List[str] = ['series_id']):
    return (
        data.groupby(key).apply(lambda x: x.iloc[window_size-1:])
        .drop(columns=key)
        .reset_index()
        .drop(columns=[f'level_{len(key)}', 'anglez', 'enmo'])
        .to_dict('records')
    )[::step]
