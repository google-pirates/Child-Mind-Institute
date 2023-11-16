# pylint: disable=no-member
from typing import Union, Dict, List
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os


class ChildInstituteDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'X': torch.from_numpy(self.data[idx].get('X')),
            'X1': torch.from_numpy(self.data[idx].get('X1')),
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
        data['hour'] = data.date.dt.hour

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


def scale(data, config: Dict[str, str], scaler: object = None) -> None:
    excluding_columnns = config.get('general').get('data').get('excluding_columns')
    target_columns = list(set(data.columns) - set(excluding_columnns))

    scaler_file_path = '/kaggle/input/models/acc,step=2,window_size=120,batch_size=256,lr=0.001,RADam/version_0/scaler.pickle'

    if os.path.exists(scaler_file_path):
        with open(scaler_file_path, 'rb') as file:
            scaler = pickle.load(file)
    else:
        if scaler is None:
            scaler = getattr(preprocessing, config.get('train').get('data').get('scaler'))()
        with open(f'{config.get("general").get("log_dir")}/scaler.pickle', 'wb') as file:
            pickle.dump(scaler, file)

    data.loc[:, target_columns] = scaler.fit_transform(data.filter(items=target_columns))


def to_list(data, window_size: int, config: Dict[str, str], step: int = 1, key: List[str] = ['series_id'], scaler: object = None) -> List[pd.DataFrame]:
    data = [datum[1] for datum in data.groupby(key)]
    for datum in data:
        scale(datum, config, scaler)
    
    slided_windows = []
    start_of_feature_index = np.where(data[0].columns.str.find('anglez') == 0)[0].item()

    slided_window = [
        np.lib.stride_tricks.sliding_window_view(
            datum.iloc[1:, -1:],
            window_size,
            axis=0)[::step]
        for datum
        in data]
    slided_windows.append(np.concatenate(slided_window))


    slided_window = [
            np.lib.stride_tricks.sliding_window_view(
                datum[['anglez', 'enmo']].diff().iloc[1:],
                window_size,
                axis=0)[::step]
            for datum
            in data]
    # slided_window = [
    #         np.lib.stride_tricks.sliding_window_view(
    #             datum.diff().iloc[1:, start_of_feature_index:-1],
    #             window_size,
    #             axis=0)[::step]
    #         for datum
    #         in data]
    
    slided_windows.append(np.concatenate(slided_window))
    slided_windows = np.concatenate(slided_windows, axis=1, dtype=np.float32)
    
    slided_windows_2 = []
    for win_size in range(window_size, 0, -30):
        slided_window = [
            np.lib.stride_tricks.sliding_window_view(
                datum.iloc[1:, start_of_feature_index:-1],
                win_size,
                axis=0)[window_size-win_size::step].mean(axis=-1)
            for datum
            in data]
        slided_windows_2.append(np.concatenate(slided_window))
    slided_windows_2 = np.concatenate(slided_windows_2, axis=1, dtype=np.float32)

    return slided_windows, slided_windows_2


def extract_keys(data, window_size: int, step: int = 1, key: List[str] = ['series_id']):
    return (
        data.groupby(key).apply(lambda x: x.iloc[window_size:])
        .drop(columns=key)
        .reset_index()
        .drop(columns=[f'level_{len(key)}', 'anglez', 'enmo'])
        .to_dict('records')
    )[::step]
