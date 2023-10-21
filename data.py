from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ChildInstituteDataset(Dataset):
    def __init__(self, data):
        self.data = data 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'X': self.data[idx].filter(items=['anglez', 'enmo']).values,
            'step': self.data[idx].filter(items=['step']).values
        }
 
def to_list(data):
    #series_id 별로 묶인 df 를 리스트화 해서 return
    list_data = []
    for dat in data.groupby('series_id'):
        list_data.append(dat[1])
    return list_data

def preprocessing(data, window_size, **kwargs):
    #train 이든 test든 들어와서 변환할 전처리 과정
    #추후 사용될 전처리 논의 후 추가할 예정
    preprocessed_data = np.lib.stride_tricks.sliding_window_view(data[1].anglez, window_size)

    return preprocessed_data

def dataloader(data, batch_size):
    #dataset
    dataset = ChildInstituteDataset(data)

    #data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader
