import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

base_dir = '/kaggle/input/child-mind-institute-detect-sleep-states'

transform = None  # 필요한 전처리 변환을 여기에 추가

class ChildInstituteDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            transform (callable, optional): 데이터에 적용할 변환(transform) 함수.
        """
        self.data = data 
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#dataset
train_series_dataset = ChildInstituteDataset(base_dir+'/train_series.parquet', transform=transform)
train_events_dataset = ChildInstituteDataset(base_dir+'/train_events.csv', transform=transform)
test_series_dataset = ChildInstituteDataset(base_dir+'/test_series.parquet', transform=transform)

#batch size
batch_size = 32

#data loader
train_series_data_loader = DataLoader(train_series_dataset, batch_size=batch_size, shuffle=True)
train_events_data_loader = DataLoader(train_events_dataset, batch_size=batch_size, shuffle=False)
test_series_data_loader = DataLoader(test_series_dataset, batch_size=batch_size, shuffle=False)
