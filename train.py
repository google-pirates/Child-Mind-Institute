# pylint: disable=no-member
import copy
import os
import pickle

import numpy as np
import pandas as pd
import re
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import ChildInstituteDataset, preprocess, to_list, extract_keys
from utils import make_logdir
import models

def train(config, model, train_dataloader, valid_dataloader):
    torch.manual_seed(config.get('train').get('random_seed'))

    criterion_config = config.get('train').get('criterion', {})
    criterion_name = criterion_config.get('name')
    criterion_params = criterion_config.get('params', {})
    criterion_class = getattr(nn, criterion_name, None)
    if criterion_class is None:
        raise ValueError(f"Criterion {criterion_name} not found in torch.nn")

    criterion = criterion_class(**criterion_params)

    optimizer_name = config.get('train').get('optimizer')
    optimizer_class = getattr(torch.optim, optimizer_name, None)
    if optimizer_class is None:
        raise ValueError(f"Optimizer {optimizer_name} not found in torch.optim")

    lr = config.get('train').get('learning_rate')
    optimizer = optimizer_class(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    total_train_loss = 0.0
    total_train_corrects = 0
    total_train_samples = 0
    
    train_labels = []
    train_preds = []
    
    model.train()
    for batch in tqdm(train_dataloader, desc='Training'):
        inputs = batch['X'].to(device)
        labels = batch['y'].to(device).squeeze()

        optimizer.zero_grad()
        outputs, hidden = model({'X':inputs, 'y':labels})
        loss = criterion(outputs.reshape(-1, config.get('train').get('n_features')), labels.long().flatten())
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(axis=-1)
        total_train_corrects += torch.sum(preds == labels)
        total_train_samples += inputs.size(0)
        
        train_labels.extend(labels.cpu().numpy())
        train_preds.extend(preds.cpu().numpy())

    avg_train_loss = total_train_loss / total_train_samples
    train_accuracy = total_train_corrects.double() / total_train_samples

    # Validation loop
    total_valid_loss = 0.0
    total_valid_corrects = 0
    total_valid_samples = 0

    valid_labels = []
    valid_preds = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc='Validation'):
            inputs = batch['X'].to(device)
            labels = batch['y'].to(device).squeeze()

            outputs, hidden = model({'X':inputs, 'y':labels})
            loss = criterion(outputs.reshape(-1, config.get('train').get('n_features')), labels.long().flatten())

            total_valid_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_valid_corrects += torch.sum(preds == labels)
            total_valid_samples += inputs.size(0)
            valid_labels.extend(labels.cpu().numpy())
            valid_preds.extend(preds.cpu().numpy())

    avg_valid_loss = total_valid_loss / total_valid_samples
    valid_accuracy = total_valid_corrects.double() / total_valid_samples

    return avg_train_loss, train_accuracy, train_preds, train_labels, avg_valid_loss, valid_accuracy, valid_preds, valid_labels

def main(config):
    ## Get n_features and seq_len
    ex_data = pd.read_parquet(config['general']['data']['path'])
    ex_data = ex_data.iloc[:config['train'][window_size]+1]
    label_mapping = {0: 0, 1: 1, 3: 2, 4: 3}
    ex_data['event'] = [label_mapping[label] for label in ex_data['event']]
    preprocessed_ex_data = preprocess(ex_data)
    ex_list = to_list(preprocessed_ex_data, config['train']['window_size'], config, 1)
    ex_keys = extract_keys(preprocessed_ex_data, config['train']['window_size'],1)
    for i, key in enumerate(ex_keys):
        key['X'] = ex_list[i][:,1:]
        key['event'] = ex_list[i][0]
    ex_list = ex_keys

    ex_dataset = ChildInstituteDataset(ex_list)
    ex_dataloader = DataLoader(ex_dataset, batch_size=128, shuffle=False)

    example_batch = next(iter(ex_dataloader))
    _, n_features, seq_len = example_batch['X'].shape

    config['train'].update({'seq_len': seq_len, 'n_features': n_features})
    ##
    tensorboard_path = config.get('general').get('tensorboard').get('path')

    log_dir = make_logdir(tensorboard_path, config.get('general').get('exp_name'))

    model_save_dir = os.path.join(log_dir, 'saved_models') 
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, 'best_model.pth')

    data_path = config.get('general').get('data').get('path')

    model_name = config.get('train').get('model')
    model_class = getattr(models, model_name)
    model = model_class(config)

    series_ids = pd.read_parquet(data_path, columns=['series_id']).series_id.unique()
    
    for epoch in tqdm(range(config.get('train').get('epochs'))):
        total_train_loss = 0.0
        total_valid_loss = 0.0
        total_train_corrects = 0
        total_valid_corrects = 0
        total_train_samples = 0
        total_valid_samples = 0
        all_train_preds = []
        all_train_labels = []
        all_valid_preds = []
        all_valid_labels = []

        for series_id in series_ids:
            series_data = pd.read_parquet(data_path, filters=[('series_id', '=', series_id)])
            series_data.event = series_data.groupby(['series_id']).event.shift(1)
            series_data.dropna(inplace=True)
            series_data.reset_index(drop=True, inplace=True)
            series_data.date = series_data.date.astype(np.int32)
            
            ##Label mapping
            label_mapping = {0: 0, 1: 1, 3: 2, 4: 3}
            series_data['event'] = [label_mapping[label] for label in series_data['event']]
            
            preprocessed_data = preprocess(series_data)

            window_size = int(config.get('train').get('window_size'))
            step = config.get('train').get('step')

            train_list = to_list(preprocessed_data, window_size, config, step)
            train_keys = extract_keys(preprocessed_data, window_size, step)

            for i, train_key in enumerate(train_keys):
                train_key['X'] = train_list[i][:, 1:]
                train_key['event'] = train_list[i][0]
            train_list = train_keys

            valid_set_size = config.get('train').get('valid_size', 0.2)
            train_data_list, valid_data_list = train_test_split(train_list, test_size=valid_set_size, shuffle=True, random_state=config.get('train').get('random_seed'))

            train_dataset = ChildInstituteDataset(train_data_list)
            valid_dataset = ChildInstituteDataset(valid_data_list)
            
            batch_size = config.get('train').get('batch_size')
            train_data_shuffle = config.get('train').get('data').get('shuffle')
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_data_shuffle)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            
            train_loss, train_acc, train_preds, train_labels, valid_loss, valid_acc, valid_preds, valid_labels = train(
                config,
                model,
                train_dataloader,
                valid_dataloader
            )

            total_train_loss += train_loss * len(train_dataloader.dataset)
            total_train_corrects += train_acc * len(train_dataloader.dataset)
            total_train_samples += len(train_dataloader.dataset)
            all_train_preds.extend(train_preds)
            all_train_labels.extend(train_labels)

            total_valid_loss += valid_loss * len(valid_dataloader.dataset)
            total_valid_corrects += valid_acc * len(valid_dataloader.dataset)
            total_valid_samples += len(valid_dataloader.dataset)
            all_valid_preds.extend(valid_preds)
            all_valid_labels.extend(valid_labels)

        avg_train_loss = total_train_loss / total_train_samples
        avg_valid_loss = total_valid_loss / total_valid_samples
        train_accuracy = total_train_corrects / total_train_samples
        valid_accuracy = total_valid_corrects / total_valid_samples

        print(f'Epoch {epoch+1}/{config.get("train").get("epochs")}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
        print(classification_report(train_labels, train_preds, target_names=['Class 0', 'Class 1', 'Class 3', 'Class 4']))
        print(f'Valid Loss: {avg_valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}')
        print(classification_report(valid_labels, valid_preds, target_names=['Class 0', 'Class 1', 'Class 3', 'Class 4']))


        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_model = copy.deepcopy(model)
            torch.jit.save(torch.jit.script(best_model), model_save_path)

    with open(f'.{log_dir}/configs.pickle', 'wb') as file:
        pickle.dump(config, file)