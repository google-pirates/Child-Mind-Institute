# pylint: disable=no-member
import copy
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
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

def train(config: dict, model: nn.Module, train_dataloader: DataLoader,
          valid_dataloader: DataLoader, writer) -> nn.Module:
    torch.manual_seed(config.get('train').get('random_seed'))

    model_save_dir = os.path.join(writer.log_dir, 'saved_models') 
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, 'best_model.pth')

    optimizer_name = config.get('train').get('optimizer')
    num_epochs = config.get('train').get('epochs')
    lr = config.get('train').get('learning_rate')

    ## Set Criterion
    criterion_config = config.get('train').get('criterion', {})
    criterion_name = criterion_config.get('name')
    criterion_params = criterion_config.get('params', {})
    criterion_class = getattr(nn, criterion_name, None)
    if criterion_class is None:
        raise ValueError(f"Criterion {criterion_name} not found in torch.nn")

    criterion = criterion_class(**criterion_params)

    ## Set Optimizer
    optimizer_class = getattr(optim, optimizer_name, None)
    if optimizer_class is None:
        raise ValueError(f"Optimizer {optimizer_name} not found in torch.optim")

    optimizer = optimizer_class(model.parameters(), lr=lr)

    ## Set scheduler
    scheduler_name = config.get('train').get('scheduler', 'StepLR')
    scheduler_params = config.get('train').get('scheduler_params', {'step_size': 10, 'gamma': 0.1})
    scheduler_class = getattr(lr_scheduler, scheduler_name, None)
    if scheduler_class is None:
        raise ValueError(f"Scheduler {scheduler_name} not found in torch.optim.lr_scheduler")
    scheduler = scheduler_class(optimizer, **scheduler_params)
    if scheduler_params is None:
        scheduler_params = {'step_size': 10, 'gamma': 0.1} ## need to refactoring

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model = None
    best_loss = float("inf")
    patience_count = 0
    patience = config.get('train').get('patience')

    # for tensorboard logging
    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(num_epochs), desc='epoch'):
        # Training phase
        model.train()
        training_loss = 0.0
        train_corrects = 0
        train_total_samples = 0
        train_confusion_matrix = np.zeros((2, 2), dtype=np.int64)
        for batch in tqdm(train_dataloader, desc='train iter'):
            ## data에서 X, y 정의
            inputs = batch['X']
            inputs1 = batch['X1']
            labels = batch['y']
            inputs, inputs1, labels = inputs.to(device), inputs1.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model({'X': inputs, 'X1': inputs1, 'y': labels}) # [batch_size, 1]

            smoothed_labels = labels * (1-0.1) + 0.5*0.1
            loss = criterion(outputs, smoothed_labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            ## accuracy 추가
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            train_corrects += torch.sum(predictions == labels)
            train_total_samples += inputs.size(0)
            train_confusion_matrix += confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())
        avg_train_loss = training_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_accuracy = train_corrects.double() / train_total_samples

        tn, fp, fn, tp = train_confusion_matrix.ravel()
        training_recall = tp / (tp+fn)
        training_precision = tp / (tp+fp)

        print(f"[Epoch {epoch + 1}/{num_epochs}] "
              f"Training Loss: {avg_train_loss:.04f} | "
              f"Training Accuracy: {train_accuracy:.04f} | "
              f"Training Recall: {training_recall:.04f} | "
              f"Training Precision: {training_precision:.04f} | "
              f"Training F1: {2*(training_recall*training_precision)/(training_recall+training_precision):.04f} | ")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Recall/train', training_recall, epoch)
        writer.add_scalar('Precision/train', training_precision, epoch)
        writer.add_scalar('F1/train', 2*(training_recall*training_precision)/(training_recall+training_precision), epoch)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_corrects = 0
        valid_total_samples = 0
        valid_confusion_matrix = np.zeros((2, 2), dtype=np.int64)
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc='valid iter'):
                ## data에서 X, y 정의
                inputs = batch['X']
                inputs1 = batch['X1']
                labels = batch['y']
                inputs, inputs1, labels = inputs.to(device), inputs1.to(device), labels.to(device)
                outputs = model({'X': inputs, 'X1': inputs1, 'y': labels}) # [batch_size, 1]

                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                ## accuracy 추가
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                valid_corrects += torch.sum(predictions == labels)
                valid_total_samples += inputs.size(0)
                valid_confusion_matrix += confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())
        avg_valid_loss = valid_loss / len(valid_dataloader)
        valid_losses.append(avg_valid_loss)

        valid_accuracy = valid_corrects.double() / valid_total_samples

        tn, fp, fn, tp = valid_confusion_matrix.ravel()
        valid_recall = tp / (tp+fn)
        valid_precision = tp / (tp+fp)

        print(f"[Epoch {epoch + 1}/{num_epochs}] "
              f"Validation Loss: {avg_valid_loss:.04f} | "
              f"Validation Accuracy: {valid_accuracy:.04f} | "
              f"Validation Recall: {valid_recall:.04f} | "
              f"Validation Precision: {valid_precision:.04f} | "
              f"Validation F1: {2*(valid_recall*valid_precision)/(valid_recall+valid_precision):.04f} | ")

        writer.add_scalar('Loss/valid', avg_valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_accuracy, epoch)
        writer.add_scalar('Recall/valid', valid_recall, epoch)
        writer.add_scalar('Precision/valid', valid_precision, epoch)
        writer.add_scalar('F1/valid', 2*(valid_recall*valid_precision)/(valid_recall+valid_precision), epoch)

        # ReduceLROnPlateau를 사용하지 않는 경우 lr 업데이트 & 로깅
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss / len(valid_dataloader))
        else:
            scheduler.step()

        current_lr = getattr(scheduler, 'get_last_lr', 
                             lambda: [scheduler.optimizer.param_groups[0]['lr']])()[0]
        writer.add_scalar('lr logging', current_lr, epoch)

        # Save the best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = copy.deepcopy(model).cpu()
            scripted_model = torch.jit.script(best_model)
            torch.jit.save(scripted_model, model_save_path)
            print(f"Best model saved to {model_save_path}")

            patience_count = 0

        else:
            patience_count += 1
        
        # prevent overfitting
        if patience_count > patience:
            break

    return scripted_model

def main(config):
    ## parsing은 script에서 수행

    tensorboard_path = config.get('general').get('tensorboard').get('path')

    # Tensorboard
    log_dir = make_logdir(tensorboard_path, config.get('general').get('exp_name'))
    writer = SummaryWriter(log_dir=log_dir)
    config['general']['log_dir'] = log_dir

    ## train data merge
    data_path = config.get('general').get('data').get('path')

    merged_train_data = pd.read_parquet(data_path) ## merged_data.parquet
    # merged_train_data = merged_train_data.iloc[::20]
    # merged_train_data['timestamp'] = pd.to_datetime(merged_train_data['timestamp'], format='%Y-%m-%d')

    preprocessed_data = preprocess(merged_train_data)

    window_size = int(config.get('train').get('window_size'))
    step = config.get('train').get('step')

    train_list, train_list2 = to_list(preprocessed_data, window_size, config, step)
    train_keys = extract_keys(preprocessed_data, window_size, step)
    for i, train_key in enumerate(train_keys):
        train_key['X'] = train_list[i]
        train_key['X1'] = train_list2[i]
    train_list = train_keys

    valid_set_size = config.get('train').get('valid_size', 0.2)

    train_data_list, valid_data_list = train_test_split(train_list,
                                                        test_size=valid_set_size,
                                                        shuffle=True,
                                                        random_state=config.get('train').get('random_seed'))

    train_dataset = ChildInstituteDataset(train_data_list)
    valid_dataset = ChildInstituteDataset(valid_data_list)

    batch_size = config.get('train').get('batch_size')
    train_data_shuffle = config.get('train').get('data').get('shuffle')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


    example_batch = next(iter(train_dataloader))
    
    _, seq_len, n_features = example_batch['X'].shape
    _, n_features1 = example_batch['X1'].shape
    config['train'].update({'seq_len': seq_len, 'n_features': n_features, 'n_features1': n_features1})

    ### train ###
    model_name = config.get('train').get('model')
    model_class = getattr(models, model_name)

    model = model_class(config)

    train(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        writer=writer,
    )

    writer.close()

    with open(f'./{log_dir}/configs.pickle', 'wb') as file:
        pickle.dump(config, file)
