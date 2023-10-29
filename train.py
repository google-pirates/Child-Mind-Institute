import argparse
import copy
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import ChildInstituteDataset, preprocess, to_list
from models.cnn import CNN
from utils import load_config, make_logdir, update_config_from_args


def train(config: dict, model: nn.Module, train_dataloader: DataLoader,
          valid_dataloader: DataLoader, writer) -> nn.Module:
    """
    Train model, save and return best model.

    Parameters:
    - config (dict): Configuration dictionary.
    - model (nn.Module): Model to train.
    - train_dataloader (DataLoader): DataLoader for train data.
    - valid_dataloader (DataLoader): DataLoader for validation data.

    Returns:
    - nn.Module: The model based on lowest valid loss.
    """
    save_path = config.get('general').get('save_path')

    criterion = config.get('train').get('criterion')
    optimizer = config.get('train').get('optimizer')
    num_epochs = config.get('train').get('epochs')
    scheduler_class = config.get('train').get('scheduler_class', torch.optim.lr_scheduler.StepLR)
    scheduler_params = config.get('train').get('scheduler_params', {'step_size': 10, 'gamma': 0.1})

    if scheduler_params is None:
        scheduler_params = {'step_size': 10, 'gamma': 0.1} ## need to refactoring

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model = None
    best_loss = float("inf")

    scheduler = scheduler_class(optimizer, **scheduler_params)

    # for tensorboard logging
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        training_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            ## accuracy 추가
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
            train_total_samples += inputs.size(0)
        avg_train_loss = training_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        train_accuracy = train_corrects.double() / train_total_samples
        print(f"Epoch {epoch + 1}/{num_epochs},"
              f"Training Loss: {avg_train_loss},"
              f"Training Accuracy: {train_accuracy}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                ## accuracy 추가
                _, preds = torch.max(outputs, 1)
                valid_corrects += torch.sum(preds == labels.data)
                valid_total_samples += inputs.size(0)

        avg_valid_loss = valid_loss / len(valid_dataloader)
        valid_losses.append(avg_valid_loss)

        valid_accuracy = valid_corrects.double() / valid_total_samples

        print(f"Epoch {epoch + 1}/{num_epochs},"
              f"Validation Loss: {avg_valid_loss},"
              f"Validation accuracy: {valid_accuracy}")
        writer.add_scalar('Loss/valid', avg_valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_accuracy, epoch)



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
            torch.jit.save(scripted_model, save_path)
            print(f"Best model saved to {save_path}")

    return scripted_model

def main(exp_name):
    ## parsing은 script에서 수행

    #### config load ####
    config = load_config()
    # Update exp_name args to 'general' config
    config = update_config_from_args(config, exp_name)
    config_path = config.get('general').get('tensorboard').get('path')

    # Tensorboard
    log_dir = make_logdir("tensorboard", config_path)
    writer = SummaryWriter(log_dir=log_dir)

    ## train data merge
    data_path = config.get('general').get('data').get('data_path')

    merged_train_data = pd.read_parquet(data_path) ## merged_data.parquet

    preprocessed_data = preprocess(merged_train_data)

    window_size = config.get('train').get('window_size')
    step = config.get('train').get('step')

    train_list = to_list(preprocessed_data, window_size, config, step)

    train_data_list = []
    valid_data_list = []

    ## train,test split 시 series_id 별로 split 할지,
    ## 전체 데이터에 대해 split할지 결정 필요
    ## 일단 series_id 별로 split 적용
    ## define valid size from config. default = 0.2
    valid_set_size = config.get('train').get('valid_size', 0.2)

    for df in train_list:
        train_df, valid_df = train_test_split(df, test_size=valid_set_size, shuffle=False)
        train_data_list.append(train_df)
        valid_data_list.append(valid_df)

    train_dataset = ChildInstituteDataset(train_data_list)
    valid_dataset = ChildInstituteDataset(valid_data_list)

    BATCH_SIZE = config.get('train').get('batch_size')
    train_data_shuffle = config.get('train').get('data').get('shuffle')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=train_data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #### train ###
    model_name = config.get('train').get('model')
    model_class = getattr(nn, model_name)
    model = model_class(config)

    trained_model = train(config=config, model=model,
                          train_dataloader=train_dataloader,
                          valid_dataloader=valid_dataloader,
                          writer=writer)
    writer.close()
