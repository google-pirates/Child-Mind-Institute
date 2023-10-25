import argparse
import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from models.cnn import CNN
import yaml
from dataloader import preprocess, to_list, ChildInstituteDataset
from util import load_config, make_logdir


def train(
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int,
        save_path: str,
        scheduler_class=torch.optim.lr_scheduler.StepLR,
        scheduler_params=None,
        **kwargs
) -> nn.Module:
    """
    Train model, save and return best model.

    Parameters:
    - model (nn.Module): Model to train.
    - train_dataloader (DataLoader): DataLoader for train data.
    - valid_dataloader (DataLoader): DataLoader for validation data.
    - criterion (nn.Module): Loss function.
    - optimizer (optim.Optimizer): Optimizer.
    - num_epochs (int): Number of training epochs.
    - save_path (str): Path to save the best model.
    - scheduler_class (Scheduler): The learning rate scheduler class. 
      Defaults to StepLR. Options include: StepLR, ExponentialLR, MultiStepLR, ReduceLROnPlateau.
    - scheduler_params (dict): Parameters to initialize the scheduler. 
      Defaults to StepLR: {'step_size': 10, 'gamma': 0.1}.

    Returns:
    - nn.Module: The model based on lowest valid loss.
    """
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
        avg_train_loss = training_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}")

        # Scheduler step (if not using ReduceLROnPlateau)
        if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        avg_valid_loss = valid_loss / len(valid_dataloader)
        valid_losses.append(avg_valid_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_valid_loss}")

        # Scheduler step (if using ReduceLROnPlateau)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss / len(valid_dataloader))

        # Save the best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = copy.deepcopy(model).cpu()
            scripted_model = torch.jit.script(best_model)
            torch.jit.save(scripted_model, save_path)
            print(f"Best model saved to {save_path}")

    return scripted_model, train_losses, valid_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    #### config load ####
    config = load_config()

    # Tensorboard
    log_dir = make_logdir("runs", args.exp_name)
    writer = SummaryWriter(log_dir=log_dir)

    preprocessed_data = preprocess(config.get('general').get('data').get('data_path'))

    ## train,test split 시 series_id 별로 split 할지, 
    ## 전체 데이터에 대해 split할지 결정 필요
    ## 일단 series_id 별로 split 적용
    train_list = to_list(preprocessed_data)

    train_data_list = []
    valid_data_list = []
    for df in train_list:
        train_size = int(.8*len(df))

        train_df = df[:train_size]
        valid_df = df[train_size:]

        train_data_list.append(train_df)
        valid_data_list.append(valid_df)

    train_dataset = ChildInstituteDataset(train_data_list)
    valid_dataset = ChildInstituteDataset(valid_data_list)

    BATCH_SIZE = config.get('train').get('batch_size')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #### train ###
    model_name = config.get('train').get('model')
    model_class = getattr(nn, model_name)
    model = model_class(config)

    # get Scheduler
    scheduler_name = config.get('train').get('scheduler')
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)

    trained_model, train_loss, valid_loss = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion=config.get('train').get('criterion'),
        optimizer = config.get('train').get('Adam') ,
        num_epochs=config.get('train').get('epochs'),
        save_path=f"{log_dir}/{args.exp_name}.pt",
        scheduler_class=scheduler_class,
        scheduler_params=config.get('train').get('scheduler_params')
    )

    for epoch, (train_loss_by_epoch, valid_loss_by_epoch) in enumerate(zip(train_loss,valid_loss)):
        writer.add_scalar('Loss/train', train_loss_by_epoch, epoch)
        writer.add_scalar('Loss/valid', valid_loss_by_epoch, epoch)

