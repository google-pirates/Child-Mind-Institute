import argparse
import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, MultiStepLR, ReduceLROnPlateau
)
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from models.cnn import CNN
import yaml
from dataloader import preprocessing, to_list, ChildInstituteDataset

def train(
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int,
        save_path: str,
        scheduler_class=StepLR,
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
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: "
              f"{training_loss / len(train_dataloader)}")

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
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: "
              f"{valid_loss / len(valid_dataloader)}")

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

    return scripted_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    #### config load ####
    config = {}
    with open('configs/general_config.yaml', 'r') as f:
        general_config = yaml.safe_load(f)
    with open('configs/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    with open('configs/train_config.yaml', 'r') as f:
        train_config = yaml.safe_load(f)

    if general_config:
        config.update(general_config)
    if model_config:
        config.update(model_config)
    if train_config:
        config.update(train_config)


    # Tensorboard
    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}")

    DATA_PATH = "train_data"

    preprocessed_data = preprocessing(DATA_PATH)

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

    trained_model = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion=config.get('train').get('criterion'),
        optimizer = config.get('train').get('Adam') ,
        num_epochs=config.get('train').get('epochs'),
        save_path=f"saved_models/{args.exp_name}.pt"
    )
