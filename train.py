from typing import Type
import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def train(model: Type[nn.Module],
          train_data_loader: DataLoader,
          val_data_loader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          num_epochs: int,
          save_path: str) -> Type[nn.Module]:
    """
    Train model, save and return best model.

    Parameters:
    - model (nn.Module): Model to train.
    - train_data_loader (DataLoader): DataLoader for train
    - val_data_loader (DataLoader): DataLoader for valid
    - criterion (nn.Module): Loss func.
    - optimizer (optim.Optimizer): Optimizer
    - num_epochs (int): Number of epochs
    - save_path (str): path to save the best model.
    
    Returns:
    - best_model (nn.Module): The model based on lowest valid loss.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):
        ## training phase
        model.train()
        running_loss_train = 0.0
        for inputs, labels in train_data_loader():
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss_train/len(train_data_loader)}")

        ## validation phase
        model.eval()
        running_loss_val = 0.0
        with torch.no_grad():
            for inputs, labels in val_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {running_loss_val/len(val_data_loader)}")

        ## save the best model
        if running_loss_val < best_loss:
            best_loss = running_loss_val
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, save_path)
            print(f"Best model saved to {save_path}")

    model.load_state_dict(best_model)

    return model
