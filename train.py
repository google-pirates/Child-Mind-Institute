import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def train(model: nn.Module,
          train_dataloader: DataLoader,
          valid_dataloader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          num_epochs: int,
          save_path: str,
          scheduler_step_size: int,
          scheduler_gamma: float,
          **kwargs) -> nn.Module:
    """
    Train model, save and return best model.

    Parameters:
    - model (nn.Module): Model to train.
    - train_dataloader (DataLoader): DataLoader for train
    - valid_dataloader (DataLoader): DataLoader for valid
    - criterion (nn.Module): Loss func.
    - optimizer (optim.Optimizer): Optimizer
    - num_epochs (int): Number of epochs
    - save_path (str): path to save the best model.
    - scheduler_step_size (int): Number of scheduler reduction step
    - scheduler_gamma (float): learning rate reduction ratio.

    Returns:
    - best_model (nn.Module): The model based on lowest valid loss.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    for epoch in range(num_epochs):
        ## training phase
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
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {training_loss/len(train_dataloader)}")
        scheduler.step()
        
        ## validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {valid_loss/len(valid_dataloader)}")

        ## save the best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = copy.deepcopy(model.state_dict())
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, save_path)
            print(f"Best model saved to {save_path}")

    model.load_state_dict(best_model)

    return model
