import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, MultiStepLR, ReduceLROnPlateau
)
from torch.utils.data import DataLoader

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
