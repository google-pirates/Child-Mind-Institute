# pylint: disable=no-member
from typing import Type
import torch
from torch import nn
from torch.utils.data import DataLoader

def inference(model_path: str,
              model: Type[nn.Module],
              test_data_loader: DataLoader) -> torch.Tensor:
    """
    Infer using trained model.

    Parameters:
    - model_path (str): Path to the saved model.
    - model (nn.Module): The model architecture.
    - test_data_loader (DataLoader): DataLoader for the inference.

    Returns:
    - predictions (torch.Tensor): The model's prediction outputs.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for inputs in test_data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())

    return torch.cat(all_preds)
