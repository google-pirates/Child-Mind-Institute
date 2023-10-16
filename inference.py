# pylint: disable=no-member
import torch
from torch.utils.data import DataLoader

def inference(model_path: str,
              test_data_loader: DataLoader) -> torch.Tensor:
    """
    Infer using trained model.

    Parameters:
    - model_path (str): Path to the saved model.
    - test_data_loader (DataLoader): DataLoader for the inference.

    Returns:
    - predictions (torch.Tensor): The model's prediction outputs.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for inputs in test_data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            probabilities = torch.softmax(outputs, dim=1)

            _, preds = torch.max(probabilities, 1)

            all_preds.append(preds.cpu())


    ## Submission phase
    submission_columns = ['series_id','step','event','score']
    submission = None
    submission.to_csv('submission.csv', index=False)
    ## Points for discussion

    return torch.cat(all_preds)
