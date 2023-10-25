# pylint: disable=no-member
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
from dataloader import ChildInstituteDataset, to_list, preprocess
from utils import load_config


def inference(model_path: str, test_dataloader: DataLoader):
    """
    Infer using trained model.

    Parameters:
    - model_path (str): Path to the saved model.
    - test_dataloader (DataLoader): DataLoader for the inference.

    Returns:
    - DataFrame: Contains series_id, step, event, and score columns.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    all_series_ids = []
    all_steps = []
    all_events = []
    all_scores = []

    with torch.no_grad():
        for data in test_dataloader:
            inputs = data["data"].to(device)
            series_id = data["series_id"]
            event_step = data["step"]
            
            outputs = model(inputs)

            probabilities = torch.softmax(outputs, dim=1)

            _, preds = torch.max(probabilities, 1)

            # 3 events, none, onset and wakeup
            event_mapping = {0: "", 1: "onset", 2: "wakeup"}
            events = [event_mapping[pred] for pred in preds.cpu().numpy()]

            # Scoring using logit values
            scores = torch.sigmoid(outputs).cpu().numpy()
            scores = [scores[i, pred].item() for i, pred in enumerate(preds.cpu().numpy())]

            all_series_ids.extend(series_id)
            all_steps.extend(event_step)
            all_events.extend(events)
            all_scores.extend(scores)

    # Make submission file

    submission = pd.DataFrame({
        'series_id': all_series_ids,
        'step': all_steps,
        'event': all_events,
        'score': all_scores
    })

    submission.to_csv('submission.csv', index=False)

    return submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference using trained model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    #### config load ####
    config = load_config()

    # Load preprocessed data for inference
    data_path = config.get('general').get('test_data').get('data_path')
    preprocessed_data = preprocess(data_path)

    # Convert data to DataLoader format
    test_list = to_list(preprocessed_data)
    test_dataset = ChildInstituteDataset(test_list)

    BATCH_SIZE = config.get('inference').get('batch_size')
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Perform inference
    submission = inference(model_path=args.checkpoint, test_dataloader=test_dataloader)

