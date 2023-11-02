# pylint: disable=no-member
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from data import ChildInstituteDataset, preprocess, to_list


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
        for batch in tqdm(test_dataloader, desc='Inference'):
            inputs = batch['X'].to(device)
            series_id = batch['series_id'][-1].item()
            step = batch['step'][-1].item()

            outputs = model({'X': inputs})

            probabilities = torch.sigmoid(outputs)
            prediction = (probabilities > 0.5).float()
            score = probabilities.item()

            event = 'wakeup' if prediction == 1 else 'onset'

            all_series_ids.append(series_id)
            all_steps.append(step)
            all_events.append(event)
            all_scores.append(score)

    submission = pd.DataFrame({
        'series_id': all_series_ids,
        'step': all_steps,
        'event': all_events,
        'score': all_scores
    })

    submission.to_csv('submission.csv', index=False)

def main(config):
    # Load preprocessed data for inference
    test_data_path = config.get('general').get('test_data').get('path')
    test_data = pd.read_parquet(test_data_path)

    ## test data를 train data와 동일한 형식으로 변경
    test_data['event'] = False
    data = test_data[['series_id', 'timestamp', 'step', 'event', 'anglez', 'enmo']]
    data['series_id'], unique_series = pd.factorize(data['series_id'])

    ## series_id map
    id_map = [id for id in unique_series]

    preprocessed_data = preprocess(data)

    window_size = config.get('inference').get('window_size')
    step = config.get('inference').get('step')

    test_list = to_list(preprocessed_data, window_size, config, step)
    test_dataset = ChildInstituteDataset(test_list)

    batch_size = config.get('inference').get('batch_size')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    inference(model_path=config.get('general').get('checkpoint'), test_dataloader=test_dataloader)
