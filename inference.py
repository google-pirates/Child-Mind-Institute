# pylint: disable=no-member
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from data import ChildInstituteDataset, preprocess, to_list, extract_keys

def inference(model_path: str, test_dataloader: DataLoader) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Inference'):
            inputs, series_ids, steps, dates = batch['X'].to(device), batch['series_id'], batch['step'].cpu().numpy(), batch['date'].cpu().numpy()
            outputs = model({'X': inputs})
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            scores = probabilities.detach().cpu().numpy().max(axis=1)

            results.extend(zip(series_ids, steps, dates, predictions, scores))

    return pd.DataFrame(results, columns=['series_id', 'step', 'date', 'event', 'score'])

def reverse_events_by_step(submission: pd.DataFrame, event, min_step: int) -> pd.DataFrame:
    submission['change_point'] = submission['event'].diff().ne(0).astype('int')
    submission.iloc[0, submission.columns.get_loc('change_point')] = 1

    change_points = submission[submission['change_point'] == 1].index.tolist()
    change_points.append(len(submission))

    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1] - 1

        current_event = submission.loc[start_idx, 'event']
        step_difference = submission.loc[end_idx, 'step'] - submission.loc[start_idx, 'step']

        if current_event == event and step_difference <= min_step:
            submission.loc[start_idx:end_idx, 'event'] = 1 - submission.loc[start_idx:end_idx, 'event']

    submission = submission.drop(columns=['change_point'])

    return submission

def postprocess_submission(submission: pd.DataFrame) -> pd.DataFrame:
    submission['event'] = np.where(submission['event'] == 3, 1, submission['event'])
    submission = submission[submission['event'] != 4]

    submission['event'] = submission['event'].rolling(window=30, min_periods=1, center=True).mean().apply(
        lambda x: 1 if x > 0.5 else (0 if pd.notnull(x) else np.nan)
    )

    submission = reverse_events_by_step(submission, 0, 300)
    submission = reverse_events_by_step(submission, 1, 300)
    
    first_rows = submission.groupby('series_id').head(1)
    submission['event_change'] = submission['event'] != submission['event'].shift(1)
    submission.loc[0, 'event_change'] = False

    changed_events = submission[submission['event_change']]
    submission = pd.concat([first_rows, changed_events]).drop_duplicates().sort_values(['series_id', 'step'])
    submission = submission.drop(columns=['event_change'])
        
    return submission

def prepare_final_submission(submission: pd.DataFrame) -> pd.DataFrame:
    submission['event'] = submission['event'].map({0: 'onset', 1: 'wakeup'})
    submission['score'] = submission['score'].astype(float) / 100

    submission = submission.sort_values(['series_id', 'step']).reset_index(drop=True)
    submission['row_id'] = submission.index.astype(int)

    submission = submission[['row_id', 'series_id', 'step', 'event', 'score']]
    submission.to_csv('submission.csv', index=False, float_format='%.5f')
    return submission

def main(config) -> pd.DataFrame:
    test_data = pd.read_parquet(config['general']['test_data']['path'])
    test_data['event'] = -1
    window_size, step_size = config['inference']['window_size'], config['inference']['step']

    submissions = []
    for series_id in test_data['series_id'].unique():
        series_data = preprocess(test_data[test_data['series_id'] == series_id].reset_index(drop=True))
        if len(series_data) <= 360:
            continue
        
        test_list = to_list(series_data, window_size, config, step_size)
        test_keys = extract_keys(series_data, window_size, step_size)
        test_dataset = ChildInstituteDataset([{'X': test_list[i], **key} for i, key in enumerate(test_keys)])
        
        test_dataloader = DataLoader(test_dataset, batch_size=config['inference']['batch_size'], shuffle=False, pin_memory=True, num_workers= int(os.cpu_count()/2))
        submission = inference(config['general']['checkpoint'], test_dataloader)
        submissions.append(postprocess_submission(submission))

    final_submission = pd.concat(submissions).reset_index(drop=True) if submissions else pd.DataFrame(columns=['row_id', 'series_id', 'step', 'event', 'score'])
    return prepare_final_submission(final_submission)