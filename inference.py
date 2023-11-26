# pylint: disable=no-member
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle

from data import ChildInstituteDataset, preprocess, to_list, extract_keys

def inference(model_path: str, test_dataloader: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    all_series_ids = []
    all_steps = []
    all_events = []
    all_scores = []
    all_dates = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Inference'):
            inputs = batch['X'].to(device)
            series_ids = batch['series_id']
            steps = batch['step'].cpu().numpy()
            dates = batch['date'].cpu().numpy()
            outputs = model({'X': inputs})

            predictions = outputs.argmax(axis=-1)
            probabilities = torch.sigmoid(outputs)
            scores = probabilities.detach().cpu().numpy()

            all_series_ids.extend(series_ids.tolist())
            all_steps.extend(steps.tolist())
            all_dates.extend(dates.tolist())

            batch_predictions = predictions.cpu().numpy().astype(int).tolist()
            for pred in batch_predictions:
                all_events.append(pred)

            batch_scores = scores.tolist()
            for score in batch_scores:
                all_scores.append(score[0])

    submission = pd.DataFrame({
        'series_id': all_series_ids,
        'step': all_steps,
        'date': all_dates,
        'event': all_events,
        'score': all_scores
    })

    return submission


def main(config):
    # Load preprocessed data for inference
    test_data_path = config.get('general').get('test_data').get('path')
    
    test_data = pd.read_parquet(test_data_path)
    test_data['event'] = -1

    labels = np.zeros(shape=((len(test_data), 5)))
    labels[test_data.event==0, 0] = 1
    labels[test_data.event==1, 1] = 1
    labels[test_data.event==2, 2] = 1
    labels[test_data.event==3, 3] = 1
    labels[test_data.event==4, 4] = 1

    with open('./data/id_map.pickle', 'rb') as handle:
            id_map = pickle.load(handle)
    reverse_id_map = {v: k for k, v in id_map.items()}
    
    window_size = config.get('inference').get('window_size')
    step_size = config.get('inference').get('step')

    unique_series_ids = test_data['series_id'].unique()
    all_submissions = []

    for series_id in unique_series_ids:
        if series_id > 0:
            break
        series_data = test_data[test_data['series_id'] == series_id].copy()
        series_data.reset_index(drop=True, inplace=True)

        preprocessed_series_data = preprocess(series_data)

        test_list = to_list(preprocessed_series_data, window_size, config, step_size)
        test_keys = extract_keys(preprocessed_series_data, window_size, step_size)

        for i, key in enumerate(test_keys):
            key['X'] = test_list[i]
            key['event'] = labels[i]
        test_list = test_keys

        test_dataset = ChildInstituteDataset(test_list)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=config['inference']['batch_size'],
                                     shuffle=False,
                                     pin_memory=True)

        submission = inference(model_path=config.get('general').get('checkpoint'),
                            test_dataloader=test_dataloader)
        submission['series_id'] = submission['series_id'].map(reverse_id_map)
        submission['event'] = np.where(submission['event']==3, 1, submission['event'])
        submission['event'] = np.where(submission['event']==4, 0, submission['event'])

        submission['event'] = submission['event'].rolling(window = 30, min_periods=1, center=True).mean().apply(
            lambda x: 1 if x > 0.5 else (0 if pd.notnull(x) else np.nan)
        )
        submission = reverse_events_by_step(submission, 0, 300)
        submission = reverse_events_by_step(submission, 1, 300)
        ##
        first_rows = submission.groupby('series_id').head(1)
        submission['event_change'] = submission['event'] != submission['event'].shift(1)
        submission.loc[0, 'event_change'] = False

        changed_events = submission[submission['event_change']]
        submission = pd.concat([first_rows, changed_events]).drop_duplicates().sort_values(['series_id', 'step'])
        submission = submission.drop(columns=['event_change'])
        ##
        all_submissions.append(submission)

    if all_submissions:
        final_submission = pd.concat(all_submissions).reset_index(drop=True)
    else:
        final_submission = pd.DataFrame(columns=['row_id', 'series_id', 'step', 'event', 'score'])

    final_submission['event'] = np.where(final_submission['event']==3, 1, final_submission['event'])
    final_submission['event'] = np.where(final_submission['event']==4, 0, final_submission['event'])

    final_submission['event'] = final_submission['event'].map({0: 'onset', 1: 'wakeup'})
    final_submission['score'] = final_submission['score'].astype(float)

    final_submission['score'] = np.where( final_submission['event'] == 'onset',
                                         (final_submission['score']),
                                         (1-final_submission['score']))

    final_submission = final_submission.sort_values(['series_id', 'step']).reset_index(drop=True)
    final_submission['row_id'] = final_submission.index.astype(int)
    final_submission = final_submission[['row_id', 'series_id', 'step', 'event', 'score']]

    final_submission.to_csv('submission.csv', index=False, float_format='%.5f')
    return final_submission


def reverse_events_by_step(submission: pd.DataFrame, event, min_step: int) -> pd.DataFrame:
    # 이벤트가 변경되는 지점을 찾기
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