# pylint: disable=no-member
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

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

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            scores = probabilities.detach().cpu().numpy()

            all_series_ids.extend(series_ids)
            all_steps.extend(steps.tolist())
            all_dates.extend(dates.tolist())

            batch_predictions = predictions.cpu().numpy().astype(int).tolist()
            for pred in batch_predictions:
                all_events.append(pred[0])

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
    test_data = pd.read_parquet(test_data_path, columns=['series_id', 'timestamp', 'step', 'anglez', 'enmo'])
    test_data['event'] = -1
    test_data = test_data[['series_id', 'timestamp', 'step', 'event', 'anglez', 'enmo']]
    # test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], format='%Y-%m-%d')
    window_size = config.get('inference').get('window_size')
    step_size = config.get('inference').get('step')

    unique_series_ids = test_data['series_id'].unique()
    all_submissions = []

    for series_id in unique_series_ids:
        series_data = test_data[test_data['series_id'] == series_id].copy()
        series_data.reset_index(drop=True, inplace=True)

        preprocessed_series_data = preprocess(series_data)

        test_list = to_list(preprocessed_series_data, window_size, config, step_size)
        test_keys = extract_keys(preprocessed_series_data, window_size, step_size)

        for i, key in enumerate(test_keys):
            key['X'] = test_list[i]

        test_dataset = ChildInstituteDataset(test_keys)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=config['inference']['batch_size'],
                                     shuffle=False,
                                     pin_memory=True)

        submission = inference(model_path=config.get('general').get('checkpoint'),
                            test_dataloader=test_dataloader)

        submission['event'] = submission['event'].rolling(window=7, min_periods=1).mean().apply(
            lambda x: 1 if x > 0.5 else (0 if pd.notnull(x) else np.nan)
        )
        

        # 연속된 이벤트를 탐지
        events = []
        start_idx = None
        min_duration = 12 * 30 ## 30분 
        max_duration = 24 * 60 * 12 ## 1일
        events = []
        start_idx = None
        last_event = None
        for idx, (event, _ ) in enumerate(zip(submission['event'], submission['step'])):
            if last_event is None or last_event != event:
                if start_idx is not None:
                    # 이전 이벤트가 min_duration과 max_duration 사이에 있는 경우에만 기록
                    duration = idx - start_idx
                    if min_duration <= duration <= max_duration:
                        events.append((submission['series_id'][start_idx],
                                       submission['step'][start_idx],
                                       last_event,
                                       submission['score'][start_idx]))
                start_idx = idx
                last_event = event

        # 마지막 연속된 이벤트 처리
        if start_idx is not None and min_duration <= (len(submission) - start_idx) <= max_duration:
            events.append((submission['series_id'][start_idx], submission['step'][start_idx], last_event, submission['score'][start_idx]))

        # 연속된 이벤트 데이터프레임 생성
        events_df = pd.DataFrame(events, columns=['series_id', 'step', 'event', 'score'])
        all_submissions.append(events_df)



    final_submission = pd.concat(all_submissions).reset_index(drop=True)

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