# pylint: disable=no-member
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from sklearn.metrics import mean_squared_error
from scipy.signal import argrelmin, argrelmax
from scipy import signal
from scipy.signal import filtfilt, butter

from data import ChildInstituteDataset, preprocess, to_list, extract_keys

def lpf(wave, fs=12*60*24, fe=60, n=3):
    if len(wave) >= 6:
        nyq = fs / 2.0
        b, a = butter(1, fe/nyq, btype='low')
        wave = wave.ravel()
        for i in range(n):
            wave = filtfilt(b, a, wave)
    return wave

def inference(model_path: str, test_dataloader: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    all_series_ids, all_steps, all_events, all_scores, all_dates = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Inference'):
            inputs, series_ids = batch['X'].to(device), batch['series_id']
            steps, dates = batch['step'].cpu().numpy(), batch['date'].cpu().numpy()
            outputs = model({'X': inputs}).view(-1)

            scores = torch.sigmoid(outputs).detach().cpu().numpy()
            if scores.ndim > 0 and scores.size > 0:
                before_RMSE = np.sqrt(mean_squared_error(scores, np.zeros_like(scores)))
                filtered_scores = lpf(scores)  # lpf 함수를 호출하기 전에 길이 검사를 수행합니다.
                after_RMSE = np.sqrt(mean_squared_error(filtered_scores, np.zeros_like(filtered_scores)))
                decay_ratio = before_RMSE / after_RMSE if after_RMSE else 1
                filtered_scores *= decay_ratio
            else:
                # scores가 적절한 형상이 아닐 경우 RMSE 계산을 건너뜁니다.
                filtered_scores = scores

            onset_candi = argrelmin(filtered_scores, order=12*60*6)[0]
            wakeup_candi = argrelmax(filtered_scores, order=12*60*6)[0]
            
            for idx in onset_candi:
                if filtered_scores[idx] < 0.5:
                    all_series_ids.append(series_ids[idx])
                    all_steps.append(steps[idx])
                    all_events.append('onset')
                    all_scores.append(filtered_scores[idx])
                    all_dates.append(dates[idx])

            for idx in wakeup_candi:
                if filtered_scores[idx] >= 0.5:
                    all_series_ids.append(series_ids[idx])
                    all_steps.append(steps[idx])
                    all_events.append('wakeup')
                    all_scores.append(filtered_scores[idx])
                    all_dates.append(dates[idx])

    return pd.DataFrame({
        'series_id': all_series_ids,
        'step': all_steps,
        'date': all_dates,
        'event': all_events,
        'score': all_scores
    })

def main(config):
    test_data_path = config.get('general').get('test_data').get('path')
    test_data = pd.read_parquet(test_data_path)
    test_data['event'] = -1
    test_data = test_data[['series_id', 'timestamp', 'step', 'event', 'anglez', 'enmo']]

    window_size, step_size = config.get('inference').get('window_size'), config.get('inference').get('step')
    all_submissions = []

    for series_id in test_data['series_id'].unique():
        series_data = test_data[test_data['series_id'] == series_id].copy()
        series_data.reset_index(drop=True, inplace=True)
        preprocessed_series_data = preprocess(series_data)
        test_list, test_keys = to_list(preprocessed_series_data, window_size, config, step_size), extract_keys(preprocessed_series_data, window_size, step_size)
        
        for key, item in zip(test_keys, test_list):
            key['X'] = item

        submission = inference(config.get('general').get('checkpoint'), DataLoader(ChildInstituteDataset(test_keys), batch_size=config['inference']['batch_size'], shuffle=False, pin_memory=True))
        all_submissions.append(submission)

    final_submission = pd.concat(all_submissions).reset_index(drop=True)
    final_submission['score'] = final_submission['score'].astype(float)
    final_submission = final_submission.sort_values(['series_id', 'step']).reset_index(drop=True)
    final_submission['row_id'] = final_submission.index.astype(int)
    final_submission = final_submission[['row_id', 'series_id', 'step', 'event', 'score']]
    final_submission.to_csv('submission.csv', index=False)
    
    return final_submission