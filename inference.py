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
            # series_id를 Tensor로 변환
            series_ids = batch['series_id'].to(device)

            steps = batch['step'].cpu().numpy()
            dates = batch['date'].cpu().numpy()
            # 모델 입력을 딕셔너리 형태로 전달
            outputs = model({'X': inputs, 'series_id': series_ids})

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            scores = probabilities.detach().cpu().numpy()

            all_series_ids.extend(batch['series_id'])  # 원래의 series_id를 사용
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

    with open('./data/id_map.pickle', 'rb') as handle:
        id_map = pickle.load(handle)
    reverse_id_map = {v: k for k, v in id_map.items()}

    # Load preprocessed data for inference
    test_data_path = config.get('general').get('test_data').get('path')

    test_data = pd.read_parquet(test_data_path)
    test_data = test_data.query('series_id=="08db4255286f"')
    test_data['event'] = -1
    test_data = test_data[['series_id', 'timestamp', 'step', 'event', 'anglez', 'enmo']]
    # test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], format='%Y-%m-%d')

    test_data['anglez'] = np.sin(test_data['anglez'])

    window_size = config.get('inference').get('window_size')

    test_data['series_id'] = test_data['series_id'].map(id_map)
    unique_series_ids = test_data['series_id'].unique()
    all_submissions = []

    for series_id in unique_series_ids:
        print('Series_id: ', series_id)
        series_data = test_data[test_data['series_id'] == series_id].copy()
        series_data.reset_index(drop=True, inplace=True)

        preprocessed_series_data = preprocess(series_data)

        test_list = to_list(preprocessed_series_data, window_size, config)
        test_keys = extract_keys(preprocessed_series_data, window_size)

        for i, test_key in enumerate(test_keys):
            test_key['X'] = test_list[i]
        test_list = test_keys

        test_dataset = ChildInstituteDataset(test_list)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=config['inference']['batch_size'],
                                     shuffle=False,
                                     pin_memory=True)

        submission = inference(model_path=config.get('general').get('checkpoint'),
                            test_dataloader=test_dataloader)
        # submission['series_id'] = submission['series_id'].map(reverse_id_map)

        # submission = reverse_events_by_step(submission, 0, 360)
        # submission = reverse_events_by_step(submission, 1, 360)
#         submission['event'] = submission['event'].rolling(window=500, min_periods=1, center=True).mean().apply(
#             lambda x: 1 if x > 0.5 else (0 if pd.notnull(x) else np.nan)
#         )

        # submission['event'] = submission['event'].map({0: 'onset', 1: 'wakeup'})

        #
        # first_rows = submission.groupby('series_id').head(1)
        # submission['event_change'] = submission['event'] != submission['event'].shift(1)
        # submission.loc[0, 'event_change'] = False

        # changed_events = submission[submission['event_change']]
        # submission = pd.concat([first_rows, changed_events]).drop_duplicates().sort_values(['series_id', 'step'])
        # submission = submission.drop(columns=['event_change'])
        ##

        all_submissions.append(submission)
    if all_submissions:
        final_submission = pd.concat(all_submissions).reset_index(drop=True)
    else:
        final_submission = pd.DataFrame(columns=['row_id', 'series_id', 'step', 'event', 'score'])

    final_submission['score'] = final_submission['score'].astype(float)
    #final_submission['score'] = final_submission['score'] / 10
#     final_submission['score'] = np.where( final_submission['event'] == 'onset',
#                                          (final_submission['score']),
#                                          (1-final_submission['score']))

    final_submission = final_submission.sort_values(['series_id', 'step']).reset_index(drop=True)
    final_submission['row_id'] = final_submission.index.astype(int)
    final_submission = final_submission[['row_id', 'series_id', 'date', 'step', 'event', 'score']]

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