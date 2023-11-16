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

            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)
            scores = probabilities.detach().cpu().numpy()

            all_series_ids.extend(series_ids)
            all_steps.extend(steps.tolist())
            all_dates.extend(dates.tolist())

            batch_predictions = predictions.cpu().numpy().tolist()
            all_events.extend(batch_predictions)

            # 각 이벤트에 대한 최대 확률 값 추출
            batch_scores = [score[pred] for score, pred in zip(scores, batch_predictions)]
            all_scores.extend(batch_scores)

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
    test_data = test_data.query('series_id=="08db4255286f"')
    test_data['event'] = -1
    test_data = test_data[['series_id', 'timestamp', 'step', 'event', 'anglez', 'enmo']]
    # test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], format='%Y-%m-%d')

    window_size = config.get('inference').get('window_size')
    step_size = config.get('inference').get('step')

    unique_series_ids = test_data['series_id'].unique()
    all_submissions = []
    series_count = 0
    for series_id in unique_series_ids:
        print("Index: ",series_count)
        series_count += 1
        series_data = test_data[test_data['series_id'] == series_id].copy()
        series_data.reset_index(drop=True, inplace=True)

        preprocessed_series_data = preprocess(series_data)

        # test_list, test_list2 = to_list(preprocessed_series_data, window_size, config, step_size)
        test_list = to_list(preprocessed_series_data, window_size, config, step_size)
        test_keys = extract_keys(preprocessed_series_data, window_size, step_size)

        for i, key in enumerate(test_keys):
            key['X'] = test_list[i]
            # key['X1'] = test_list2[i]
        test_list = test_keys

        test_dataset = ChildInstituteDataset(test_list)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=config['inference']['batch_size'],
                                     shuffle=False,
                                     pin_memory=True)

        submission = inference(model_path=config.get('general').get('checkpoint'),
                            test_dataloader=test_dataloader)
        
        # submission['event'] = submission['event'].rolling(window=1000, min_periods=1, center=True).mean().apply(
        #     lambda x: 1 if x > 0.5 else (0 if pd.notnull(x) else np.nan)
        # )
        # submission = reverse_events_if_below_min_count(submission, 1000)
        # submission['event'] = submission['event'].map({0: 'wakeup', 1: 'onset'})

        # ##
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
    final_submission = final_submission[['row_id', 'date', 'series_id','step', 'event', 'score']]

    final_submission.to_csv('submission.csv', index=False, float_format='%.5f')
    return final_submission


def reverse_events_if_below_min_count(submission: pd.DataFrame, min_count: int) -> pd.DataFrame:
    submission['change_point'] = submission['event'].diff().ne(0).astype('int')
    submission.iloc[0, submission.columns.get_loc('change_point')] = 1

    change_points = submission[submission['change_point'] == 1].index.tolist()
    change_points.append(len(submission))

    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1] - 1
        event_duration = end_idx - start_idx + 1
        if event_duration <= min_count:
            submission.loc[start_idx:end_idx, 'event'] = 1 - submission.loc[start_idx:end_idx, 'event']
    
    submission = submission.drop(columns=['change_point'])

    return submission
