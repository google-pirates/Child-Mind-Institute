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
            series_id = batch['series_id'][0]
            step = batch['step'][-1].item()
            date = batch['date'][-1].item()
            # print("X:", inputs.shape)
            outputs = model({'X': inputs})

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            scores = probabilities.detach().cpu().numpy()

            # events = ['wakeup' if pred == 1 else 'onset' for pred in predictions]

            all_series_ids.extend([series_id] * len(predictions))
            all_steps.extend([step] * len(predictions))
            # all_events.extend(events)
            all_events.append(predictions)
            all_scores.extend(["{:.10f}".format(score.item()) for score in scores])
            all_dates.append(date)

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
    test_data['event'] = 0
    test_data = test_data[['series_id', 'timestamp', 'step', 'event', 'anglez', 'enmo']]
    # test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], format='%Y-%m-%d')
    unique_series_ids = test_data['series_id'].unique()
    all_submissions = []

    for series_id in unique_series_ids:
        series_data = test_data[test_data['series_id'] == series_id].copy()
        series_data.reset_index(drop=True, inplace=True)

        preprocessed_series_data = preprocess(series_data)

        window_size = config.get('inference').get('window_size')
        step_size = config.get('inference').get('step')
        test_list = to_list(preprocessed_series_data, window_size, config, step_size)
        test_keys = extract_keys(preprocessed_series_data, window_size, step_size)

        for i, key in enumerate(test_keys):
            key['X'] = test_list[i]

        test_dataset = ChildInstituteDataset(test_keys)
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=config.get('inference').get('batch_size'),
                                    shuffle=False)

        submission = inference(model_path=config.get('general').get('checkpoint'),
                            test_dataloader=test_dataloader)
        ## rolling
        submission['event'] = submission['event'].transform(
            lambda x: x.rolling(window=7, min_periods=1).aggregate(
                lambda y: y.mode()[0] if not y.mode().empty else np.nan
            )
        )
        ##
        ## event 드랍 시에 onset 이벤트는 처음을 남기고, wakeup 이벤트는 마지막을 남기도록 수정
        # onset_df = submission[submission['event'] == 0.0].drop_duplicates(subset=['date'], keep='first')
        # wakeup_df = submission[submission['event'] == 1.0].drop_duplicates(subset=['date'], keep='last')
        # droped_submission = pd.concat([onset_df, wakeup_df]).sort_values(['date', 'event'])
        # droped_submission['event'] = droped_submission['event'].map({0.0: 'onset', 1.0: 'wakeup'})
        # all_submissions.append(droped_submission)
        ##
        submission = submission.drop_duplicates(subset=['date', 'event'], keep='first')
        submission['event'] = submission['event'].map({0.0: 'onset', 1.0: 'wakeup'})
        all_submissions.append(submission)

    final_submission = pd.concat(all_submissions).reset_index(drop=True)
    final_submission['score'] = final_submission['score'].astype(float)


    ## 이벤트가 onset 인 경우 스코어 반전
    final_submission['score'] = np.where( final_submission['event'] == 'onset', 
                                         1 - final_submission['score'], 
                                         final_submission['score'])
    
    final_submission = final_submission.sort_values(['series_id', 'step']).reset_index(drop=True)
    final_submission['row_id'] = final_submission.index.astype(int)
    final_submission = final_submission[['row_id', 'series_id', 'step', 'event', 'score']]

    final_submission.to_csv('submission_rolling_tsmixer_onset_1st_wakeup_last_testing2.csv', index=False)
    return final_submission

