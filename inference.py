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
            outputs = model({'X': inputs.permute(0, 2, 1)})

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

# def select_events(series_data, onset_min_step, wakeup_min_step, max_step):
#     selected_events = []
#     series_data = series_data.copy()

#     # 이벤트 변화를 감지하기 위한 표시자 추가
#     series_data['change_point'] = series_data['event'].diff().ne(0).astype('int')
#     series_data.iloc[0, series_data.columns.get_loc('change_point')] = 1  # 첫 번째 행은 항상 변화 지점으로 처리

#     # 모든 변화 지점을 가져옵니다
#     change_points = series_data[series_data['change_point'] == 1].index.tolist()
#     change_points.append(series_data.index[-1] + 1)  # 마지막 인덱스 추가

#     # 각 변화 지점부터 다음 변화 지점까지의 스텝 차이를 기반으로 이벤트 선택
#     for i in range(len(change_points) - 1):
#         start_index = change_points[i]
#         end_index = change_points[i + 1] - 1

#         # 스텝 차이 계산
#         step_difference = series_data.loc[end_index, 'step'] - series_data.loc[start_index, 'step']

#         # 이벤트 유형에 따라 다른 최소 스텝 기준 적용
#         event_type = series_data.loc[start_index, 'event']
#         min_step = onset_min_step if event_type == 0 else wakeup_min_step

#         # 스텝 차이가 min_step 이상, max_step 미만이면 해당 이벤트 그룹을 처리
#         if min_step <= step_difference < max_step:
#             event_group = series_data.iloc[start_index:end_index + 1]
#             # 이벤트 유형에 따라 선택 기준 적용
#             selected_event = event_group.loc[event_group['score'].idxmin()] if event_type == 0 else event_group.loc[event_group['score'].idxmax()]
#             selected_events.append(selected_event)
#         # 스텝 차이가 min_step 미만이면 해당 이벤트들 드랍
#         elif step_difference < min_step:
#             continue

#     return pd.DataFrame(selected_events)

def select_events(series_data, onset_min_step, wakeup_min_step):
    selected_events = []
    series_data = series_data.copy()

    # 이벤트 변화를 감지하기 위한 표시자 추가
    series_data['change_point'] = series_data['event'].diff().ne(0).astype('int')
    series_data.iloc[0, series_data.columns.get_loc('change_point')] = 1  # 첫 번째 행은 항상 변화 지점으로 처리

    # 모든 변화 지점을 가져옵니다
    change_points = series_data[series_data['change_point'] == 1].index.tolist()
    change_points.append(series_data.index[-1] + 1)  # 마지막 인덱스 추가

    # 각 변화 지점부터 다음 변화 지점까지의 스텝 차이를 기반으로 이벤트 선택
    for i in range(len(change_points) - 1):
        start_index = change_points[i]
        end_index = change_points[i + 1] - 1

        # 스텝 차이 계산
        step_difference = series_data.loc[end_index, 'step'] - series_data.loc[start_index, 'step']

        # 이벤트 유형에 따라 다른 최소 스텝 기준 적용
        event_type = series_data.loc[start_index, 'event']
        min_step = onset_min_step if event_type == 0 else wakeup_min_step

        # 스텝 차이가 이벤트 유형별 min_step 이상이면 해당 이벤트 그룹을 처리
        if step_difference >= min_step:
            event_group = series_data.iloc[start_index:end_index + 1]
            # 첫 스텝의 이벤트만 선택
            selected_event = event_group.iloc[0]
            selected_events.append(selected_event)
        # 스텝 차이가 이벤트 유형별 min_step 미만이면 해당 이벤트들 드랍
        else:
            continue

    return pd.DataFrame(selected_events)


def main(config):
    # Load preprocessed data for inference
    test_data_path = config.get('general').get('test_data').get('path')
    
    test_data = pd.read_parquet(test_data_path)
    test_data['event'] = -1
    test_data = test_data[['series_id', 'timestamp', 'step', 'event', 'anglez', 'enmo']]
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], format='%Y-%m-%d')
    
    window_size = config.get('inference').get('window_size')
    step_size = config.get('inference').get('step')

    unique_series_ids = test_data['series_id'].unique()
    all_submissions = []

    ## Train set 내에서 이벤트 간 step 차이의 최소 576(48분), 최대 18228(약 26시간)
    onset_min_step = 500
    wakeup_min_step = 2000
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

    #     submission['event'] = submission['event'].rolling(window=7, min_periods=1).mean().apply(
    #         lambda x: 1 if x > 0.5 else (0 if pd.notnull(x) else np.nan)
    #     )
    #     selected_submission = select_events(submission, onset_min_step, wakeup_min_step)

    #     if not selected_submission.empty:
    #         all_submissions.append(selected_submission)
        all_submissions.append(submission)
    if all_submissions:
        final_submission = pd.concat(all_submissions).reset_index(drop=True)
    else:
        final_submission = pd.DataFrame(columns=['row_id', 'series_id', 'step', 'event', 'score'])

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
