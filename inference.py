# pylint: disable=no-member
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ChildInstituteDataset, preprocess, to_list, extract_keys


# def inference(model_path: str, test_dataloader: DataLoader):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = torch.jit.load(model_path, map_location=device)
#     model.to(device)
#     model.eval()

#     all_series_ids = []
#     all_steps = []
#     all_events = []
#     all_scores = []

#     with torch.no_grad():
#         for batch in tqdm(test_dataloader, desc='Inference'):
#             inputs = batch['X'].to(device)
#             series_id = batch['series_id'][0].item()
#             # series_id = id_map[batch['series_id'][-1].item()]
#             step = batch['step'][-1].item()

#             outputs = model({'X': inputs})

#             probabilities = torch.sigmoid(outputs)
#             predictions = (probabilities > 0.5).float()
#             scores = probabilities.detach().cpu().numpy()

#             events = ['wakeup' if pred == 1 else 'onset' for pred in predictions]

#             all_series_ids.extend([series_id] * len(predictions))
#             all_steps.extend([step] * len(predictions))
#             all_events.extend(events)
#             all_scores.extend(["{:.10f}".format(score.item()) for score in scores])

#     submission = pd.DataFrame({
#         'series_id': all_series_ids,
#         'step': all_steps,
#         'event': all_events,
#         'score': all_scores
#     })

#     return submission


# def main(config):
#     # Load preprocessed data for inference
#     test_data_path = config.get('general').get('test_data').get('path')

#     test_data = pd.read_parquet(test_data_path, columns=['series_id', 'timestamp','step', 'anglez', 'enmo'])

#     ## test data를 train data와 동일한 형식으로 변경
#     test_data['event'] = False

#     data = test_data[['series_id', 'timestamp', 'step', 'event', 'anglez', 'enmo']]

#     data['series_id'], unique_series = pd.factorize(data['series_id'])

#     ## series_id map
#     id_map = {idx: id_ for idx, id_ in enumerate(unique_series)}

#     preprocessed_data = preprocess(data)

#     window_size = config.get('inference').get('window_size')
#     step = config.get('inference').get('step')

#     test_list = to_list(preprocessed_data, window_size, config, step)
#     test_keys = extract_keys(preprocessed_data, window_size, step)

#     for i, test_key in enumerate(test_keys):
#         test_key['X'] = test_list[i]
#     test_list = test_keys

#     test_dataset = ChildInstituteDataset(test_list)

#     batch_size = config.get('inference').get('batch_size')
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     submission = inference(model_path=config.get('general').get('checkpoint'), test_dataloader=test_dataloader)
#     submission['score'] = submission['score'].astype(float)
#     submission['series_id'] = submission['series_id'].map(id_map)

#     # submission = submission.drop_duplicates(subset=['series_id', 'event'], keep='first')
#     submission = submission.sort_values(['series_id','step']).reset_index(drop=True)
#     submission['row_id'] = submission.index.astype(int)
#     submission = submission[['row_id','series_id','step','event','score']]
#     submission.to_csv('submission3.csv',index=False)
#     return submission


## Groupby series_id

def inference(model_path: str, test_dataloader: DataLoader):
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
            series_id = batch['series_id'][0]
            step = batch['step'][-1].item()

            outputs = model({'X': inputs})

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            scores = probabilities.detach().cpu().numpy()

            events = ['wakeup' if pred == 1 else 'onset' for pred in predictions]

            all_series_ids.extend([series_id] * len(predictions))
            all_steps.extend([step] * len(predictions))
            all_events.extend(events)
            all_scores.extend(["{:.10f}".format(score.item()) for score in scores])

    submission = pd.DataFrame({
        'series_id': all_series_ids,
        'step': all_steps,
        'event': all_events,
        'score': all_scores
    })

    return submission

def main(config):
    # Load preprocessed data for inference
    test_data_path = config.get('general').get('test_data').get('path')
    test_data = pd.read_parquet(test_data_path, columns=['series_id', 'timestamp', 'step', 'anglez', 'enmo'])
    test_data['event'] = 0  # Add the 'event' column with False as placeholder

    # Get unique series_ids
    unique_series_ids = test_data['series_id'].unique()

    all_submissions = []  # To collect submissions for each series_id

    # Iterate over each unique series_id
    for series_id in unique_series_ids:
        # Filter data for the current series_id
        series_data = test_data[test_data['series_id'] == series_id].copy()
        series_data.reset_index(drop=True, inplace=True)
        # print('data: ', series_data)
        # Preprocess the series_data
        preprocessed_series_data = preprocess(series_data)

        # Generate the test list and keys for the DataLoader
        window_size = config.get('inference').get('window_size')
        step_size = config.get('inference').get('step')
        test_list = to_list(preprocessed_series_data, window_size, config, step_size)
        test_keys = extract_keys(preprocessed_series_data, window_size, step_size)

        # Append 'X' to test_keys
        for i, key in enumerate(test_keys):
            key['X'] = test_list[i]

        # Create DataLoader for the current series_id
        test_dataset = ChildInstituteDataset(test_keys)
        test_dataloader = DataLoader(test_dataset, batch_size=config.get('inference').get('batch_size'), shuffle=False)

        # Perform inference and collect the submission
        submission = inference(model_path=config.get('general').get('checkpoint'), test_dataloader=test_dataloader)
        all_submissions.append(submission)

    # Concatenate all submissions into one DataFrame
    final_submission = pd.concat(all_submissions).reset_index(drop=True)

    # Convert scores to float and map series_id back to original
    final_submission['score'] = final_submission['score'].astype(float)

    # Sort and reset index
    final_submission = final_submission.sort_values(['series_id', 'step']).reset_index(drop=True)
    final_submission['row_id'] = final_submission.index.astype(int)
    final_submission = final_submission[['row_id', 'series_id', 'step', 'event', 'score']]

    # Save to CSV
    final_submission.to_csv('submission.csv', index=False)

    return final_submission