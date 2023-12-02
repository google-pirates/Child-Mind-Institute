# pylint: disable=no-member
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from data import ChildInstituteDataset, preprocess, to_list, extract_keys


def get_predictions(res_df, target, SIGMA):
    q = res_df[target].max() * 0.1
    tmp = res_df.loc[res_df[target] > q].copy()
    tmp['gap'] = tmp['step'].diff()
    tmp = tmp[tmp['gap'] > 5*5]

    res = []
    for i in range(len(tmp) + 1):
        start_i = 0 if i == 0 else tmp['step'].iloc[i-1]
        end_i = tmp['step'].iloc[i] if i < len(tmp) else res_df['step'].max()
        v = res_df.loc[(res_df['step'] > start_i) & (res_df['step'] < end_i)]
        if v[target].max() > q:
            idx = v.idxmax()[target]
            step = v.loc[idx, 'step']
            span = 3*SIGMA
            score = res_df.loc[(res_df['step'] > step - span) & (res_df['step'] < step + span), target].sum()
            res.append([step, target, score])
            
    return res

def inference(model_path: str, test_dataloader: DataLoader) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    total_len = len(test_dataloader.dataset)
    n_classes = 5 
    pred = torch.zeros(total_len, n_classes)

    series_ids_list = []
    steps_list = []

    with torch.no_grad():
        offset = 0
        for batch in tqdm(test_dataloader, desc='inference'):
            inputs, series_ids, steps, dates = batch['X'].to(device), batch['series_id'], batch['step'].cpu().numpy(), batch['date'].cpu().numpy()
            y_pred = model({'X':inputs})
            batch_size = inputs.size(0)
            
            pred[offset:offset + batch_size, :] = y_pred.cpu()
            series_ids_list.extend(series_ids)
            steps_list.extend(steps)
            offset += batch_size
    res_df = pd.DataFrame(torch.softmax(pred, axis=1).numpy(), 
                      columns=['onset', 'wakeup', 'sleep', 'nap', 'notwear'])
    res_df['step'] = steps_list
    res_df['series_id'] = series_ids_list

    ##
    greater_sleep = res_df['sleep'] > res_df['onset']
    res_df.loc[greater_sleep, 'onset'] = res_df.loc[greater_sleep, 'sleep']

    greater_nap = res_df['nap'] > res_df['wakeup']
    res_df.loc[greater_nap, 'wakeup'] = res_df.loc[greater_nap, 'nap']

    non_zero_notwear = res_df['notwear'] != 0
    res_df = res_df[~non_zero_notwear]

    res_df.drop(['sleep', 'nap', 'notwear'], axis=1, inplace=True)
    ##
    return res_df



def main(config) -> pd.DataFrame:
    test_data = pd.read_parquet(config['general']['test_data']['path'])
    window_size, step_size = config['inference']['window_size'], config['inference']['step']

    submissions = []
    sigma = 60
    for series_id in test_data['series_id'].unique():
        series_data = preprocess(test_data[test_data['series_id'] == series_id].reset_index(drop=True))
        if len(series_data) <= 360:
            continue
        
        test_list = to_list(series_data, window_size, config, step_size)
        test_keys = extract_keys(series_data, window_size, step_size)
        test_dataset = ChildInstituteDataset([{'X': test_list[i], **key} for i, key in enumerate(test_keys)])
        
        test_dataloader = DataLoader(test_dataset, batch_size=config['inference']['batch_size'], shuffle=False, pin_memory=True, num_workers= int(os.cpu_count()/2))
        res_df_by_series_id = inference(config['general']['checkpoint'], test_dataloader)

        onset_pred = get_predictions(res_df_by_series_id, target='onset', SIGMA=sigma)
        wakeup_pred = get_predictions(res_df_by_series_id, target='wakeup', SIGMA=sigma)
        
        pred_df = pd.DataFrame(wakeup_pred + onset_pred, columns=['step', 'event', 'score'])
        pred_df = pred_df.sort_values(by='step').drop_duplicates(subset='step').reset_index(drop=True)
        pred_df = pred_df[['row_id', 'step', 'event', 'score']]
        pred_df.sort_values(by=['series_id', 'step'])

        min_step, max_step = res_df_by_series_id.step.min(), res_df_by_series_id.step.max()
        pred_df = pred_df[(pred_df.step > min_step + 12*60) & (pred_df.step < max_step - 12*60)]

        pred_df = pred_df[['row_id', 'step', 'event', 'score']]
        pred_df['score'] = pred_df['score']/100
        pred_df = pred_df.sort_values(by=['step'])

        submissions.append(pred_df)

    return submissions
