import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import os, glob

from tqdm import tqdm 
from collections import OrderedDict

from math import pi, sqrt, exp
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

plt.style.use("ggplot")

NOTDEBUG = True # False -> DEBUG, True -> normally train
WORKERS = os.cpu_count() // 4
N_FOLDS = 5
TRAIN_FOLD = 0

SIGMA = 720 #average length of day is 24*60*12 = 17280 for comparison
SAMPLE_FREQ = 12 # 1 obs per minute

MAX_LEN = (24*60*12) # 1day per chunk
MAX_CHUNK_LEN = MAX_LEN // SAMPLE_FREQ
USE_AMP = False
SEED = 8620

# Optimizer config
LR = 4e-4
WD = 1e-2
WARMUP_PROP = 0.1

# Train config
EPOCHS = 10 if NOTDEBUG else 2
BS = 4
MAX_GRAD_NORM = 2.
GRAD_ACC = 32 // BS

device = 'cuda' if torch.cuda.is_available() else 'cpu'


skf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
metadata = pd.read_csv('../data/train_events.csv')
unique_ids = metadata['series_id'].unique()
meta_cts = pd.DataFrame(unique_ids, columns=['series_id'])
for i, (train_index, valid_index) in enumerate(skf.split(X=meta_cts['series_id'], y=[1]*len(meta_cts))):
    if i != TRAIN_FOLD:
        continue
    print(f"Fold = {i}")
    train_ids = meta_cts.loc[train_index, 'series_id']
    valid_ids = meta_cts.loc[valid_index, 'series_id']
    print(f"Length of Train = {len(train_ids)}, Length of Valid = {len(valid_ids)}")
    
    if i == TRAIN_FOLD:
        break
        
train_fpaths = [f"./train_csvs/{_id}.csv" for _id in train_ids]
valid_fpaths = [f"./train_csvs/{_id}.csv" for _id in valid_ids]
train_ids[:5], train_fpaths[:5] ,len(train_fpaths)

class SleepDataset(Dataset):
    def __init__(
        self,
        folder,
        max_len=17280,
        is_train=False,
        sample_per_epoch=10000
    ):
        self.enmo_mean = np.load('/enmo_mean.npy')
        self.enmo_std = np.load('/enmo_std.npy')
        
        self.max_len = max_len
        assert max_len % SAMPLE_FREQ == 0
        
        self.is_train = is_train
        
        self.max_df_size = 0
        self.min_df_size = 1e9
        
        self.sample_per_epoch = sample_per_epoch
        
        self.feat_list = ['anglez','enmo']
        
        self.Xys = self.read_csvs(folder)        
        
        self.label_list = ['onset', 'wakeup']
        
        self.hour_feat= ['hour']
        
        self.compress_methods = ['mean', 'median', 'fixed']
        
    def read_csvs(self, folder):
        res = []
        if type(folder) is str:
            files = glob.glob(f'{folder}/*.csv')
        else:
            files = folder
        for i, f in tqdm(enumerate(files), total=len(files), leave=False):
            df = pd.read_csv(f)
            df = self.norm_feat_eng(df, init=True if i==0 else False)
                
            res.append(df)
            self.max_df_size = max(self.max_df_size, len(df))
            self.min_df_size = min(self.min_df_size, len(df))
        return res
    
    def compress(self, xt, method='mean', sample_freq=SAMPLE_FREQ):
        x, t = xt
        seq_len = x.shape[0]
        if method == 'mean':
            x = x.reshape(seq_len//sample_freq, sample_freq, -1).mean(1)
            t = t.reshape(seq_len//sample_freq, sample_freq, 1).mean(1)
            
        if method == 'median':
            x = np.median(x.reshape(seq_len//sample_freq, sample_freq, -1), axis=1)
            t = np.median(t.reshape(seq_len//sample_freq, sample_freq, 1), axis=1)
            
        if method == 'fixed':
            start = torch.randint(0, sample_freq, size=(1,))[0].numpy()
            x = x[start::sample_freq]
            t = t[start::sample_freq]
        
        return x.astype(np.float32), t.astype(np.int32)

    def norm_feat_eng(self, X, init=False):
        X['anglez'] = X['anglez'] / 90.0
        X['enmo'] = (X['enmo'] - self.enmo_mean) / (self.enmo_std + 1e-12)
        
        for w in [1, 2, 4, 8, 16]:    
            X['anglez_shift_pos_' + str(w)] = X['anglez'].shift(w).fillna(0)
            X['anglez_shift_neg_' + str(w)] = X['anglez'].shift(-w).fillna(0)
            
            X['enmo_shift_pos_' + str(w)] = X['enmo'].shift(w).fillna(0)
            X['enmo_shift_neg_' + str(w)] = X['enmo'].shift(-w).fillna(0)
            
            if init:
                self.feat_list.append('anglez_shift_pos_' + str(w))
                self.feat_list.append('anglez_shift_neg_' + str(w))
                
                self.feat_list.append('enmo_shift_pos_' + str(w))
                self.feat_list.append('enmo_shift_neg_' + str(w))
            
        for r in [17, 33, 65]:
            tmp_anglez = X['anglez'].rolling(r, center=True)
            X[f'anglez_mean_{r}'] = tmp_anglez.mean()
            X[f'anglez_std_{r}'] = tmp_anglez.std()            
            
            tmp_enmo = X['enmo'].rolling(r, center=True)
            X[f'enmo_mean_{r}'] = tmp_enmo.mean()
            X[f'enmo_std_{r}'] = tmp_enmo.std()
            
            if init:
                self.feat_list.append(f'anglez_mean_{r}')
                self.feat_list.append(f'anglez_std_{r}')

                self.feat_list.append(f'enmo_mean_{r}')
                self.feat_list.append(f'enmo_std_{r}')
                
        X = X.fillna(0)
        
        return X.astype(np.float32)

    def gauss(self, n=SIGMA, sigma=SIGMA*0.15):
        # guassian distribution function
        r = range(-int(n/2),int(n/2)+1)
        return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
    
    def __len__(self):
        return self.sample_per_epoch if self.is_train else len(self.Xys)

    def __getitem__(self, index):
        if self.is_train:
            ind = torch.randint(0, len(self.Xys), size=(1,))[0].numpy()
            Xy = self.Xys[ind]
            
            X = Xy[self.feat_list].values.astype(np.float32)
            y = Xy[self.label_list].values.astype(np.float32)
            t = Xy[self.hour_feat].values.astype(np.int32)

            if len(Xy)+1<self.max_len:
                res = self.max_len - len(Xy) + 1
                X = np.pad(X, ((0, res), (0, 0)))
                y = np.pad(y, ((0, res), (0, 0)))
                t = np.pad(t, ((0, res), (0, 0)))

            start = torch.randint(0, len(X)-self.max_len, size=(1,))[0].numpy()

            X = X[start:start+self.max_len]
            y = y[start:start+self.max_len]    
            t = t[start:start+self.max_len]    
            
            method_idx = torch.randint(0, len(self.compress_methods), size=(1,))[0].numpy()
            X, t = self.compress((X, t), method=self.compress_methods[method_idx])

        else:
            Xy = self.Xys[index]
            X = Xy[self.feat_list].values.astype(np.float32)
            y = Xy[self.label_list].values.astype(np.float32)        
            t = Xy[self.hour_feat].values.astype(np.int32)
            
            if len(Xy)%SAMPLE_FREQ!=0:
                res = SAMPLE_FREQ - (len(Xy)%SAMPLE_FREQ)
                X = np.pad(X, ((0, res), (0, 0)))
                y = np.pad(y, ((0, res), (0, 0)))
                t = np.pad(t, ((0, res), (0, 0)))
            
            X, t = self.compress((X, t), method='mean')
            
        return X, t, y

train_fpaths = train_fpaths if NOTDEBUG else train_fpaths[:25]
valid_fpaths = valid_fpaths if NOTDEBUG else valid_fpaths[:5]
sample_per_epoch = 20_000 if NOTDEBUG else 100

train_ds = SleepDataset(train_fpaths, max_len=MAX_LEN, is_train=True, sample_per_epoch=sample_per_epoch)
val_ds = SleepDataset(valid_fpaths, is_train=False)

train_dl = DataLoader(
    train_ds,
    batch_size=BS,
    pin_memory=True,
    num_workers=WORKERS,
    shuffle=False,
    drop_last=True
)
val_dl = DataLoader(
    val_ds,
    batch_size=1,
    pin_memory=True,
    num_workers=WORKERS,
    shuffle=False,
    drop_last=False
)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss.mean()

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or np.any([v in name.lower()  for v in skip_list]):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def padding_(x, tgt_len=MAX_CHUNK_LEN):
    res = tgt_len - (x.size(-2) % tgt_len)
    x = F.pad(x, (0, 0, 0, res))
    return x

class LSTM(nn.Module):
    def __init__(self, in_channel, seq_len, hidden, out_channels, dr= [0.4]):
        super(LSTM, self).__init__()
        self.in_c = in_channel
        self.seq_len = seq_len
        self.out_c = hidden
        
        num_lstm_layers = len(self.out_c)
        self.dropout_rates = dr * num_lstm_layers
        
        self.fc_outputs = out_channels
        num_fc_layers = len(self.fc_outputs)
        self.fc_dropout_rates = dr * num_fc_layers
        
        ## CNN
        in_feature = self.in_c
        self.lstm_blocks = []
        for (out_feature, d_r) in zip(self.out_c, self.dropout_rates):
            self.lstm_blocks.append(nn.LSTM(input_size=in_feature, hidden_size=out_feature, batch_first=True, bidirectional=True))
            self.lstm_blocks.append(nn.LayerNorm(out_feature*2))
            self.lstm_blocks.append(nn.ReLU())
            self.lstm_blocks.append(nn.Dropout(p=d_r))
            
            in_feature = out_feature*2
        print(self.lstm_blocks)
        ## FC
        fc_input = self.out_c[-1] * self.seq_len * 2
        fc_layers = []
        for (i, (fc_output, fc_d_r)) in enumerate(zip(self.fc_outputs, self.fc_dropout_rates), 1):
            fc_layers.append(nn.Linear(in_features=fc_input, out_features=fc_output))
            if i < num_fc_layers:
                fc_layers.append(nn.BatchNorm1d(fc_output))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(p=fc_d_r))
                fc_input = fc_output
        
        self.fc_layers = nn.Sequential(*fc_layers)
        print(self.fc_layers)
        ## add hour embedding
        self.fc_in = nn.Linear(in_channel, 26)
        self.hr_emb = nn.Embedding(24, 8)
        ##
#         self.dense = nn.Linear()
    def forward(self, x, t):
        x = self.fc_in(x)
        t = self.hr_emb(t)
        x = torch.cat([x, t.squeeze(2)], dim=-1) 
        
        for block in self.lstm_blocks:
            x = block(x)
            if isinstance(x, tuple):
                x, _ = x

        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x
            

target_length = int(BS * MAX_LEN * 2)
output_length = int(138240/BS)
print(output_length)
model = LSTM(in_channel = len(train_ds.feat_list),
             seq_len = 1440,
             hidden = [50, 100], 
             out_channels = [1000, int(output_length/2) ,output_length] # BS = Batchsize
             )


optimizer_parameters = add_weight_decay(model, weight_decay=WD, skip_list=['bias'])
optimizer = AdamW(optimizer_parameters, lr=LR, eps=1e-6, betas=(0.9, 0.999))

steps = len(train_dl) * EPOCHS // GRAD_ACC
warmup_steps = int(steps * WARMUP_PROP)

print(steps, warmup_steps)

scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=steps,
                                            num_cycles=0.5)

dt = time.time()

model_path = "./"

os.makedirs(model_path, exist_ok=True)



history = {
    "train_loss": [],
    "valid_loss": [],
    "lr": [],
}

best_valid_loss = 1e5

criterion = FocalLoss(alpha=1., gamma=2.)


autocast = torch.cuda.amp.autocast(enabled=USE_AMP)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

es_step = 0

model.to(device)
for epoch in range(EPOCHS):
    if epoch>=7: break
    total_loss = 0.0
    model.train()
    optimizer.zero_grad()
    with tqdm(train_dl, leave=True) as pbar:
        for step, (X_batch, hr_batch, y_batch) in enumerate(pbar):
            X_batch = X_batch.to(device)
            hr_batch = hr_batch.to(device)
            y_batch = y_batch.to(device)           

            with autocast:
                pred = model(X_batch, hr_batch)
                loss = criterion(pred, y_batch)
                
                if torch.isnan(loss).any():
                    raise RuntimeError('Detected NaN.')
    
                total_loss += loss.item()
                if GRAD_ACC > 1:
                    loss = loss / GRAD_ACC
                        
                pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item()*GRAD_ACC:.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
        
                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)
    
            if (step + 1) % GRAD_ACC == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

        train_loss = total_loss/len(train_dl)
    

    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        with tqdm(val_dl, leave=True) as pbar:
            for step, (X_batch, hr_batch, y_batch) in enumerate(pbar):
                X_batch = X_batch.to(device)
                hr_batch = hr_batch.to(device)
                y_batch = y_batch.to(device)
                pred = torch.zeros_like(y_batch)
                x_seq_len = X_batch.shape[1]
                y_seq_len = y_batch.shape[1]

                if x_seq_len%MAX_CHUNK_LEN != 0:
                    X_batch = padding_(X_batch,MAX_CHUNK_LEN)
                    hr_batch = padding_(hr_batch,MAX_CHUNK_LEN)

                if y_seq_len%(MAX_CHUNK_LEN*SAMPLE_FREQ) != 0:
                    pred = padding_(pred,MAX_CHUNK_LEN*SAMPLE_FREQ)

                j = 0
                for i in range(0, x_seq_len, MAX_CHUNK_LEN):
                    X_chunk = X_batch[:, i : i + MAX_CHUNK_LEN]
                    hr_chunk = hr_batch[:, i : i + MAX_CHUNK_LEN]
                    with autocast:
                        p = model(X_chunk, hr_chunk)
                        pred[:, j : j + MAX_CHUNK_LEN*SAMPLE_FREQ] = p

                    j += MAX_CHUNK_LEN*SAMPLE_FREQ

                pred = pred[:, :y_seq_len]
                loss = criterion(pred, y_batch)
                
                total_loss += loss.item()
            
    valid_loss = total_loss/len(val_dl)
    
    history["train_loss"].append(train_loss)
    history["valid_loss"].append(valid_loss)
    history["lr"].append(optimizer.param_groups[0]["lr"])

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(
            model.state_dict(),
            os.path.join(model_path, f"model_best_fold-{TRAIN_FOLD}.pth"),
        )
        es_step = 0
        
    else:
        es_step += 1
        if es_step >= 2:
            break

    dt = time.time() - dt
    print(
        f"{epoch+1}/{EPOCHS} -- ",
        f"train_loss = {train_loss:.6f} -- ",
        f"valid_loss = {valid_loss:.6f} -- ",
        f"time = {dt:.6f}s",
    )
    dt = time.time()

history_path = os.path.join(model_path, "history.json")
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=4)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

