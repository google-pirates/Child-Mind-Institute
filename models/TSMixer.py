from typing import Dict, Any
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super(ResBlock, self).__init__()

        self.model_config = config.get('model').get('tsmixer')
        self.n_features = config.get('train').get('n_features')
        self.seq_len = config.get('train').get('seq_len')
        self.norm_type = self.model_config.get('normalization')
        self.dropout_rates = self.model_config.get('dropout_rates')
        self.ff_dim = self.model_config.get('ff_dim')
        self.activation_name = self.model_config.get('activation')

        # Choose normalization using getattr
        self.norm = getattr(nn, self.norm_type)(self.n_features)

        # Choose activation using getattr
        self.activation = getattr(nn, self.activation_name)()

        # Temporal Linear
        self.temporal_norm = self.norm
        self.temporal_linear = nn.Linear(self.seq_len, self.seq_len)
        self.temporal_dropout = nn.Dropout(self.dropout_rates)

        # Feature Linear
        self.feature_nurm = self.norm
        self.feature_linear1 = nn.Linear(self.n_features, self.ff_dim)
        self.feature_linear2 = nn.Linear(self.ff_dim, self.n_features)
        self.feature_dropout = nn.Dropout(self.dropout_rates)

    def time_mixer(self, x):
        """Temporal mixing function."""
        x_norm0 = self.temporal_norm(x)  # batch, seq_len, n_features
        x_transposed = x_norm0.transpose(1, 2)  # batch, n_features, seq_len
        x_linear0 = self.temporal_linear(x_transposed)  # batch, n_features, seq_len
        x_dropout0 = self.temporal_dropout(x_linear0.transpose(1, 2))  # batch, seq_len, n_features
        x_res0 = x_dropout0 + x  # batch, seq_len, n_features
        return x_res0

    def feature_mixer(self, x):
        """Feature mixing function."""
        x_norm1 = self.feature_nurm(x)  # batch, seq_len, n_features
        x_linear1 = self.feature_linear1(x_norm1)  # batch, seq_len, ff_dim
        x_activation = self.activation(x_linear1)  # batch, seq_len, ff_dim
        x_dropout1 = self.feature_dropout(x_activation)  # batch, seq_len, ff_dim
        x_linear2 = self.feature_linear2(x_dropout1)  # batch, seq_len, n_features
        x_dropout2 = self.feature_dropout(x_linear2)  # batch, seq_len, n_features
        x_res1 = x_dropout2 + x  # batch, seq_len, n_features
        return x_res1

    def forward(self, x):
        x = self.time_mixer(x)
        x = self.feature_mixer(x)
        return x


class TSMixer(nn.Module):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super(TSMixer, self).__init__()

        self.model_config = config.get('model').get('tsmixer')
        self.n_features = config.get('train').get('n_features')
        self.seq_len = config.get('train').get('seq_len')
        self.batch_size = config.get('train').get('batch_size')

        self.out_seq_len = self.model_config.get('out_seq_len')
        self.n_block = self.model_config.get('n_block')
        self.target_slice = self.model_config.get('target_slice', None)

        if self.n_block <= 0:
            raise ValueError("n_block should be greater than 0.")

        self.blocks = nn.ModuleList([ResBlock(config) for _ in range(self.n_block)])

        self.final_dense = nn.Linear(self.n_features, self.out_seq_len)


    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch.get('X')  # batch, seq_len, n_features
        if x is None:
            raise ValueError("Input 'X' is not found in the batch") ## TorchScript에서 Optional[Tensor] 를 트래킹하기 위함

        for block in self.blocks:
            x = block(x)
        
        if self.target_slice is not None:
            x = x[:, :, self.target_slice]

        # x_transposed = x.transpose(1, 2)  # batch, n_features, seq_len
        # x_out = self.final_dense(x_transposed)
        # x_out = x_out.transpose(1, 2)  # batch, seq_len, n_features
        x_selected = x[:, -1, :] ## 마지막 시간 단계를 이용하여 출력 생성
        return self.final_dense(x_selected)