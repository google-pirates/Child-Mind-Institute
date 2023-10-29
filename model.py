import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, num_features, norm_type, dropout_rate, ff_dim, activation):
        super(ResBlock, self).__init__()

        # Choose normalization using getattr
        self.norm = getattr(nn, norm_type, nn.BatchNorm1d)(num_features)

        # Choose activation using getattr
        self.activation = getattr(nn, activation, nn.GELU)()

        # Temporal Linear
        self.temporal_norm = self.norm
        self.temporal_linear = nn.Linear(num_features, num_features)
        self.temporal_dropout = nn.Dropout(dropout_rate)

        # Feature Linear
        self.feature_norm = self.norm
        self.feature_linear1 = nn.Linear(num_features, ff_dim)
        self.feature_linear2 = nn.Linear(ff_dim, num_features)
        self.feature_dropout = nn.Dropout(dropout_rate)

    def time_mixer(self, x):
        """Temporal mixing function."""
        x_norm = self.temporal_norm(x)  # batch, seq_len, n_features
        x_transposed = x_norm.transpose(1, 2)  # batch, n_features, seq_len
        x_linear = self.temporal_linear(x_transposed)  # batch, n_features, seq_len
        x_dropout = self.temporal_dropout(x_linear.transpose(1, 2))  # batch, seq_len, n_features
        x_res = x_dropout + x  # batch, seq_len, n_features
        return x_res

    def feature_mixer(self, x):
        """Feature mixing function."""
        x_norm = self.feature_norm(x)  # batch, seq_len, n_features
        x_linear1 = self.feature_linear1(x_norm)  # batch, seq_len, ff_dim
        x_activation = self.activation(x_linear1)  # batch, seq_len, ff_dim
        x_dropout1 = self.feature_dropout(x_activation)  # batch, seq_len, ff_dim
        x_linear2 = self.feature_linear2(x_dropout1)  # batch, seq_len, n_features
        x_dropout2 = self.feature_dropout(x_linear2)  # batch, seq_len, n_features
        x_res = x_dropout2 + x  # batch, seq_len, n_features
        return x_res

    def forward(self, x):
        x = self.time_mixer(x)
        x = self.feature_mixer(x)
        return x


class TSMixer(nn.Module):
    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout_rate, ff_dim, target_slice=None):
        super(TSMixer, self).__init__()

        if n_block <= 0:
            raise ValueError("n_block should be greater than 0.")

        self.in_channels = input_shape[1]
        self.blocks = nn.ModuleList(
            [ResBlock(self.in_channels, norm_type, dropout_rate, ff_dim, activation) 
             for _ in range(n_block)]
        )
        self.target_slice = target_slice
        self.pred_len = pred_len
        self.final_dense = nn.Linear(self.in_channels, pred_len)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.target_slice:
            x = x[:, :, self.target_slice]

        x_transposed = x.transpose(1, 2)  # batch, n_features, seq_len
        x_out = self.final_dense(x_transposed)
        x_out = x_out.transpose(1, 2)  # batch, seq_len, n_features
        return x_out