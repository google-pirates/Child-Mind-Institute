import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, activation, dropout, ff_dim):
        super(ResBlock, self).__init__()

        # Choose normalization
        self.norm = nn.LayerNorm if norm_type == 'L' else nn.BatchNorm1d

        # Temporal Linear
        self.temporal_norm = self.norm([in_channels, out_channels])
        self.temporal_linear = nn.Linear(in_channels, out_channels)
        self.temporal_dropout = nn.Dropout(dropout)

        # Feature Linear
        self.feature_norm = self.norm([in_channels, out_channels])
        self.feature_linear1 = nn.Linear(out_channels, ff_dim)
        self.feature_linear2 = nn.Linear(ff_dim, out_channels)
        self.feature_dropout1 = nn.Dropout(dropout)
        self.feature_dropout2 = nn.Dropout(dropout)

        # Activation
        self.activation = activation

    def time_mixer(self, x):
        x_norm = self.temporal_norm(x)
        x_transposed = x_norm.transpose(1, 2)
        x_linear = self.temporal_linear(x_transposed)
        x_res = x_linear.transpose(1, 2) + x
        return self.temporal_dropout(x_res)

    def feature_mixer(self, x):
        x_norm = self.feature_norm(x)
        x_linear = self.feature_linear1(x_norm)
        x_linear = self.activation(x_linear)
        x_linear = self.feature_dropout1(x_linear)

        x_linear = self.feature_linear2(x_linear)
        x_linear = self.feature_dropout2(x_linear)
        return x_linear + x    

    def forward(self, x):
        x = self.time_mixer(x)
        x = self.feature_mixer(x)
        return x


class TSMixer(nn.Module):
    def __init__(self, input_shape, pred_len, norm_type,
                 activation, n_block, dropout, ff_dim, target_slice=None):
        super(TSMixer, self).__init__()

        self.in_channels, self.out_channels = input_shape[1], input_shape[2]
        self.blocks = nn.ModuleList([ResBlock(self.in_channels,
                                              self.out_channels,
                                              norm_type,
                                              activation,
                                              dropout, ff_dim) for _ in range(n_block)])
        self.target_slice = target_slice
        self.pred_len = pred_len
        self.final_dense = nn.Linear(self.in_channels, pred_len)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.target_slice:
            x = x[:, :, self.target_slice]

        x_transposed = x.transpose(1, 2)
        x_out = self.final_dense(x_transposed)
        return x_out.transpose(1, 2)
