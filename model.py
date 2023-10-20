import torch.nn as nn

# Define activation functions dictionary
ACTIVATIONS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh()
}

class ResBlock(nn.Module):
    def __init__(self, in_channels, norm_type, dropout, ff_dim, activation="gelu"):
        super(ResBlock, self).__init__()

        # Choose normalization
        if norm_type == 'L':
            self.norm = nn.LayerNorm(in_channels)
        elif norm_type == 'B':
            self.norm = nn.BatchNorm1d(in_channels)
        else:
            raise ValueError(
                "Invalid norm_type. Choose either 'L' for LayerNorm or 'B' for BatchNorm1d."
            )

        # Choose activation
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Invalid activation type. Available options are: {list(ACTIVATIONS.keys())}"
            )
        self.activation = ACTIVATIONS[activation]

        # Temporal Linear
        self.temporal_norm = self.norm
        self.temporal_linear = nn.Linear(in_channels, in_channels)
        self.temporal_dropout = nn.Dropout(dropout)

        # Feature Linear
        self.feature_norm = self.norm
        self.feature_linear1 = nn.Linear(in_channels, ff_dim)
        self.feature_linear2 = nn.Linear(ff_dim, in_channels)
        self.feature_dropout = nn.Dropout(dropout)

    def time_mixer(self, x):
        """Temporal mixing function."""
        x_norm = self.temporal_norm(x)
        x_transposed = x_norm.transpose(1, 2)
        x_linear = self.temporal_linear(x_transposed)
        x_res = x_linear.transpose(1, 2) + x
        x_dropout = self.temporal_dropout(x_res)
        return x_dropout

    def feature_mixer(self, x):
        """Feature mixing function."""
        x_norm = self.feature_norm(x)
        x_linear = self.feature_linear1(x_norm)
        x_linear = self.activation(x_linear)
        x_linear = self.feature_dropout(x_linear)
        x_linear = self.feature_linear2(x_linear)
        x_linear = self.feature_dropout(x_linear)
        return x_linear + x

    def forward(self, x):
        x = self.time_mixer(x)
        x = self.feature_mixer(x)
        return x

class TSMixer(nn.Module):
    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice=None):
        super(TSMixer, self).__init__()

        if n_block <= 0:
            raise ValueError("n_block should be greater than 0.")

        self.in_channels = input_shape[1]
        self.blocks = nn.ModuleList(
            [ResBlock(self.in_channels, norm_type, dropout, ff_dim, activation) for _ in range(n_block)]
        )
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