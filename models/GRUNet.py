from typing import Dict, Any
import torch.nn as nn


class RNNBlock(nn.Module):
    def __init__(
            self,
            configs: Dict[str, Any],
            **kwargs,
            ):
        super(RNNBlock, self).__init__()

        self.block_config = configs
        self.hidden_size = self.block_config.get('hidden_size')
        self.n_stacks = self.block_config.get('n_stacks')
        self.n_layers = self.block_config.get('n_layers')
        self.cell_type = self.block_config.get('cell_type')
        self.activation = self.block_config.get('activation')
        self.normalization_layer = self.block_config.get('normalization_layer')
        self.dropout_rates = self.block_config.get('dropout_rates')
        self.bidirectional = self.block_config.get('bidirectional')

        bidirectional_unit_factor = 2 if self.bidirectional else 1

        layers = []
        for _ in range(self.n_layers):
            layers.append(
                getattr(nn, self.cell_type)(
                    self.hidden_size,
                    self.hidden_size,
                    self.n_stacks,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            )

            layers.append(
                nn.Linear(
                    in_features=self.hidden_size*bidirectional_unit_factor,
                    out_features=self.hidden_size*bidirectional_unit_factor*2,
                )
            )

            if self.normalization_layer:
                layers.append(
                    getattr(nn, self.normalization_layer)(
                        self.hidden_size*bidirectional_unit_factor*2,
                    )
                )

            layers.append(
                getattr(nn, self.activation)(),
            )

            if self.dropout_rates:
                layers.append(
                    nn.Dropout(p=self.dropout_rates),
            )

            layers.append(
                nn.Linear(
                    in_features=self.hidden_size*bidirectional_unit_factor*2,
                    out_features=self.hidden_size,
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x, hidden=None):
        for layer in self.layers:
            if isinstance(layer, getattr(nn, self.cell_type)):
                residual, hidden = layer(x, hidden)
            else:
                residual = layer(residual)
        x += residual

        return x, hidden


class RNN(nn.Module):
    def __init__(
            self,
            configs,
            **kwargs,
        ):
        super(RNN, self).__init__()

        self.input_size = configs.get('model').get('grunet').get('grunet').get('out_channels')[-1]
        self.block_configs = configs.get('model').get('grunet').get('block')
        self.configs = configs.get('model').get('grunet').get('rnn')
        self.hidden_size = self.configs.get('hidden_size')
        self.output_size = self.configs.get('output_size')
        self.n_layers = self.configs.get('n_layers')
        self.cell_type = self.configs.get('cell_type')

        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.normalization_layer = nn.LayerNorm(self.hidden_size)
        self.rnns = nn.ModuleList(
            [
                RNNBlock(self.block_configs)
                for _
                in range(self.n_layers)
            ]
        )
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = [None for _ in range(self.n_layers)]

        x = self.fc_in(x)
        x = self.normalization_layer(x)
        x = nn.LeakyReLU()(x)

        new_hiddens = []
        for i, rnn_block in enumerate(self.rnns):
            x, new_hidden = rnn_block(x, new_hiddens[i-1] if i > 1 else hidden[i])
            new_hiddens.append(new_hidden)

        x = self.fc_out(x)

        return x, new_hiddens


class EncoderBlock(nn.Module):
    def __init__(
            self,
            configs,
            **kwargs
            ):
        super(EncoderBlock, self).__init__()

        self.in_channels = kwargs.get('in_channel')
        self.output_size = kwargs.get('out_channel')
        self.kernel_size = kwargs.get('kernel_size')
        self.stride = kwargs.get('stride')
        self.padding = kwargs.get('padding')
        self.dilation = kwargs.get('dilation')
        self.encoder_block_configs = configs.get('model').get('grunet').get('encoder_block')
        self.activation = self.encoder_block_configs.get('activation')
        self.normalization_layer = self.encoder_block_configs.get('normalization_layer')
        self.dropout_rates = self.encoder_block_configs.get('dropout_rates')

        layers = []
        layers.append(
            nn.Conv1d(
                self.in_channels,
                self.output_size,
                self.kernel_size,
                self.stride,
                padding=self.padding,
                dilation=self.dilation,
                )
        )

        if self.normalization_layer:
            layers.append(
                getattr(nn, self.normalization_layer)(
                    self.output_size,
                )
            )

        layers.append(
            getattr(nn, self.activation)(),
        )

        if self.dropout_rates:
            layers.append(
                nn.Dropout(p=self.dropout_rates),
        )

        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x.transpose(-1, -2))

        return x


class GRUNet(nn.Module):
    def __init__(
            self,
            configs,
            **kwargs,
            ):
        super(GRUNet, self).__init__()

        self.configs = configs
        self.model_configs = configs.get('model').get('grunet')
        self.n_layers = self.model_configs.get('rnn').get('n_layers')
        self.in_channels = configs.get('train').get('n_features')
        self.out_channels = self.model_configs.get('grunet').get('out_channels')
        self.channels = [self.in_channels] + self.out_channels
        self.kernel_sizes = (
            self.model_configs.get('grunet').get('kernel_sizes')*(len(self.out_channels)+1)
            if len(self.model_configs.get('grunet').get('kernel_sizes')) == 1 and len(self.out_channels) >= 2
            else self.model_configs.get('grunet').get('kernel_sizes')
        )
        self.strides = (
            self.model_configs.get('grunet').get('strides')*(len(self.out_channels)+1)
            if len(self.model_configs.get('grunet').get('strides')) == 1 and len(self.out_channels) >= 2
            else self.model_configs.get('grunet').get('strides')
        )
        self.dilations = (
            self.model_configs.get('grunet').get('dilations')*(len(self.out_channels)+1)
            if len(self.model_configs.get('grunet').get('dilations')) == 1 and len(self.out_channels) >= 2
            else self.model_configs.get('grunet').get('dilations')
        )
        self.deconvolution_paddings = (
            self.model_configs.get('grunet').get('deconvolution_paddings')*(len(self.out_channels)+1)
            if len(self.model_configs.get('grunet').get('deconvolution_paddings')) == 1 and len(self.out_channels) >= 2
            else self.model_configs.get('grunet').get('deconvolution_paddings')
        )

        convolutions = []
        for i, _ in enumerate(self.channels):
            if i == len(self.channels)-1:
                break

            convolutions.append(
                EncoderBlock(
                    configs=self.configs,
                    in_channel=self.channels[i],
                    out_channel=self.channels[i+1],
                    kernel_size=self.kernel_sizes[i],
                    stride=self.strides[i],
                    padding=self.kernel_sizes[i]//2,
                    dilation=self.dilations[i],
                )
            )
        self.convolutions = nn.Sequential(*convolutions)

        self.rnns = RNN(self.configs)

        deconvolutions = []
        for i, _ in enumerate(reversed(self.channels)):
            if i == len(self.channels)-1:
                break

            deconvolutions.append(
                nn.ConvTranspose1d(
                    self.channels[::-1][i],
                    self.channels[::-1][i+1],
                    self.kernel_sizes[::-1][i],
                    stride=self.strides[i],
                    padding=self.kernel_sizes[::-1][i]//2,
                    dilation=self.dilations[::-1][i],
                    output_padding=1,
                )
            )

            for _ in range(2):
                deconvolutions.append(
                    nn.Conv1d(    
                        in_channels=self.channels[::-1][i+1],
                        out_channels=self.channels[::-1][i+1],
                        kernel_size=self.kernel_sizes[::-1][i],
                        stride=1,
                        padding=self.kernel_sizes[::-1][i]//2,
                        dilation=self.dilations[::-1][i],
                    )
                )

        self.deconvolutions = nn.Sequential(*deconvolutions)
        self.output_layer = nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1, stride=1)

    def forward(self, batch, h=None):
        x = batch.get('X').permute(0, 2, 1)
        if h is None:
            h = [None for _ in range(self.n_layers)]

        x = self.convolutions(x)

        x, new_hiddens = self.rnns(x)

        x = self.deconvolutions(x.transpose(-1, -2))

        x = self.output_layer(x)
        x = x.transpose(-1, -2)

        return x, new_hiddens
