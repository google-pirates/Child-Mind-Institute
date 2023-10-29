from typing import Dict, Any
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super(LSTM, self).__init__()
        self.model_config = config.get('model').get('lstm')
        self.n_features = config.get('train').get('n_features')
        self.seq_len = config.get('train').get('seq_len')
        self.batch_size = config.get('train').get('batch_size')

        self.activation = self.model_config.get('activation')
        self.normalization_layer = self.model_config.get('normalization_layer')
        self.bidirectional = self.model_config.get('bidirectional')

        self.out_features = self.model_config.get('out_features')
        num_lstm_layers = len(self.out_features)
        self.dropout_rates = (self.model_config.get('dropout_rates')
                              if num_lstm_layers == len(self.model_config.get('dropout_rates'))
                              else self.model_config.get('dropout_rates')*num_lstm_layers)

        self.fc_outputs = self.model_config.get('fc_outputs')
        num_fc_layers = len(self.fc_outputs)
        self.fc_dropout_rates = (self.model_config.get('fc_dropout_rates')
                                 if num_fc_layers == len(self.model_config.get('fc_dropout_rates'))
                                 else self.model_config.get('fc_dropout_rates')*(num_fc_layers-1))

        ## CNN Layers
        in_feature = self.n_features
        seq_len = self.seq_len
        self.lstm_blocks = []
        for (
            out_feature,
            dropout_rate,
        ) in zip(
            self.out_features,
            self.dropout_rates
        ):
            self.lstm_blocks.append(
                nn.LSTM(
                    input_size=in_feature,
                    hidden_size=out_feature,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            )

            if self.normalization_layer:
                self.lstm_blocks.append(getattr(nn, self.normalization_layer)())

            self.lstm_blocks.append(getattr(nn, self.activation)())

            if self.dropout_rates:
                self.lstm_blocks.append(nn.Dropout(p=dropout_rate))

            in_feature = out_feature*2 if self.bidirectional else out_feature

        ## FC Layers
        fc_input = (
            self.out_features[-1]*seq_len*2
            if self.bidirectional
            else self.out_features[-1]*seq_len
        )
        fc_layers = []
        for (
            i, (fc_output, fc_dropout_rate)
        ) in enumerate(
            zip(self.fc_outputs, self.fc_dropout_rates),
            1
        ):
            fc_layers.append(
                nn.Linear(
                    in_features=fc_input,
                    out_features=fc_output,
                )
            )

            if i == num_fc_layers:
                break

            if self.normalization_layer:
                fc_layers.append(nn.BatchNorm1d(num_features=fc_output))

            fc_layers.append(getattr(nn, self.activation)())

            if self.fc_dropout_rates:
                fc_layers.append(nn.Dropout(p=fc_dropout_rate))

            fc_input = fc_output
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch.get('X')
        for block in self.lstm_blocks:
            x = block(x)

            if isinstance(block, nn.RNNBase):
                x, _ = x

        x = x.reshape(self.batch_size, -1)
        x = self.fc_layers(x)

        return x
