from typing import Dict, Any, List
from math import floor
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super(CNN, self).__init__()
        self.model_config = config.get('model').get('cnn')
        self.batch_size = config.get('train').get('batch_size')
        self.n_features = config.get('train').get('n_features')
        self.seq_len = config.get('train').get('seq_len')

        self.activation = self.model_config.get('activation')
        self.normalization_layer = self.model_config.get('normalization_layer')
        self.pooling = self.model_config.get('pooling')

        self.out_features = self.model_config.get('out_features')
        num_cnn_layers = len(self.out_features)
        self.pooling_sizes = (self.model_config.get('pooling_sizes')
                              if num_cnn_layers == len(self.model_config.get('pooling_sizes'))
                              else self.model_config.get('pooling_sizes')*num_cnn_layers)
        self.kernel_sizes = (self.model_config.get('kernel_sizes')
                              if num_cnn_layers == len(self.model_config.get('kernel_sizes'))
                              else self.model_config.get('kernel_sizes')*num_cnn_layers)
        self.strides = (self.model_config.get('strides')
                        if num_cnn_layers == len(self.model_config.get('strides'))
                        else self.model_config.get('strides')*num_cnn_layers)        
        self.dilations = (self.model_config.get('dilations')
                          if num_cnn_layers == len(self.model_config.get('dilations'))
                          else self.model_config.get('dilations')*num_cnn_layers)
        self.dropout_rates = (self.model_config.get('dropout_rates')
                              if num_cnn_layers == len(self.model_config.get('dropout_rates'))
                              else self.model_config.get('dropout_rates')*num_cnn_layers)   

        self.fc_outputs = self.model_config.get('fc_outputs')
        num_fc_layers = len(self.fc_outputs)
        self.fc_dropout_rates = (self.model_config.get('fc_dropout_rates')
                                 if num_fc_layers == len(self.model_config.get('fc_dropout_rates'))
                                 else self.model_config.get('fc_dropout_rates')*(num_fc_layers-1))

        ## CNN Layers
        in_feature = self.n_features
        seq_len = self.seq_len
        self.cnn_blocks = nn.ModuleList()  ## torchscript의 scripting을 위해 추가.
        for (
            out_feature,
            kernel_size,
            stride,
            dilation,
            dropout_rate,
        ) in zip(
            self.out_features,
            self.kernel_sizes,
            self.strides,
            self.dilations,
            self.dropout_rates
        ):
            self.cnn_blocks.append(
                nn.Conv1d(
                    in_channels=in_feature,
                    out_channels=out_feature,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                )
            )
            seq_len = floor((seq_len - dilation*(kernel_size-1) - 1) / stride + 1)

            if self.normalization_layer:
                self.cnn_blocks.append(getattr(nn, self.normalization_layer)())

            self.cnn_blocks.append(getattr(nn, self.activation)())

            if self.dropout_rates:
                self.cnn_blocks.append(nn.Dropout(p=dropout_rate))

            if self.pooling:
                self.cnn_blocks.append(
                    nn.MaxPool1d(
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                    )
                )

                seq_len = floor((seq_len - dilation*(kernel_size-1) - 1) / stride + 1)

            in_feature = out_feature

        ## FC Layers
        fc_input = self.out_features[-1]*seq_len
        fc_layers = nn.ModuleList() ## for TorchScript
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
        
        if x is None:
            raise ValueError("Input 'X' is not found in the batch") ## TorchScript에서 Optional[Tensor] 를 트래킹하기 위함
        
        x = x.permute(0, 2, 1)
        for block in self.cnn_blocks:
            x = block(x)

        # x = x.reshape(self.batch_size, -1)
        ## 마지막 배치의 경우 입력한 batch size 보다 작을 수 있음.
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)

        return x
