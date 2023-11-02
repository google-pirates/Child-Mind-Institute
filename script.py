from utils import load_config

import inference
import train

import argparse
import nni

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == "__main__":
    """
    Usage:
        python3 script.py train --exp_name ...
        python3 script.py inference --checkpoint ...
    """
    parser = argparse.ArgumentParser(
        description="Run train or inference scripts."
    )
    parser.add_argument("task", choices=["train", "inference"])
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--checkpoint", type=str)

    args = parser.parse_args()

    #### config load ####
    config = load_config()

    # Update exp_name args to 'general' config
    config['general'].update(vars(args))

    #### nni ####
    if config.get('train').get('nni'):
        params = nni.get_next_parameter()

        config['train']['learnig_rate'] = params.get('learnig_rate')
        config['train']['batch_size'] = params.get('batch_size')
        config['train']['window_size'] = params.get('window_size')
        config['train']['data']['scaler'] = params.get('scaler')
        config['train']['optimizer'] = params.get('optimizer')

        model_name = config.get('train').get('model').lower()
        if model_name == 'cnn':
            config['model']['cnn']['out_features'] = params.get('out_features')
            config['model']['cnn']['pooling_sizes'] = [params.get('pooling_sizes')]
            config['model']['cnn']['kernel_sizes'] = [params.get('kernel_sizes')]
            config['model']['cnn']['strides'] = [params.get('strides')]
            config['model']['cnn']['dilations'] = [params.get('dilations')]
            config['model']['cnn']['dropout_rates'] = [params.get('dropout_rates')]
            config['model']['cnn']['fc_outputs'] = params.get('fc_outputs')
            config['model']['cnn']['fc_dropout_rates'] = params.get('fc_dropout_rates')

        elif model_name == 'LSTM':
            config['model']['lstm']['out_features'] = params.get('out_features')
            config['model']['lstm']['bidirectional'] = params.get('bidirectional')
            config['model']['lstm']['dropout_rates'] = [params.get('dropout_rates')]
            config['model']['lstm']['fc_outputs'] = params.get('fc_outputs')
            config['model']['lstm']['fc_dropout_rates'] = params.get('fc_dropout_rates')

        elif model_name == 'tsmixer':
            config['model']['tsmixer']['n_block'] = params.get('n_block')
            config['model']['tsmixer']['dropout_rates'] = [params.get('dropout_rates')]
            config['model']['tsmixer']['ff_dim'] = params.get('ff_dim')

    if args.task == "train":
        if args.exp_name is None:
            parser.error("--exp_name is required for training!")
        train.main(config)
    elif args.task == "inference":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for inference!")
        inference.main(config)
