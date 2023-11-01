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
        config['train']['learnig_rate'] = None
        config['train']['batch_size'] = None
        config['train']['window_size'] = None
        config['train']['data']['scaler'] = None
        config['train']['optimizer'] = None

        model_name = config.get('train').get('model').lower()
        if model_name == 'cnn':
            config['model']['cnn']['out_features'] = None
            config['model']['cnn']['pooling_sizes'] = None
            config['model']['cnn']['kernel_sizes'] = None
            config['model']['cnn']['strides'] = None
            config['model']['cnn']['dilations'] = None
            config['model']['cnn']['dropout_rates'] = None
            config['model']['cnn']['fc_outputs'] = None
            config['model']['cnn']['fc_dropout_rates'] = None

        elif model_name == 'LSTM':
            config['model']['lstm']['out_features'] = None
            config['model']['lstm']['bidirectional'] = None
            config['model']['lstm']['dropout_rates'] = None
            config['model']['lstm']['fc_outputs'] = None
            config['model']['lstm']['fc_dropout_rates'] = None

        elif model_name == 'tsmixer':
            config['model']['tsmixer']['n_block'] = None
            config['model']['tsmixer']['dropout_rates'] = None
            config['model']['tsmixer']['ff_dim'] = None

    if args.task == "train":
        if args.exp_name is None:
            parser.error("--exp_name is required for training!")
        train.main(config)
    elif args.task == "inference":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for inference!")
        inference.main(config)
