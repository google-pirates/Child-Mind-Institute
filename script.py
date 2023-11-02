from utils import load_config

import inference
import train

import argparse

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


    if args.task == "train":
        if args.exp_name is None:
            parser.error("--exp_name is required for training!")
        train.main(config)
    elif args.task == "inference":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for inference!")
        inference.main(config)
