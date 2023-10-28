import argparse

import inference
import train

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

    if args.action == "train":
        if args.exp_name is None:
            parser.error("--exp_name is required for training!")
        train.main(args.exp_name)
    elif args.action == "inference":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for inference!")
        inference.main(args.checkpoint)
