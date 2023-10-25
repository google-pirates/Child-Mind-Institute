import argparse
from train import train_main
from inference import inference_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train or inference scripts.")
    parser.add_argument("action", choices=["train", "inference"], help="Action to perform: train or inference.")
    parser.add_argument("--exp_name", type=str, help="Experiment name for training.")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path for inference.")

    args = parser.parse_args()

    if args.action == "train" and args.exp_name is None:
        parser.error("--exp_name is required for training!")
    elif args.action == "inference" and args.checkpoint is None:
        parser.error("--checkpoint is required for inference!")

    if args.action == "train":
        train_main(args.exp_name) 
    elif args.action == "inference":
        inference_main(args.checkpoint)