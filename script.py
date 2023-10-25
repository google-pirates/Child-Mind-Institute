import subprocess
import argparse

def run_train(args):
    cmd = [
        "python", "/mnt/data/train.py",
        "--exp_name", args.exp_name,
        # ... 다른 인자들 ...
    ]
    subprocess.run(cmd)

def run_inference(args):
    cmd = [
        "python", "/mnt/data/inference.py",
        "--checkpoint", args.checkpoint,
        # ... 다른 인자들 ...
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train or inference scripts.")
    parser.add_argument("action", choices=["train", "inference"], help="Action to perform: train or inference.")
    parser.add_argument("--exp_name", type=str, help="Experiment name for training.")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path for inference.")
    # ... 여기에 다른 인자들을 추가 ...

    args = parser.parse_args()

    if args.action == "train":
        run_train(args)
    elif args.action == "inference":
        run_inference(args)