import os
import yaml
import argparse
import glob

def load_config(config_dir='configs'):
    config_files = glob.glob('*_config.yaml')
    config = {}

    for config_file in config_files:
        file_path = os.path.join(config_dir, config_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    config.update(loaded_config)

    return config

def make_logdir(base_dir: str, exp_name: str) -> str:
    """Return a unique log directory within a subfolder named after the experiment"""
    exp_dir = os.path.join(base_dir, exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    version_cnt = 0
    log_dir = os.path.join(exp_dir, f"version_{version_cnt}")
    while os.path.exists(log_dir):
        version_cnt += 1
        log_dir = os.path.join(exp_dir, f"version_{version_cnt}")

    os.makedirs(log_dir)
    return log_dir

def update_config_from_args(config, args):
    """Update the 'general' key of the config dict based on command line arguments"""
    args_dict = vars(args)
    config['general'].update(args_dict)
    return config