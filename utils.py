import os
import glob
import yaml

def load_config(config_dir='./configs'):
    config_files = glob.glob(os.path.join(config_dir, '*.yaml'))
    config = {}

    for config_file in config_files:
        file_path = config_file
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    config.update(loaded_config)
    return config

def make_logdir(base_dir: str, exp_name: str) -> str:
    """Return a unique log directory within a subfolder named after the experiment"""
    safe_exp_name = exp_name.replace(":", "_")
    exp_dir = os.path.join(base_dir, safe_exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    version_cnt = 0
    log_dir = os.path.join(exp_dir, f"version_{version_cnt}")
    while os.path.exists(log_dir):
        version_cnt += 1
        log_dir = os.path.join(exp_dir, f"version_{version_cnt}")

    os.makedirs(log_dir)
    return log_dir
