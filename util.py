import os
import yaml


def load_config():
    config = {}
    with open('configs/general_config.yaml', 'r') as f:
        general_config = yaml.safe_load(f)
    with open('configs/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    with open('configs/train_config.yaml', 'r') as f:
        train_config = yaml.safe_load(f)

    if general_config:
        config.update(general_config)
    if model_config:
        config.update(model_config)
    if train_config:
        config.update(train_config)

    return config

def make_logdir(base_dir : str, exp_name : str) -> str:
    """Return a unique log directory if existing"""
    log_dir = os.path.join(base_dir, exp_name)
    cnt = 0
    while os.path.exists(log_dir):
        cnt += 1
        log_dir = os.path.join(base_dir, f"{exp_name}_{cnt}")
    return log_dir

def update_config_from_args(config, exp_name):
    """ Update the 'general' key of the config dict based on the exp_name """

    exp_params = exp_name.split(',')
    for param in exp_params:
        key, value = param.split("=")

        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        config['general'].update({key: value})
    
    return config