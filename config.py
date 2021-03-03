import argparse
import json
import multiprocessing
from pathlib import Path

"""
Argument and config parsing
"""


def get_cfg():
    """
    Get command line arguments and JSON config.
    JSON config is by default "config.json" in the root directory, defining an experiment configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('name', default='test', nargs='?')
    parser.add_argument('--root', type=str, default='.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_cpu', type=int, default=multiprocessing.cpu_count() / 2)
    parser.add_argument('--n_eval_episodes', type=int, default=10)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    args_dict = vars(args)

    with open('config.json') as f:
        config_json = json.load(f)
        json_dict = config_json[args_dict["name"]]

    cfg_dict = {**json_dict, **args_dict}

    propagate_fields(cfg_dict)

    return cfg_dict


def propagate_fields(cfg_dict):
    """
    Fill in some fields which are transforms of user-defined config
    """
    cfg_dict["home"] = Path(cfg_dict["root"]) / 'runs' / cfg_dict["name"]
    cfg_dict["model_pkl"] = cfg_dict["home"] / f'PPO-{cfg_dict["env_id"]}-{cfg_dict["timesteps"]}'
