import argparse
import json
from pathlib import Path

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv


def monitor_dir(cfg):
    return Path(cfg["root"]) / cfg["name"] / 'monitor'


def make_env(cfg, rank=0, seed=0):
    def _init():
        env = gym.make(cfg["env_id"])
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', default='test', nargs='?')
    parser.add_argument('--root', type=str, default='.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    args_dict = vars(args)

    with open('config.json') as f:
        config_json = json.load(f)
        json_dict = config_json[args_dict["name"]]

    config_dict = {**json_dict, **args_dict}

    config_dict["home"] = Path(config_dict["root"]) / 'runs' / config_dict["name"]
    config_dict["model_pkl"] = config_dict["home"] / f'PPO-{config_dict["env_id"]}-{config_dict["timesteps"]}'

    return config_dict


def train(cfg):
    ts = cfg["timesteps"]
    env = SubprocVecEnv([make_env(cfg, i, cfg["seed"]) for i in range(6)])
    model = PPO('MlpPolicy', env,
                tensorboard_log=cfg["model_pkl"], verbose=True)
    model.learn(total_timesteps=ts)
    model.save()


def evaluate(cfg):
    model = PPO.load(cfg["model_pkl"])
    env = Monitor(gym.make(cfg["env_id"]))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=True)
    print(f'{mean_reward}+={std_reward}')


def main():
    cfg = parse()

    if cfg["train"]:
        train(cfg)

    evaluate(cfg)


if __name__ == '__main__':
    main()
