"""
RLLib tutorial https://docs.ray.io/en/master/rllib-training.html#basic-python-api
"""

import os
import ray
from ray import tune
from ray.rllib.agents.ppo import ppo

if __name__ == "__main__":
    ray.init()
    num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = num_gpus
    config["num_workers"] = 1
    config["framework"] = 'torch'
    config["log_level"] = 'INFO'
    config["env"] = 'CartPole-v0'
    resources = ppo.PPOTrainer.default_resource_request(config).to_json()
    tune.run('PPO', config=config, local_dir='ray_results')
