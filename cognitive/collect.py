import gym
import ray
import ray.rllib.agents.a3c as a3c
from ray import tune

from cachey.cache_model import MyCNNRNNModel
from cognitive.primitive_arena import *

import ray
import ray.rllib.agents.a3c as a3c

from cachey.config import get_cfg
from cognitive.primitive_arena import Occlusion
from ray.rllib.models import ModelCatalog
from cachey.custom_model import *


def train_agent():
    ray.init(include_dashboard=False)
    config = {
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.0001]),  # ([0.01, 0.001, 0.0001]),
    }
    analysis = tune.run(
        "A3C",
        stop={"episode_reward_mean": 120},
        config=config,
        local_dir="log",
        checkpoint_at_end=True
    )

    # list of lists: one list per checkpoint; each checkpoint list contains
    # 1st the path, 2nd the metric value
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean")
    print(checkpoints)
    return checkpoints


def access_agent():
    ray.init(include_dashboard=False)
    config = {
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
    }
    # checkpoint_path = checkpoints[0][0]
    agent = a3c.A3CTrainer(config=config)
    # agent.restore('D:\\Git\\AnimalAI-Olympics\\log\\PPO\\PPO_CartPole-v0_314b5_00000_0_lr=0.0001_2021-03-19_15-32-15\\checkpoint_000005\\checkpoint-5')
    agent.restore(
        'D:\\Git\\AnimalAI-Olympics\\log\\A3C\\A3C_CartPole-v0_b1b96_00000_0_lr=0.0001_2021-03-19_15-50-10\\checkpoint_000040\\checkpoint-40')
    env = gym.make('CartPole-v0')
    obs = env.reset()
    sum_reward = 0
    for i in range(1000):
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            print(agent.get_policy().model)
            print(agent.get_policy())
            print(agent.get_policy().model.base_model.summary())
            break


# class DIRCollection:
#     def __init__(self):
#         self.modes = [BeforeOrBehind, Occlusion, Rotation]
#
#     def run(self, mode, agent):
#         bob = Rotation()
#         arena_config, con = bob.generate_config()

def register_models():
    # ModelCatalog.register_custom_model("my_fc_model", MyCNNRNNModel)
    ModelCatalog.register_custom_model("my_rnn_model", MyRNNModel)
    ModelCatalog.register_custom_model("my_convgru_model", MyConvGRUModel)  # NOTE: Only works with image observations.
    ModelCatalog.register_custom_model("mm", MyCNNRNNModel)  # NOTE: Only works with image observations.


def collect(Arena: ArenaManager = BeforeOrBehind, ds_size =1000):
    register_models()
    ray.init(include_dashboard=False)
    env_config = ppo.DEFAULT_CONFIG.copy()
    arena = Arena()
    arena_config, settings = arena.generate_config()
    env_config_ = {
        "env": GymFactory(arena_config),
        "num_gpus": 1,
        "num_workers": 0,
        "framework": 'torch',
        "model": {
            "custom_model": 'mm',
        },
        "log_level": 'INFO',
    }
    env_config.update(env_config_)
    trainer = ppo.PPOTrainer(config=env_config)
    # trainer.restore()

    input, target = [], []
    for i in range(ds_size):
        a, b = arena.collect_dir(trainer, arena_config, settings, env_config)
        input += a
        target += b
    dataset = DIRDataset(input, target)
    trainer.cleanup()
    ray.shutdown()
    return dataset

def collect_all(ds_size = 2):
    collect(Arena=BeforeOrBehind, ds_size=ds_size)
    collect(Arena=Occlusion, ds_size=ds_size)
    collect(Arena=Rotation, ds_size=ds_size)



if __name__ == '__main__':
    ds_size=2
    collect_all()