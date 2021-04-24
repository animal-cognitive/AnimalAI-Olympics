import ray
from mlagents_envs.exception import UnityWorkerInUseException, UnityTimeOutException

from cognitive.primitive_arena import GymFactory


def load_trainer(model="reduced"):
    ## Import All Needed Libraries

    import os
    import gym
    import ray
    from animalai.envs.arena_config import ArenaConfig
    from animalai.envs.gym.environment import AnimalAIGym
    from ray.rllib.agents import ppo

    from ray.tune import register_env, tune

    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.models import ModelCatalog
    from ray.tune.logger import pretty_print

    # from config import get_cfg
    # from custom_model import *
    # %%

    ## Reuse Wrapper for AnimalAI Environment
    def env_factory(env_config):
        arena_config = ArenaConfig(env_config['arena_to_train'])
        GymClass = GymFactory(arena_config)
        return GymClass(env_config)

    # %%
    ## Setup configuration to use

    conf = {
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "env_config": {
            "unity_worker_id": 483,
            "arena_to_train": 'examples/configurations/curriculum/0.yml',
            "base_port": 1234
        },
        "model": {
            "custom_model": 'my_cnn_rnn_model',
            "custom_model_config": {},
        },
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
        "framework": "torch",
        "train_batch_size": 500
    }
    conf

    # %%

    ## Setup and register environment
    # ray.shutdown()
    # ray.init(num_gpus=2)
    if model == "reduced":
        from cachey.reduced_cache_model import MyCNNRNNModel
        ModelCatalog.register_custom_model("my_cnn_rnn_model", MyCNNRNNModel)
        PATH_TO_CHECKPOINT_FILE = 'log/PPO_unity_env_2021-04-09_02-16-38lwq8v0v4/checkpoint_3500/checkpoint-3500'
    elif model == "whole":
        from cachey.whole_cache_model import MyCNNRNNModel
        ModelCatalog.register_custom_model("my_cnn_rnn_model", MyCNNRNNModel)
        PATH_TO_CHECKPOINT_FILE = 'log/PPO_unity_env_2021-04-13_00-31-01qhgyp4n0/checkpoint_1144/checkpoint-1144'
    elif model == "lstm":
        from cachey.whole_cache_model import MyCNNRNNModel
        ModelCatalog.register_custom_model("my_cnn_rnn_model", MyCNNRNNModel)
        PATH_TO_CHECKPOINT_FILE = 'log/PPO_unity_env_2021-04-18_10-51-24rl1myjn5_lstm/checkpoint_5000/checkpoint-5000'
    elif model == "rnn":
        from cachey.whole_cache_model import MyCNNRNNModel
        ModelCatalog.register_custom_model("my_cnn_rnn_model", MyCNNRNNModel)
        PATH_TO_CHECKPOINT_FILE = 'log/PPO_unity_env_2021-04-18_20-51-57f1rzdwyo_cnn/checkpoint_5000/checkpoint-5000'
    else:
        raise ValueError

    register_env("unity_env", env_factory)

    # %%

    # conf["env_config"]["base_port"] = 5009
    trainer = PPOTrainer(config=conf, env="unity_env")
    # trainer.restore('/home/azibit/ray_results/PPO_unity_env_2021-04-08_16-48-575mk5ga4m/checkpoint_500/checkpoint-500')
    trainer.restore(PATH_TO_CHECKPOINT_FILE)
    return trainer

if __name__ == '__main__':
    ray.init()
    tr = load_trainer(model="whole")
    print(tr)