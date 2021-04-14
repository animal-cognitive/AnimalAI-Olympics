def load_trainer():
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

    from cachey.cache_model import MyCNNRNNModel
    # from config import get_cfg

    # from custom_model import *

    # %%

    ## Reuse Wrapper for AnimalAI Environment

    class UnityEnvWrapper(gym.Env):
        def __init__(self, env_config):
            self.vector_index = env_config.vector_index
            self.worker_index = env_config.worker_index
            self.worker_id = env_config["unity_worker_id"] + env_config.worker_index
            print("Opened")
            self.env = AnimalAIGym(
                environment_filename="examples/env/AnimalAI",
                worker_id=self.worker_id,
                flatten_branched=True,
                uint8_visual=True,
                arenas_configurations=ArenaConfig(env_config['arena_to_train'])
            )
            self.env.base_port = env_config['base_port']
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

        def reset(self):
            return self.env.reset()

        def step(self, action):
            return self.env.step(action)

    # %%

    ## Setup configuration to use

    conf = {
        "num_workers": 0,
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
        "num_workers": 1,  # parallelism
        "framework": "torch",
        "train_batch_size": 500
    }
    conf

    # %%

    ## Setup and register environment
    # ray.shutdown()
    # ray.init(num_gpus=2)

    ModelCatalog.register_custom_model("my_cnn_rnn_model", MyCNNRNNModel)

    register_env("unity_env", lambda config: UnityEnvWrapper(config))

    # %%

    # conf["env_config"]["base_port"] = 5009
    trainer = PPOTrainer(config=conf, env="unity_env")
    PATH_TO_CHECKPOINT_FILE = 'log/PPO_unity_env_2021-04-09_02-16-38lwq8v0v4/checkpoint_3500/checkpoint-3500'
    # trainer.restore('/home/azibit/ray_results/PPO_unity_env_2021-04-08_16-48-575mk5ga4m/checkpoint_500/checkpoint-500')
    trainer.restore(PATH_TO_CHECKPOINT_FILE)
    return trainer

if __name__ == '__main__':
    tr = load_trainer()
    print(tr)