import os
from pathlib import Path

import gym
import ray
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import AnimalAIGym


# this is a dummy agent that I can use to interact with the environment.
# the goal is to create a environment and throw it into baselines unity integration.
# I allocate 20 minutes to it right now.

# def main():
#     env = AnimalAIGym(
#             environment_filename='examples/env/AnimalAI',
#             arenas_configurations=ArenaConfig('examples/configurations/curriculum/0.yml')
#         )
#     model = PPO('MlpPolicy', env, verbose=1)
#     model.learn(total_timesteps=10)
# 
#     obs = env.reset()
#     for i in range(1000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         # inspect the model, we have access to the states.
#         # this is good.
#         env.render()
#         if done:
#             obs = env.reset()
# 
#     env.close()
from ray.tune import register_env, tune


class RayAIIGym(AnimalAIGym):
    def __init__(self, env_config):
        super(RayAIIGym, self).__init__(worker_id=env_config.worker_index,
                                        environment_filename='examples/env/AnimalAI',
                                        arenas_configurations=ArenaConfig('examples/configurations/curriculum/0.yml'))


def main():
    import ray
    from ray.rllib.agents import ppo
    from ray.tune.logger import pretty_print

    ray.init(include_dashboard=False)
    # trainer = ppo.PPOTrainer(env=RayAIIGym, config={"log_level":"INFO"})
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 4
    config["env"] = RayAIIGym
    # trainer = ppo.PPOTrainer(config=config, env=RayAIIGym)
    log_dir = "log"
    analysis = ray.tune.run(
        ppo.PPOTrainer,
        config=config,
        local_dir=log_dir,
        checkpoint_at_end=True)


def main2():
    import ray
    import ray.rllib.agents.ppo as ppo
    from ray.tune.logger import pretty_print

    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    # config["eager"] = False
    trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")

    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent

class UnityEnvWrapper(gym.Env):
    def __init__(self, env_config):
        self.vector_index = env_config.vector_index
        self.worker_index = env_config.worker_index
        self.worker_id = env_config["unity_worker_id"] + env_config.worker_index
        self.env = AnimalAIGym(worker_id=self.worker_id,
                               environment_filename=str(ROOT_DIR/'examples/env/AnimalAI'),
                               arenas_configurations=ArenaConfig(str(ROOT_DIR/'examples/configurations/curriculum/3.yml'))
                               )  #
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

def main3():

    register_env("unity_env", lambda config: UnityEnvWrapper(config))
    ray.init(include_dashboard=False)

    tune.run(
        "PPO",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": "unity_env",
            "num_workers": 0,  # for multi env training
            "env_config": {
                "unity_worker_id": 52
            },
            "train_batch_size": 500,
            # "local_dir":str(ROOT_DIR/"log")
        },
        checkpoint_at_end=True,
    )

if __name__ == '__main__':
    main()
    """
    TODO
    None of the training is successful.
    Assume the training is successful and retrieve data.
    """
