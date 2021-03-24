import logging
import random
import time
import unittest

import gym
import mlagents_envs
import torch
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import AnimalAIGym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import MultiAgentDict

logger = logging.getLogger(__name__)


class AnimalAIRayEnv(gym.Env):

    def __init__(self,
                 config, ):
        """
        Copied from ray.rllib.env.wrappers.Unity3DEnv
        """

        super().__init__()

        # Try connecting to the Unity3D game instance. If a port is blocked
        worker_id = 0
        while True:
            # Sleep for random time to allow for concurrent startup of many
            # environments (num_workers >> 1). Otherwise, would lead to port
            # conflicts sometimes.
            time.sleep(random.randint(1, 10))
            try:
                self.unity_env = AnimalAIGym(
                    environment_filename=config["environment_filename"],
                    arenas_configurations=ArenaConfig(config["yaml_path"]),
                    worker_id=worker_id,
                )
                print("Created UnityEnvironment for worker id {}".format(worker_id))
            except mlagents_envs.exception.UnityWorkerInUseException:
                worker_id += 1
                pass
            else:
                break

    @override(MultiAgentEnv)
    def step(self, action):
        return self.unity_env.step(action)

    @override(MultiAgentEnv)
    def reset(self) -> MultiAgentDict:
        """Resets the entire Unity3D scene (a single multi-agent episode)."""
        return self.unity_env.reset()

    @property
    def observation_space(self):
        return self.unity_env.observation_space

    @property
    def action_space(self):
        return self.unity_env.action_space

    def close(self):
        self.unity_env.close()

    def __reduce__(self):
        """https://docs.ray.io/en/master/serialization.html"""
        deserializer = AnimalAIRayEnv
        serialized_data = tuple()  # TODO: implement serialization
        return deserializer, serialized_data


class TestAnimalAIGym(unittest.TestCase):
    def test_sanity_animalai_gym(self):
        """
        Test that we can pass a batch of observations from the environment to our model.
        """
        from cachey.cache_model import MyCNNRNNModel
        env = AnimalAIRayEnv({
            'environment_filename': '../examples/env_windows/AnimalAI',
            'yaml_path': '../examples/configurations/curriculum/0.yml',
        })
        obs_space = env.observation_space
        action_space = env.action_space
        num_outputs = len(env.action_space.shape)
        model_config = {'gimme_lstm': False}
        name = 'test'
        prep = get_preprocessor(env.observation_space)(env.observation_space)
        obs = torch.stack([torch.from_numpy(prep.transform(env.observation_space.sample())) for _ in range(100)], dim=0)
        input_dict = {
            "obs": obs,
        }
        model = MyCNNRNNModel(obs_space, action_space, num_outputs, model_config, name)
        model(input_dict)


if __name__ == '__main__':
    unittest.main()
