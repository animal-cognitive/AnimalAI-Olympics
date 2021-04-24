#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import All Needed Libraries

import os
from pathlib import Path

from animalai.envs.gym.environment import AnimalAIGym

from reduced_cache_model import *

# In[2]:
## Reuse Wrapper for AnimalAI Environment
content_root = Path(__file__).parent.parent


class UnityEnvWrapper(gym.Env):
    def __init__(self, env_config):
        self.vector_index = env_config.vector_index
        self.worker_index = env_config.worker_index
        self.worker_id = env_config["unity_worker_id"] + env_config.worker_index
        self.env = AnimalAIGym(
            environment_filename=str((content_root / "examples/env/AnimalAI").absolute()),
            worker_id=self.worker_id,
            flatten_branched=True,
            uint8_visual=True,
            arenas_configurations=ArenaConfig(env_config['arena_to_train'])
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


## Setup and register environment
ray.init(num_gpus=1, include_dashboard=False)

# Register custom models so that we can give the ID to the policy trainer
ModelCatalog.register_custom_model("my_cnn_rnn_model", MyCNNRNNModel)

register_env("unity_env", lambda config: UnityEnvWrapper(config))

# In[26]:


arena_configurations = ['0.yml', '1.yml', '2.yml', '3.yml', '4.yml']
thresholds = [
    0.8,
    0.8,
    0.8,
    0.6,
    0.2
]

## Setup configuration to use
conf = {
    "model": {
        "custom_model": 'my_cnn_rnn_model',
        "custom_model_config": {},
    },
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
    "num_workers": 0,  # parallelism
    "framework": "torch",
    "train_batch_size": 500
}


def update_phase(phase, ep_rew_mean, lesson_length):
    min_lesson_length = 100
    print(f"Lesson length: {lesson_length}/{min_lesson_length}")
    if lesson_length < min_lesson_length:
        return phase
    if ep_rew_mean > thresholds[phase]:
        return phase + 1
    return phase


def train(conf, reporter):
    phase = 0
    lesson_length = 0
    conf["env_config"] = {
        "unity_worker_id": 107,
        "arena_to_train": str(
            (content_root / 'examples/configurations/curriculum' / arena_configurations[phase]).absolute()),
    }
    trainer = PPOTrainer(config=conf, env="unity_env")

    # Load the first checkpoint from arena0-14runs
    load_path = 'C:\\Users\\benja\\ray_results\\arena0-14times\\checkpoint_100\\checkpoint-100'
    load_phase = 0
    load_lesson_length = 100

    print('Loading checkpoint ', load_path)
    trainer.restore(load_path)
    phase = load_phase
    lesson_length = load_lesson_length

    print("HERE")
    while True:
        result = trainer.train()
        lesson_length += 1
        reporter(**result)
        new_phase = update_phase(phase, result["episode_reward_mean"], lesson_length)

        # Check new phase
        if new_phase != phase:
            print('New phase.', phase, '->', new_phase)
            lesson_length = 0
            phase = new_phase
            checkpoint = trainer.save()
            print(checkpoint)

            # Update curriculum
            trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.env._env.reset(arenas_configurations=
                    ArenaConfig(
                        str((content_root / 'examples/configurations/curriculum' /
                             arena_configurations[phase]).absolute())))))
        if new_phase >= len(arena_configurations):
            print('DONE')
            break

        # Checkpoint every 500 training iterations
        if result['training_iteration'] % 50 == 0:
            checkpoint = trainer.save()
            print(checkpoint)


from ray import tune

result = tune.run(
    train,
    config=conf,
    resources_per_trial={
        "cpu": 1,
        "gpu": 1,
    },
    stop={"timesteps_total": 1e7}
)
