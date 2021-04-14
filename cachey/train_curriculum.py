#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

from animalai.envs.gym.environment import AnimalAIGym

from cachey.cache_model import *

content_root = Path(__file__).parent.parent


class UnityEnvWrapper(gym.Env):
    def __init__(self, env_config):
        self.vector_index = env_config.vector_index
        self.worker_index = env_config.worker_index
        self.worker_id = env_config.worker_index
        if "unity_worker_id" in env_config:
            self.worker_id += env_config["unity_worker_id"]
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


def test_train():
    # Load the first checkpoint from arena0-14runs
    # load_path = 'C:\\Users\\benja\\ray_results\\arena0-14times\\checkpoint_100\\checkpoint-100'
    # load_phase = 0
    # load_lesson_length = 100
    load_path = 'C:\\Users\\benja\\ray_results\\PPO_unity_env_2021-04-12_09-34-59ai5d4rj4\\checkpoint_1100\\checkpoint-1100'
    load_phase = 2
    load_lesson_length = 841

    phase = 0
    lesson_length = 0

    phase = load_phase
    lesson_length = load_lesson_length

    conf["env_config"] = {
        "arena_to_train": str(
            (content_root / 'examples/configurations/curriculum' / arena_configurations[phase]).absolute()),
    }
    trainer = PPOTrainer(config=conf, env="unity_env")
    print('Loading checkpoint ', load_path)
    trainer.restore(load_path)

    print("HERE")
    while True:
        result = trainer.train()
        lesson_length += 1
        # reporter(**result)
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


def test_eval():
    import pandas as pd
    trained = True
    df = pd.DataFrame()
    animalai_exe = content_root / "examples/env/AnimalAI"
    arena_root = content_root / 'competition_configurations'
    test_arena = str((arena_root / '1-1-1.yml').absolute())
    conf["env_config"] = {
        "arena_to_train": test_arena,
    }
    trainer = PPOTrainer(config=conf, env="unity_env")
    if trained:
        trainer.restore('C:\\Users\\benja\\ray_results\\arena2\\checkpoint_1144\\checkpoint-1144')
    # test env
    env = AnimalAIGym(
        environment_filename=str(animalai_exe.absolute()),
        worker_id=700,
        flatten_branched=True,
        uint8_visual=True,
        arenas_configurations=ArenaConfig(test_arena)
    )
    try:
        for i in [1, 2, 3, 4, 5, 6, 7]:
            arena_name = f'1-{i}-1.yml'
            arena_configuration = arena_root / arena_name
            env._env.reset(ArenaConfig(str(arena_configuration.absolute())))

            # Load trainer with test env too, although we won't be rolling out in it
            # conf["record_env"] = True
            conf["env_config"] = {
                "unity_worker_id": i * 10,
                "arena_to_train": str(arena_configuration.absolute()),
            }
            state = trainer.get_policy().model.get_initial_state()

            # _ = rollout(env, state, trainer, 5000) # rollout a few times to get a good recurrent state
            # episode_rewards = rollout(env, state, trainer, 6000)
            episode_rewards = rollout_eps(env, state, trainer, 100)

            # env.close()
            # trainer.cleanup()
            results = {
                "iteration": i,
                "arena": str(arena_configuration),
                "trained": trained,
                "episodes_ran": len(episode_rewards),
                "mean_eps_rew": np.mean(episode_rewards),
                "std_eps_rew": np.std(episode_rewards),
                "eps_success": len([j for j in episode_rewards if j >= 0])
            }
            df = df.append(results, ignore_index=True)
            for k, v in results.items():
                print(k + ':', v)
    finally:
        print(df)
        df.to_csv(str((content_root / "results.csv").absolute()))


def rollout(env, state, trainer, timesteps_to_run):
    timesteps_passed = 0
    episode_rewards = []
    while timesteps_passed < timesteps_to_run:
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = trainer.compute_action(obs, state)
            action, state = action[0], action[1]
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            timesteps_passed += 1
        episode_rewards.append(episode_reward)
    return episode_rewards


def rollout_eps(env, state, trainer, eps_to_run):
    timesteps_passed = 0
    episode_rewards = []
    while len(episode_rewards) < eps_to_run:
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = trainer.compute_action(obs, state)
            action, state = action[0], action[1]
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            timesteps_passed += 1
        episode_rewards.append(episode_reward)
    return episode_rewards
