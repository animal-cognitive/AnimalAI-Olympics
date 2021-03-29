import subprocess

from animalai_train.run_options_aai import RunOptionsAAI
from animalai_train.run_training_aai import run_training_aai
from mlagents.trainers.trainer_util import load_config

from cognitive.primitive_arena import GymFactory, Occlusion
import ray.rllib.agents.ppo as ppo


def main():
    with open("examples/configurations/training_configurations/train_ml_agents_config_ppo.yaml") as f:
        print(f.read())
    import warnings
    warnings.filterwarnings('ignore')
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    trainer_config_path = (
        "examples/configurations/training_configurations/train_ml_agents_config_ppo.yaml"
    )
    environment_path = "examples/env/AnimalAI"
    curriculum_path = "examples/configurations/curriculum"
    run_id = "self_control_curriculum"
    base_port = 5005
    number_of_environments = 4
    number_of_arenas_per_environment = 8

    args = RunOptionsAAI(
        trainer_config=load_config(trainer_config_path),
        env_path=environment_path,
        run_id=run_id,
        base_port=base_port,
        num_envs=number_of_environments,
        curriculum_config=curriculum_path,
        n_arenas_per_env=number_of_arenas_per_environment,
    )
    import os
    # logging.getLogger('tensorflow').disabled = True

    logs_dir = "summaries/"
    os.makedirs(logs_dir, exist_ok=True)
    subprocess.run(f"tensorboard --logdir {logs_dir}")

    run_training_aai(0, args)


def load_agent_get_action(environment):
    agent_groups = environment.get_agent_groups()
    agent_group_spec = environment.get_agent_group_spec(agent_groups[0])

    print(f'Here we only have {len(agent_groups)} agent group: {agent_groups[0]}')
    print(f'''\nAnd you can get their caracteristics: \n
    visual observations shape: {agent_group_spec.observation_shapes[0]}
    vector observations shape (velocity): {agent_group_spec.observation_shapes[1]}
    actions are discrete: {agent_group_spec.action_type}
    actions have shape: {agent_group_spec.action_shape}
    ''')
    import numpy as np

    actions = [[0, 0]] * 50  # Do nothing until the lights come back on
    actions += [[1, 0]] * 40  # Go forward
    actions += [[0, 2]] * 15  # turn left
    actions += [[1, 0]] * 50  # go forward again

    agent_group = agent_groups[0]
    visual_observations = []
    velocity_observations = []
    rewards = []
    done = False
    step = 0

    while not done and step < len(actions):
        action = np.array(actions[step]).reshape(1, 2)

        environment.set_actions(agent_group=agent_group, action=action)
        environment.step()
        step_result = environment.get_step_result(agent_group)

        visual_observations.append(step_result.obs[0])
        velocity_observations.append(step_result.obs[1])
        done = step_result.done[0]
        rewards.append(step_result.reward[0])
        max_step_reached = step_result.max_step[0]
        step += 1
    environment.close()


def play():
    from animalai.envs.arena_config import ArenaConfig
    from animalai.envs.environment import AnimalAIEnvironment
    from mlagents_envs.exception import UnityCommunicationException

    environment = None
    try:
        environment = AnimalAIEnvironment(
            file_name='examples/env/AnimalAI',
            base_port=5005,
            arenas_configurations=ArenaConfig('competition_configurations/3-1-2.yml'),
            play=True,
        )
    except UnityCommunicationException:
        # you'll end up here if you close the environment window directly
        # always try to close it from script
        environment.close()


def what_actions():
    env_config = ppo.DEFAULT_CONFIG.copy()
    occ = Occlusion()
    arena_config, settings = occ.generate_config()
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

    Env = GymFactory(arena_config)

    env = Env(env_config)
    obs = env.reset()
    for i in range(1000):
        obs, reward, done, info = env.step([0, 2])

    # [0,0] is no action
    # [1,0] is forward
    # [2,0] is backward
    # [0,1] is turning left
    # [0,2] is turning right


if __name__ == '__main__':
    what_actions()
