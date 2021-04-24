import colorsys
import copy
import math
import random
import warnings
from pathlib import Path

import ray
import ray.rllib.agents.a3c as a3c
import torch
import yaml
from gym_unity.envs import UnityGymException

from animalai.envs.arena_config import ArenaConfig
from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.gym.environment import AnimalAIGym
from mlagents_envs.exception import UnityCommunicationException, UnityTimeOutException, UnityWorkerInUseException
from ray.rllib.agents import ppo
from ray.rllib.models.preprocessors import get_preprocessor
from tqdm import tqdm

from cognitive.dir import DIRWrapper, rotate_180, remove_goal, DIRDataset
import numpy as np


# class RayAIIGym(AnimalAIGym):
#     def __init__(self, env_config, arena_config):
#         super(RayAIIGym, self).__init__(worker_id=env_config.worker_index,
#                                         environment_filename='../examples/env/AnimalAI',
#                                         arenas_configurations=arena_config)
# global init
# init = 1000


def GymFactory(arena_config, worker_id=None):
    class RayAAIGym(AnimalAIGym):
        def __init__(self, env_config):
            # print("Opened")
            if worker_id is not None:
                wid = worker_id
            else:
                wid = env_config.worker_index
            while True:
                try:
                    super(RayAAIGym, self).__init__(worker_id=wid,
                                                    environment_filename='examples/env/AnimalAI',
                                                    arenas_configurations=arena_config,
                                                    flatten_branched=True,
                                                    uint8_visual=True,
                                                    )
                except UnityWorkerInUseException:
                    wid += 1
                except (UnityTimeOutException, UnityGymException) as e:
                    pass
                else:
                    break

    return RayAAIGym

class ArenaConfigRegenerationRequest(Exception):
    def __init__(self):
        super(ArenaConfigRegenerationRequest, self).__init__()


class ArenaManager:
    """
    Outputs arena yml
    """

    def __init__(self, base_port=1000):
        self.template_path = None
        self.port_counter = 0
        self.base_port = base_port

    def dump_temp_config(self, arena_config, path):
        st = yaml.dump(arena_config)
        with open(path, "w") as f:
            f.write(st)

    def play(self, arena_config):
        wid = 1
        environment = None
        while True:
            try:
                environment = AnimalAIEnvironment(
                    file_name='examples/env/AnimalAI',
                    worker_id=wid,
                    base_port=5005,
                    arenas_configurations=arena_config,
                    play=True,
                )
            except UnityCommunicationException:
                # you'll end up here if you close the environment window directly
                # always try to close it from script
                if environment:
                    environment.close()
            except UnityWorkerInUseException:
                wid += 1
                pass
            else:
                break

        if environment:
            environment.close()  # takes a few seconds

    def generate_config(self):
        pass

    def modify_yaml(self, *args, **kwargs):
        pass

    def collect_dir(self, trainer, ds_size_per_env, arena_config, settings, env_config, initial_steps=10):
        pass

    # def interact(self):
    #     # TODO get the agent policy and weights here
    #     # inject interactions here
    #     ray.init(include_dashboard=False)
    #     config = ppo.DEFAULT_CONFIG.copy()
    #     # checkpoint_path = checkpoints[0][0]
    #     agent = a3c.A3CTrainer(config=config)
    #     # agent.restore('D:\\Git\\AnimalAI-Olympics\\log\\PPO\\PPO_CartPole-v0_314b5_00000_0_lr=0.0001_2021-03-19_15-32-15\\checkpoint_000005\\checkpoint-5')
    #     agent.restore(
    #         'D:\\Git\\AnimalAI-Olympics\\log\\A3C\\A3C_CartPole-v0_b1b96_00000_0_lr=0.0001_2021-03-19_15-50-10\\checkpoint_000040\\checkpoint-40')
    #     arena_config, _ = self.generate_config()
    #     env = RayAIIGym(config, arena_config)
    #     obs = env.reset()
    #     sum_reward = 0
    #     for i in range(1000):
    #         action = agent.compute_action(obs)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if done:
    #             print(agent.get_policy().model)
    #             print(agent.get_policy())
    #             print(agent.get_policy().model.base_model.summary())
    #             break

    def generate_dir_dataset(self, agent, env_config, size=1000):
        dir, labels = [], []
        for i in range(size):
            d, l = self.collect_dir(agent, env_config)
            dir += d
            labels += l
        return DIRDataset(dir, labels)

    def get_port(self):
        self.base_port += 1
        return self.base_port


class BeforeOrBehind(ArenaManager):
    def __init__(self):
        super(BeforeOrBehind, self).__init__()
        self.template_path = Path("cognitive/before_or_behind_template.yml")
        self.template_arena_config = ArenaConfig(self.template_path)

    def generate_config(self):
        # the object before will be in [
        fore = randdouble(20, 34)
        back = randdouble(1, 5) + fore

        before = random.random() < 0.5
        if before:
            goal_z_pos = fore
            wall_z_pos = back
        else:
            goal_z_pos = back
            wall_z_pos = fore

        if random.random() < 0.5:
            agent_rotation = randdouble(0, 30)
        else:
            agent_rotation = randdouble(330, 360)

        arena_config = self.modify_yaml(goal_z_pos, wall_z_pos, agent_rotation)
        return arena_config, {"before": before,
                              "goal_z_pos": goal_z_pos,
                              "wall_z_pos": wall_z_pos,
                              "agent_rotation": agent_rotation}

    def modify_yaml(self, goal_z_pos, wall_z_pos, agent_rotation):
        """
        The strategy is to shuffle item's z

        :param goal_z_pos:
        :param wall_z_pos:
        :param agent_rotation:
        :return:
        """
        arena_config = copy.deepcopy(self.template_arena_config)

        # others are by default randomized
        items = arena_config.arenas[-1].items
        wall = items[2]
        wall.positions[0].z = wall_z_pos

        good_goal = items[1]
        good_goal.positions[0].z = goal_z_pos

        agent = items[0]
        agent.rotations[0] = agent_rotation
        return arena_config

    def collect_dir(self, trainer, ds_size_per_env, arena_config, settings, env_config, initial_steps=10):
        """
        Procedure:
        1) Start agent
        2) Roll 10 steps
        3) Collect DIR
        4) Label determined by the arena configuration

        :param agent:
        :return:
        (DIR, is_before)
        """
        port = self.get_port()
        Env = GymFactory(arena_config, port)
        env = None
        while True:
            try:
                env = Env(env_config)
            except UnityTimeOutException:
                if env:
                    try:
                        env.close()
                    except AttributeError:
                        pass
            else:
                break
        x = []
        y = []
        # batch_size = 128
        dir = DIRWrapper(trainer.get_policy().model)

        try:
            for _ in tqdm(range(ds_size_per_env)):
                obs = env.reset()
                state = trainer.get_policy().model.get_initial_state()
                for i in range(initial_steps):
                    action, state, _ = trainer.compute_action(obs, state)
                    obs, reward, done, info = env.step(action)
                x.append(dir.get_dir())
                y.append(settings["before"])
            return x, y
        finally:
            try:
                env.close()
            except AttributeError:
                pass


class Occlusion(ArenaManager):
    def __init__(self):
        super(Occlusion, self).__init__()
        self.template_path = Path("cognitive/occlusion_template.yml")
        self.template_arena_config = ArenaConfig(self.template_path)

    def generate_config(self):
        """

        :return: arena_config, settings
        """
        agent_z = randdouble(1, 7)

        # remove left rotation because the ball is not visible
        # if random.random() < 0.5:
        agent_rotation = randdouble(20, 50)
        # else:
        # agent_rotation = randdouble(330, 360)

        wall_length = randdouble(10, 30)

        arena_config = self.modify_yaml(agent_z, agent_rotation, wall_length)
        return arena_config, {"agent_z": agent_z,
                              "agent_rotation": agent_rotation,
                              "wall_length": wall_length}

    def modify_yaml(self, agent_z, agent_rotation, wall_length):
        """
        The strategy is to shuffle item's z

        :param goal_z_pos:
        :param wall_z_pos:
        :param agent_rotation:
        :return:
        """
        arena_config = copy.deepcopy(self.template_arena_config)

        # others are by default randomized
        items = arena_config.arenas[0].items

        wall = items[2]
        wall.sizes[0].z = wall_length
        wall.positions[0].x = wall_length / 2
        agent = items[0]
        agent.rotations[0] = agent_rotation
        agent.positions[0].z = agent_z
        return arena_config

    def collect_dir(self, trainer, ds_size_per_env, arena_config, settings, env_config, initial_steps=0):
        """
        Procedure:
        1) Start agent
        2) Roll 10 steps
        3) Freeze: Do not allow agent to take action
        4) Collect DIR at different time steps
        5) Label determined by occlusion

        :param trainer:
        :return:
        (DIR, is_occluded)
        """
        Env = GymFactory(arena_config, self.get_port())
        env = None
        while True:
            try:
                env = Env(env_config)
            except UnityTimeOutException:
                if env:
                    try:
                        env.close()
                    except AttributeError:
                        pass
            else:
                break
        dir = DIRWrapper(trainer.get_policy().model)
        x, y = [], []
        try:
            for _ in tqdm(range(ds_size_per_env)):
                obs = env.reset()
                # sum_reward = 0
                state = trainer.get_policy().model.get_initial_state()
                # state = [s.unsqueeze(0) for s in state]
                for i in range(initial_steps):
                    # trainer.compute_action()
                    # prep = get_preprocessor(env.observation_space)(env.observation_space)
                    # obs = torch.stack([torch.from_numpy(prep.transform(env.observation_space.sample())) for _ in range(100)],
                    #                   dim=0)
                    action, state, _ = trainer.compute_action(obs, state)
                    obs, reward, done, info = env.step(action)
                    if done:
                        break
                if not goal_visible(obs):
                    raise ArenaConfigRegenerationRequest()

                visible_dir = dir.get_dir()
                # vis = obs.copy()
                # visualize_observation(vis, "visible.png")
                for i in range(150):
                    _, state, _ = trainer.compute_action(obs, state)
                    obs, reward, done, info = env.step(0)
                # assert: the ball is behind the wall
                if goal_visible(obs):
                    raise ArenaConfigRegenerationRequest()

                occluded_dir = dir.get_dir()
                # visualize_observation(obs, "occluded.png")

                x += [visible_dir, occluded_dir]
                y += [False, True]
            return x, y
        finally:
            try:
                env.close()
            except AttributeError:
                pass


class Rotation(ArenaManager):
    def __init__(self):
        super(Rotation, self).__init__()
        # this should be randomized, depending on the arenas that the model excels at
        # if a model does well in an environment, take it out, claim successful
        # A/B test: DIR after finding target, DIR with same trajectory but no object
        self.template_path = Path("competition_configurations/3-1-2.yml")
        self.template_arena_config = ArenaConfig(self.template_path)

    def generate_config(self):
        agent_x = randdouble(10, 20)

        if random.random() < 0.5:
            agent_rotation = randdouble(0, 15)
        else:
            agent_rotation = randdouble(345, 360)

        wall_length = randdouble(10, 30)

        arena_config = self.modify_yaml(agent_x, agent_rotation, wall_length)
        return arena_config, {"agent_x": agent_x,
                              "agent_rotation": agent_rotation,
                              "wall_length": wall_length}

    def modify_yaml(self, agent_x, agent_rotation, wall_length):
        arena_config = copy.deepcopy(self.template_arena_config)

        # others are by default randomized
        items = arena_config.arenas[0].items

        wall = items[0]
        wall.sizes[0].x = wall_length
        agent = items[2]
        agent.rotations[0] = agent_rotation
        agent.positions[0].x = agent_x
        return arena_config

    def collect_dir(self, trainer, ds_size_per_env, arena_config, settings, env_config, initial_steps=1000):
        """
        Procedure:
        1) Start agent, record action
        2) Freely find the goal
        3) When the goal is reached, rotate the agent.
        4) Collect DIR for positive
        5) Replay the same arena, same trajectory, without the goal
        6) Rotate the agent with the same direction
        7) Collect DIR for negative

        :param agent:
        :return:
        (DIR, is_removed)
        """
        Env = GymFactory(arena_config, self.get_port())
        env = None
        while True:
            try:
                env = Env(env_config)
            except UnityTimeOutException:
                if env:
                    try:
                        env.close()
                    except AttributeError:
                        pass
            else:
                break

        # invisible
        removed_ac = remove_goal(arena_config)
        Env = GymFactory(removed_ac, self.get_port())
        removed_env = None
        while True:
            try:
                removed_env = Env(env_config)
            except UnityTimeOutException:
                if removed_env:
                    try:
                        removed_env.close()
                    except AttributeError:
                        pass
            else:
                break

        dir = DIRWrapper(trainer.get_policy().model)

        try:
            x = []
            y = []
            for _ in tqdm(range(ds_size_per_env)):
                obs = env.reset()
                actions = []
                states = []
                state = trainer.get_policy().model.get_initial_state()
                # state = [s.unsqueeze(0) for s in state]
                states.append(state)
                done = False
                for i in range(initial_steps):
                    action, state, _ = trainer.compute_action(obs, state)
                    # state = dir.dir[1]
                    # state = [s.squeeze(0) for s in state]

                    states.append(state)
                    actions.append(action)
                    obs, reward, done, info = env.step(action)
                    if done:
                        break
                if not done:
                    raise RuntimeError("The agent failed to find the goal")

                # visible
                # obs = env.reset()
                # for idx, ac in enumerate(actions[:-1]):
                #     state = states[idx]
                #     _, _, _ = trainer.compute_action(obs, state)
                #     obs, reward, done, info = env.step(ac)

                state = states[-1]
                for ac in rotate_180():
                    _, state, _ = trainer.compute_action(obs, state)
                    obs, reward, done, info = env.step(ac)
                    # state = dir.dir[1]
                    # state = [s.squeeze(0) for s in state]
                    # obs, reward, done, info = env.step(ac)

                visible_dir = dir.get_dir()

                obs = removed_env.reset()
                for idx, ac in enumerate(actions[:-1]):
                    state = states[idx]
                    _, _, _ = trainer.compute_action(obs, state)
                    obs, reward, done, info = env.step(ac)

                state = states[-1]
                for ac in rotate_180():
                    _, state, _ = trainer.compute_action(obs, state)
                    obs, reward, done, info = env.step(ac)
                    # state = dir.dir[1]
                    # state = [s.squeeze(0) for s in state]
                    # obs, reward, done, info = env.step(ac)

                # for idx, ac in enumerate(actions[:-1]):
                #     state = states[idx]
                #     action = trainer.get_policy().compute_actions(obs_batch=np.expand_dims(obs, axis=0),
                #                                                   state_batches=state)
                #     obs, reward, done, info = env.step(ac)
                #
                # state = states[-1]
                # for ac in rotate_180():
                #     action = trainer.get_policy().compute_actions(obs_batch=np.expand_dims(obs, axis=0),
                #                                                   state_batches=state)
                #     state = dir.dir[1]
                #     state = [s.squeeze(0) for s in state]
                #     obs, reward, done, info = env.step(ac)

                removed_dir = dir.get_dir()
                x += [visible_dir, removed_dir]
                y += [False, True]
            return x, y
        finally:
            try:
                env.close()
            except AttributeError:
                pass
            try:
                removed_env.close()
            except AttributeError:
                pass


def randdouble(a, b):
    return a + (b - a) * random.random()


def visualize_observation(obs, save_name="vis.png"):
    from PIL import Image
    import numpy as np
    img = Image.fromarray(obs, "RGB")
    # img.show()
    img.save(save_name)
    # img.close()


def goal_visible(obs):
    green = colorsys.rgb_to_hls(129, 191, 65)
    for i in obs:
        for pixel in i:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hls = colorsys.rgb_to_hls(*pixel)
                if abs(hls[0] - green[0]) < 0.03:
                    return True
    return False


def plot_obs(obs):
    from matplotlib import pyplot as plt
    plt.imshow(obs)
    plt.show()
