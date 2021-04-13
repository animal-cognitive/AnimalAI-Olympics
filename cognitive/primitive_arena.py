import copy
import random
from pathlib import Path

import ray
import ray.rllib.agents.a3c as a3c
import yaml
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.gym.environment import AnimalAIGym
from mlagents_envs.exception import UnityCommunicationException
from ray.rllib.agents import ppo

from cognitive.dir import DIRWrapper, no_action, rotate_180, remove_goal, DIRDataset


class RayAIIGym(AnimalAIGym):
    def __init__(self, env_config, arena_config):
        super(RayAIIGym, self).__init__(worker_id=env_config.worker_index,
                                        environment_filename='../examples/env/AnimalAI',
                                        arenas_configurations=arena_config)


class ArenaManager:
    """
    Outputs arena yml
    """

    def __init__(self):
        self.template_path = None

    def dump_temp_config(self, arena_config, path):
        st = yaml.dump(arena_config)
        with open(path, "w") as f:
            f.write(st)

    def play(self, arena_config):
        try:
            environment = AnimalAIEnvironment(
                file_name='examples/env/AnimalAI',
                base_port=5005,
                arenas_configurations=arena_config,
                play=True,
            )
        except UnityCommunicationException:
            # you'll end up here if you close the environment window directly
            # always try to close it from script
            environment.close()

        if environment:
            environment.close()  # takes a few seconds

    def generate_config(self):
        pass

    def modify_yaml(self, *args, **kwargs):
        pass

    def collect_dir(self, agent, env_config):
        pass

    def interact(self):
        # TODO get the agent policy and weights here
        # inject interactions here
        ray.init(include_dashboard=False)
        config = ppo.DEFAULT_CONFIG.copy()
        # checkpoint_path = checkpoints[0][0]
        agent = a3c.A3CTrainer(config=config)
        # agent.restore('D:\\Git\\AnimalAI-Olympics\\log\\PPO\\PPO_CartPole-v0_314b5_00000_0_lr=0.0001_2021-03-19_15-32-15\\checkpoint_000005\\checkpoint-5')
        agent.restore(
            'D:\\Git\\AnimalAI-Olympics\\log\\A3C\\A3C_CartPole-v0_b1b96_00000_0_lr=0.0001_2021-03-19_15-50-10\\checkpoint_000040\\checkpoint-40')
        arena_config, _ = self.generate_config()
        env = RayAIIGym(config, arena_config)
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

    def generate_dir_dataset(self, agent, env_config, size=1000):
        dir, labels = [], []
        for i in range(size):
            d, l = self.collect_dir(agent, env_config)
            dir += d
            labels += l
        return DIRDataset(dir, labels)


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

    def collect_dir(self, agent, env_config, initial_steps=10):
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
        arena_config, inter = self.generate_config()
        env = RayAIIGym(env_config, arena_config)
        obs = env.reset()

        dir = DIRWrapper(agent)

        for i in range(initial_steps):
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
        return [dir.get_dir()], [inter["before"]]


class Occlusion(ArenaManager):
    def __init__(self):
        super(Occlusion, self).__init__()
        self.template_path = Path("cognitive/occlusion_template.yml")
        self.template_arena_config = ArenaConfig(self.template_path)

    def generate_config(self):
        agent_z = randdouble(1, 7)

        if random.random() < 0.5:
            agent_rotation = randdouble(0, 30)
        else:
            agent_rotation = randdouble(330, 360)

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

    def collect_dir(self, agent, env_config, initial_steps=10):
        """
        Procedure:
        1) Start agent
        2) Roll 10 steps
        3) Freeze: Do not allow agent to take action
        4) Collect DIR at different time steps
        5) Label determined by occlusion

        :param agent:
        :return:
        (DIR, is_occluded)
        """
        arena_config, inter = self.generate_config()
        env = RayAIIGym(env_config, arena_config)
        obs = env.reset()

        dir = DIRWrapper(agent)

        # sum_reward = 0
        for i in range(initial_steps):
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            # assert: the agent can see the wall and the ball

        visible_dir = dir.get_dir()

        for i in range(50):
            action = no_action()
            obs, reward, done, info = env.step(action)
            # assert: the ball is behind the wall

        occluded_dir = dir.get_dir()
        return [visible_dir, occluded_dir], [False, True]


class Rotation(ArenaManager):
    def __init__(self):
        super(Rotation, self).__init__()
        # this should be randomized, depending on the arenas that the model excels at
        # if a model does well in an environment, take it out, claim successful
        # A/B test: DIR after finding target, DIR with same trajectory but no object
        self.template_path = Path("competition_configurations/3-1-2.yml")
        self.template_arena_config = ArenaConfig(self.template_path)

    def generate_config(self):
        agent_x = randdouble(10, 30)

        if random.random() < 0.5:
            agent_rotation = randdouble(0, 30)
        else:
            agent_rotation = randdouble(330, 360)

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

    def collect_dir(self, agent, env_config, initial_steps=1000):
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
        arena_config, inter = self.generate_config()
        env = RayAIIGym(env_config, arena_config)
        obs = env.reset()

        dir = DIRWrapper(agent)
        actions = []

        for i in range(initial_steps):
            action = agent.compute_action(obs)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            if done:
                print(agent.get_policy().model)
                print(agent.get_policy())
                print(agent.get_policy().model.base_model.summary())
                break

        # visible
        obs = env.reset()
        for ac in actions[:-1]:
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(ac)

        for ac in rotate_180():
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(ac)

        visible_dir = dir.get_dir()

        removed_ac = remove_goal(arena_config)
        env = RayAIIGym(env_config, removed_ac)
        obs = env.reset()

        for ac in actions[:-1]:
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(ac)

        for ac in rotate_180():
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(ac)

        removed_dir = dir.get_dir()

        return [visible_dir, removed_dir], [False, True]


def randdouble(a, b):
    return a + (b - a) * random.random()
