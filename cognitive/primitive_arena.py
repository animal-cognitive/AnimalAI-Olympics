import copy
import random
from pathlib import Path

import yaml
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.environment import AnimalAIEnvironment
from mlagents_envs.exception import UnityCommunicationException


class ArenaManager:
    """
    Outputs arena yml
    """

    def __init__(self):
        self.template = None

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

        arena_config = self.modify_yaml_and_dump(goal_z_pos, wall_z_pos, agent_rotation)
        return arena_config, {"before": before,
                              "goal_z_pos": goal_z_pos,
                              "wall_z_pos": wall_z_pos,
                              "agent_rotation": agent_rotation}

    def modify_yaml_and_dump(self, goal_z_pos, wall_z_pos, agent_rotation):
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


class Occlusion(ArenaManager):
    def __init__(self):
        super(Occlusion, self).__init__()
        self.template_path = Path("cognitive/occlusion_template.yml")
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

        agent_z = randdouble(1, 7)

        if random.random() < 0.5:
            agent_rotation = randdouble(0, 30)
        else:
            agent_rotation = randdouble(330, 360)

        wall_length = randdouble(17, 23)

        arena_config = self.modify_yaml_and_dump(agent_z, agent_rotation, wall_length)
        return arena_config, {"agent_z": agent_z,
                              "agent_rotation": agent_rotation,
                              "wall_length": wall_length}

    def modify_yaml_and_dump(self, agent_z, agent_rotation, wall_length):
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
        wall.sizes[0].z = wall_length
        agent = items[0]
        agent.rotations[0] = agent_rotation
        agent.positions[0].z = agent_z
        return arena_config


def randdouble(a, b):
    return a + (b - a) * random.random()
