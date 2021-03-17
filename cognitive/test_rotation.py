from animalai.envs.arena_config import ArenaConfig
from cognitive.primitive_arena import Rotation


def test_play():
    bob = Rotation()
    bob.play(bob.template_arena_config)


def test_modify_yaml():
    bob = Rotation()
    bob.modify_yaml(30, 32, 0)


def test_play_modified_yaml():
    bob = Rotation()
    arena_config=bob.modify_yaml(10, 13, 20)
    bob.play(arena_config)


def test_generate_config():
    bob = Rotation()
    arena_config, con = bob.generate_config()
    print(con)
    bob.play(arena_config)
