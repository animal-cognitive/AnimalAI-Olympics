from animalai.envs.arena_config import ArenaConfig
from cognitive.primitive_arena import Occlusion


def test_play():
    bob = Occlusion()
    bob.play(bob.template_arena_config)


def test_modify_yaml():
    bob = Occlusion()
    bob.modify_yaml(30, 32, 0)


def test_play_modified_yaml():
    bob = Occlusion()
    arena_config=bob.modify_yaml(10, 13, 20)
    bob.play(arena_config)


def test_generate_config():
    bob = Occlusion()
    arena_config, con = bob.generate_config()
    print(con)
    bob.play(arena_config)
