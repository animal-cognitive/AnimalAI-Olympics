from animalai.envs.arena_config import ArenaConfig
from cognitive.primitive_arena import Occlusion


def test_play():
    bob = Occlusion()
    bob.play(bob.template_arena_config)


def test_modify_yaml():
    bob = Occlusion()
    bob.modify_yaml_and_dump(30, 32, 0)


def test_play_modified_yaml():
    bob = Occlusion()
    bob.modify_yaml_and_dump(10, 13, 0)
    bob.play()


def test_generate_config():
    bob = Occlusion()
    con = bob.generate_config()
    print(con)
    bob.play()
