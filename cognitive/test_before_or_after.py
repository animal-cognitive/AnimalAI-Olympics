from cognitive.primitive_arena import BeforeOrBehind

def test_modify_yaml():
    bob = BeforeOrBehind()
    bob.modify_yaml_and_dump(30, 32, 0)


def test_play_modified_yaml():
    bob = BeforeOrBehind()
    arena_config = bob.modify_yaml_and_dump(10, 13, 0)
    bob.play(arena_config)


def test_generate_config():
    bob = BeforeOrBehind()
    arena_config, con = bob.generate_config()
    print(con)
    bob.play(arena_config)
