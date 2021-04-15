import ray
from mlagents_envs.exception import UnityTimeOutException

from cachey.load_trained_model import load_trainer
from cognitive.dir import DIRWrapper, rotate_180
from cognitive.primitive_arena import Occlusion, GymFactory


def test_occlusion_generate_config():
    ray.init()
    arena = Occlusion()
    arena_config, settings = arena.generate_config()
    Env = GymFactory(arena_config, 1234)
    env = None
    trainer = load_trainer()
    env_config = trainer.config
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

    obs = env.reset()
    visualize_observation(obs)

    x, y = [], []

    obs = env.reset()
    # sum_reward = 0
    state = trainer.get_policy().model.get_initial_state()
    # state = [s.unsqueeze(0) for s in state]
    for i in range(3):
        # trainer.compute_action()
        # prep = get_preprocessor(env.observation_space)(env.observation_space)
        # obs = torch.stack([torch.from_numpy(prep.transform(env.observation_space.sample())) for _ in range(100)],
        #                   dim=0)
        action, state, _ = trainer.compute_action(obs, state)
        obs, reward, done, info = env.step(action)
        # assert: the agent can see the wall and the ball
        if done:
            break
    visualize_observation(obs, "1.png")

    for i in range(100):
        action = 0
        obs, reward, done, info = env.step(action)
        # assert: the ball is behind the wall
    visualize_observation(obs, "2.png")
    try:
        env.close()
    except AttributeError:
        pass

    ray.shutdown()
    

def test_rotation_generate_config():
    ray.init()
    arena = Occlusion()
    arena_config, settings = arena.generate_config()
    Env = GymFactory(arena_config, 1234)
    env = None
    trainer = load_trainer()
    env_config = trainer.config
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

    obs = env.reset()
    visualize_observation(obs)

    x, y = [], []

    obs = env.reset()
    # sum_reward = 0
    state = trainer.get_policy().model.get_initial_state()
    # state = [s.unsqueeze(0) for s in state]
    for i in range(3):
        # trainer.compute_action()
        # prep = get_preprocessor(env.observation_space)(env.observation_space)
        # obs = torch.stack([torch.from_numpy(prep.transform(env.observation_space.sample())) for _ in range(100)],
        #                   dim=0)
        action, state, _ = trainer.compute_action(obs, state)
        obs, reward, done, info = env.step(action)
        # assert: the agent can see the wall and the ball
        if done:
            break
    visualize_observation(obs, "1.png")

    for ac in rotate_180():
        obs, reward, done, info = env.step(4)
        # assert: the ball is behind the wall
    visualize_observation(obs, "2.png")
    try:
        env.close()
    except AttributeError:
        pass

    ray.shutdown()

def visualize_observation(obs, save_name = "vis.png"):
    from PIL import Image
    import numpy as np
    img = Image.fromarray(obs, "RGB")
    # img.show()
    img.save(save_name)
    # img.close()