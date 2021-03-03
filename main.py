import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecFrameStack

from config import get_cfg


def make_env(cfg, rank=0, seed=0):
    """
    Subproc to make one of many vectorized environments
    """

    def _init():
        env = gym.make(cfg["env_id"])
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def train(cfg):
    """
    Train a model with the given config
    """
    # Needed to use this fix https://stackoverflow.com/a/64104353 bruh am I getting hacked
    env = make_atari_env(cfg["env_id"], n_envs=cfg["n_cpu"], seed=cfg["seed"])
    env = VecFrameStack(env, n_stack=4)
    model = A2C('CnnPolicy', env, tensorboard_log=cfg["model_pkl"], verbose=True)
    model.learn(total_timesteps=cfg["timesteps"])
    model.save(cfg["model_pkl"])


def evaluate(cfg):
    """
    Test a model and return statistics
    """
    env = make_atari_env(cfg["env_id"])
    env = VecFrameStack(env, n_stack=4)
    model = A2C.load(cfg["model_pkl"], env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=cfg["n_eval_episodes"], render=cfg["render"])
    print(f'{mean_reward}+={std_reward}')


def main():
    cfg = get_cfg()

    if cfg["train"]:
        train(cfg)

    evaluate(cfg)


if __name__ == '__main__':
    main()
