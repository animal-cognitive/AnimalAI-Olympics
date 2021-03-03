import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

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
    env = SubprocVecEnv([make_env(cfg, i, cfg["seed"]) for i in range(cfg["n_cpu"])])
    model = PPO('MlpPolicy', env, tensorboard_log=cfg["model_pkl"], verbose=True)
    model.learn(total_timesteps=cfg["timesteps"])
    model.save()


def evaluate(cfg):
    """
    Test a model and return statistics
    """
    model = PPO.load(cfg["model_pkl"])
    env = Monitor(gym.make(cfg["env_id"]))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=cfg["n_eval_episodes"], render=cfg["render"])
    print(f'{mean_reward}+={std_reward}')


def main():
    cfg = get_cfg()

    if cfg["train"]:
        train(cfg)

    evaluate(cfg)


if __name__ == '__main__':
    main()
