import gym
import ray
from ray.rllib.agents.a3c import A3CTrainer
from ray.tune.logger import pretty_print
from stable_baselines3.common.utils import set_random_seed

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
    trainer = get_trainer(cfg)
    for i in range(1000):
        result = trainer.train()
        print(pretty_print(result))

        if i > 0 and i % 100 == 0:
            save_checkpoint(trainer)
        if result["timesteps_total"] >= cfg["timesteps"]:
            print("done training")
            save_checkpoint(trainer)
            break


def get_trainer(cfg):
    trainer = A3CTrainer(config={
        "env": cfg["env_id"],
        "num_gpus": 0,
        "num_workers": 1,
        "framework": 'torch',
    })
    return trainer


def save_checkpoint(trainer):
    """
    Save a checkpoint of the trainer
    """
    checkpoint = trainer.save()
    print("checkpoint saved at", checkpoint)
    return checkpoint


def main():
    cfg = get_cfg()

    if cfg["train"]:
        train(cfg)


if __name__ == '__main__':
    ray.init()
    main()
