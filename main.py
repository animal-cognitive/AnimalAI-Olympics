import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

from config import get_cfg
from custom_model import MyFCForwardModel

# Register custom model so that we can give the ID to the policy trainer
ModelCatalog.register_custom_model("my_pytorch_model", MyFCForwardModel)


def train(cfg):
    """
    Train a model with the given config
    """

    # Transform our config to a Ray config
    ray_cfg = {
        "env": cfg["env_id"],
        "num_gpus": 0,
        "num_workers": 1,
        "framework": 'torch',
        "model": {
            "custom_model": "my_pytorch_model",
        },
        "log_level": 'INFO',
    }

    """Use Ray Tune to train an agent.
    Configure cfg["stop"] for what criterion to use to stop training (e.g. timesteps_total, episode_reward_mean).
    https://docs.ray.io/en/master/tune/index.html
    The results will be saved in ./ray_results and can be loaded by instantiating an Analysis.
    https://docs.ray.io/en/master/tune/api_docs/analysis.html#analysis-tune-analysis
    """
    analysis = ray.tune.run(
        PPOTrainer,
        config=ray_cfg,
        local_dir=cfg["local_dir"],
        stop=cfg["stop"],
        checkpoint_freq=10,
        checkpoint_at_end=True)
    print(analysis)


def main():
    cfg = get_cfg()
    train(cfg)


if __name__ == '__main__':
    ray.init()
    main()
