import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

from config import get_cfg
from custom_model import *

# Register custom models so that we can give the ID to the policy trainer
ModelCatalog.register_custom_model("my_fc_model", MyFCForwardModel)
ModelCatalog.register_custom_model("my_rnn_model", MyRNNModel)
ModelCatalog.register_custom_model("my_convgru_model", MyConvGRUModel)


def train(cfg):
    """
    Train a model with the given config
    """

    # Added this flag so I can debug easier (without having to reason about Tune)
    simple_train = False

    if simple_train:
        trainer = PPOTrainer(env=cfg["env_id"], config={"model": {
            "custom_model": 'my_convgru_model',
        },
            "framework": 'torch',
            "num_workers": 0,
        })
        for i in range(1000):
            print('training', i)
            # Perform one iteration of training the policy with PPO
            result = trainer.train()
            print(pretty_print(result))
    else:
        # Transform our config to a Ray config
        ray_cfg = {
            "env": cfg["env_id"],
            "num_gpus": 0,
            "num_workers": 0,
            "framework": 'torch',
            "model": {
                "custom_model": 'my_convgru_model',
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
    ray.init(include_dashboard=False)
    main()
