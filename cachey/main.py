import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

from cache_model import *
from config import get_cfg
from custom_model import *

# Register custom models so that we can give the ID to the policy trainer
# ModelCatalog.register_custom_model("my_fc_model", MyFCForwardModel)
ModelCatalog.register_custom_model("my_rnn_model", MyRNNModel)
ModelCatalog.register_custom_model("my_convgru_model", MyConvGRUModel)  # NOTE: Only works with image observations.
ModelCatalog.register_custom_model("my_cnn_rnn_model", MyCNNRNNModel)


def train(cfg):
    """
    Train a model with the given config
    """

    import torch
    print('CUDA available:', torch.cuda.is_available())
    ray.init(include_dashboard=False)

    # Transform our config to a Ray config
    ray_cfg = {
        "env": cfg["env_id"],
        "num_gpus": 1,
        "num_workers": 8,
        "framework": 'torch',
        "model": {
            "custom_model": 'my_cnn_rnn_model',
        },
        "log_level": 'WARN',
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
        keep_checkpoints_num=6,
        checkpoint_at_end=True)
    print(analysis)


def example_instrument(cfg):
    """Added a flag so I can debug easier (without having to reason about Tune)"""
    trainer = PPOTrainer(env=cfg["env_id"], config={"model": {
        "custom_model": 'my_rnn_model',
    },
        "framework": 'torch',
        "num_workers": 0,
    })
    model = trainer.get_policy().model

    """
    Prepare a dummy batch of 32 images (B x W x H), assuming one sequence.
    The dimensions of LSTM input and output are (B x T x F), for B=Batch, T=Time, F=Features.
    A batch is a set of sequences, where a sequence is a list of observations.
    Each sequence must be the same length, so Ray pads 0's to fill out shorter sequences to the same length.
    `seq_lens` is a list of the length (in steps) of each sequence.
    Example:
        obs: torch.tensor((32, 84, 84))
        seq_lens: [8, 6, 4, 8]
        input to model: torch.tensor((8, 4, 84, 84))
    All sequences must be length 8, so the second and third sequences are padded with 0's for the missing observations.
    """
    # This is how a batch of observations looks coming out of the environment.
    # 32 images, each 84 x 84 pixels
    obs = torch.zeros((32, 84, 84)).cuda()
    # This input dict would be generated by rllib in ModelV2.__call__
    # (which wraps our custom model's forward() function)
    input_dict = {
        "obs_flat": torch.flatten(obs, start_dim=1),
    }

    # Get a dummy initial LSTM state using the utility method built into our custom model.
    state = model.get_initial_state()
    # Unsqueeze to encode 1 sequence. This would be handled by rllib.
    state = [s.cuda().unsqueeze(dim=0) for s in state]

    # 1 sequence of length 32.
    seq_lens = torch.tensor([32])

    # Register hook to be called after forward()
    def print_dir_shape(self, input, output):
        """Example hook that just prints the input/output shapes."""
        in_features, in_state = input
        out_features, out_state = output
        # Expect (1 x 1 x 64) coming from FC layer into LSTM
        print('input features:', in_features.shape, 'previous state:', [s.shape for s in in_state])
        # Expect (1 x 1 x 256) coming out of LSTM. This is the DIR.
        print('DIR output:', out_features.shape, 'current state:', [s.shape for s in out_state])

    model.lstm.register_forward_hook(print_dir_shape)

    # Pass observations to our model
    model.forward(input_dict, state, seq_lens)


def main():
    cfg = get_cfg()
    # example_instrument(cfg)
    if cfg["train"]:
        train(cfg)


if __name__ == '__main__':
    main()
