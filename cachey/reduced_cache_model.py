"""
Custom models specific to Cache architecture.
This is the only tested implementation, see bottom.
"""

import operator
from functools import reduce

import gym
import numpy as np
import pytest
import ray
import torch
import torch.nn.functional as F
# import cognitive
# from animalai.envs.arena_config import ArenaConfig
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.atari_wrappers import wrap_deepmind
from ray.rllib.models import ModelV2, ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import override
from ray.tune import register_env
from torch import nn

# from cognitive.primitive_arena import RayAIIGym


class MyCNNRNNModel(RecurrentNetwork, nn.Module):
    """
    Feeds observations to a CNN, then LSTM.
    Can use only the CNN with custom_model_config.
    Docs: https://docs.ray.io/en/master/rllib-models.html#custom-pytorch-models

    Use it like this:
    ModelCatalog.register_custom_model("my_cnn_rnn_model", MyCNNRNNModel)
    config = {
    ...
    "custom_model": 'my_cnn_rnn_model',
    "custom_model_config": {
        'gimme_lstm': True|False
    }
    }
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        assert len(obs_space.shape) == 2 or len(obs_space.shape) == 3, \
            f'MyCNNRNNModel must take image observations!'

        if len(obs_space.shape) == 2:
            w, h = obs_space.shape
            in_channels = 1  # Assume BW input
        else:
            w, h, in_channels = obs_space.shape

        # By default, use LSTM. But leave an option to disable it.
        if "gimme_lstm" not in model_config:
            model_config["gimme_lstm"] = True
        self.use_lstm = model_config["gimme_lstm"]

        # Nature CNN (Mnih et. al 2015)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Do a forward pass of dummy data to get the shape of the CNN output
        with torch.no_grad():
            dummy_input = torch.zeros(obs_space.shape)
            if len(obs_space.shape) == 2:
                dummy_input = dummy_input.unsqueeze(2)
            dummy_input = dummy_input.permute(2, 0, 1)
            dummy_input = dummy_input.unsqueeze(0)  # Add batch dim
            dummy_output = self.cnn(dummy_input)
            self.cnn_output_flat_size = reduce(operator.mul, dummy_output.shape[1:])

        if not model_config.get("linear_size"):
            model_config["linear_size"] = 64
        self.linear_size = model_config["linear_size"]
        self.linear = nn.Linear(self.cnn_output_flat_size, self.linear_size)

        if self.use_lstm:
            # Build the Module from fc + LSTM + 2xfc (action + value outs).
            # Parameter lstm_state_size configures the size of the hidden state.
            if not model_config.get("lstm_state_size"):
                model_config["lstm_state_size"] = 256
            self.lstm_state_size = model_config["lstm_state_size"]

            self.lstm = nn.LSTM(self.linear_size, self.lstm_state_size, batch_first=True)
            n_logits = self.lstm_state_size
        else:
            n_logits = self.linear_size

        self.action_branch = nn.Linear(n_logits, num_outputs)
        self.value_branch = nn.Linear(n_logits, 1)

        # Holds the last "base" output (before logits layer) for calls to value_function.
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(ModelV2)
    def forward(self, input_dict,
                state,
                seq_lens):
        """
        Encodes and adds time dimension to batch of observations before sending inputs to forward_rnn().
        """

        # Pass observation to CNN
        # The default RecurrentNetwork flattens the observation first.
        # We want to preserve spatials.
        x = input_dict["obs"].float()
        if len(x.shape[1:]) < 3:
            x = x.unsqueeze(dim=3).float()  # Add channel dim
        x = x.permute(0, 3, 1, 2)  # Permute to (B x C x W x H)
        x = self.cnn(x)
        x = F.relu(self.linear(x))

        if self.use_lstm:
            # Add time dimension
            if isinstance(seq_lens, np.ndarray):
                seq_lens = torch.Tensor(seq_lens).int()
            max_seq_len = x.shape[0] // seq_lens.shape[0]
            self.time_major = self.model_config.get("_time_major", False)
            x = add_time_dimension(
                x,
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=self.time_major,
            )

            # Pass to LSTM and out
            self._features, state = self.forward_rnn(x, state, seq_lens)
        else:
            self._features = x

        action_out = self.action_branch(self._features)
        action_out = torch.reshape(action_out, [-1, self.num_outputs])  # Remove time dim if present

        return action_out, state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        output, [h, c] = self.lstm(
            inputs, [torch.unsqueeze(state[0], 0),
                     torch.unsqueeze(state[1], 0)])
        return output, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


@pytest.mark.parametrize('model_config', [
    {'gimme_lstm': False},  # CNN only
    {'gimme_lstm': True},  # CNN+LSTM
    {},  # Also CNN+LSTM, just checking that the defaults are sane
])
def test_sanity(model_config):
    """Test simple passing observations."""
    env = wrap_deepmind(gym.make('Pong-v4'))
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    obs = torch.stack([torch.from_numpy(prep.transform(env.observation_space.sample())) for _ in range(100)], dim=0)
    input_dict = {
        "obs": obs,
    }
    model = MyCNNRNNModel(env.observation_space, env.action_space, env.action_space.n, model_config, 'test')
    if model.use_lstm:
        state = [s.unsqueeze(0) for s in model.get_initial_state()]
        seq_lens = torch.tensor([input_dict["obs"].shape[0]])  # One sequence of the whole thing
        model(input_dict, state, seq_lens)
    else:
        model(input_dict)
    env.close()


# def test_trainonebatch():
#     """Test hooking into an RLLib trainer."""
#     ray.shutdown()
#     ray.init(include_dashboard=False)
#     register_env("my_env", lambda c: RayAIIGym(c, ArenaConfig('examples/configurations/curriculum/0.yml')))
#     ModelCatalog.register_custom_model("my_cnn_rnn_model", MyCNNRNNModel)
#     trainer = PPOTrainer(env="my_env", config={
#         "model": {
#             "custom_model": 'my_cnn_rnn_model',
#             "custom_model_config": {},
#         },
#         "framework": 'torch',
#     })
#     trainer.train()
