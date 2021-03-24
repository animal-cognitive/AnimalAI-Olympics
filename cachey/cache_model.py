"""
Custom models specific to Cache architecture.
"""

import operator
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import override
from torch import nn


class MyCNNRNNModel(RecurrentNetwork, nn.Module):
    """
    Feeds observations to a CNN, then LSTM.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        assert len(obs_space.shape) == 2 or len(obs_space.shape) == 3, \
            f'MyCNNRNNModel must take image observations!'

        if len(obs_space.shape) < 4:
            in_channels = 1  # Assume BW input
        else:
            in_channels = obs_space.shape[0]
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
            if len(obs_space.shape) < 4:
                dummy_input = dummy_input.unsqueeze(0).unsqueeze(0)
            dummy_output = self.cnn(dummy_input)
            self.cnn_output_flat_size = reduce(operator.mul, dummy_output.shape[1:])

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        # Parameter lstm_state_size configures the size of the hidden state.
        if not model_config.get("lstm_state_size"):
            model_config["lstm_state_size"] = 256
        if not model_config.get("linear_size"):
            model_config["linear_size"] = 64
        self.lstm_state_size = model_config["lstm_state_size"]
        self.linear_size = model_config["linear_size"]

        self.linear = nn.Linear(self.cnn_output_flat_size, self.linear_size)
        self.lstm = nn.LSTM(self.linear_size, self.lstm_state_size, batch_first=True)

        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)

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
        obs = input_dict["obs"].float()
        if len(obs.shape) < 4:
            obs = obs.unsqueeze(dim=1).float()  # Add channel dim
        obs = self.cnn(obs)
        obs = F.relu(self.linear(obs))

        # Add time dimension
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = obs.shape[0] // seq_lens.shape[0]
        self.time_major = self.model_config.get("_time_major", False)
        obs = add_time_dimension(
            obs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )

        # Pass to LSTM and out
        output, new_state = self.forward_rnn(obs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        self._features, [h, c] = self.lstm(
            inputs, [torch.unsqueeze(state[0], 0),
                     torch.unsqueeze(state[1], 0)])
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]
