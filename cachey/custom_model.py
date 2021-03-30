"""
Some custom RLLib models
"""
import functools
import operator

import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.models import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import override
from torch import nn


class MyConvGRUModel(RecurrentNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = obs_space.shape
        gru_hidden_dims = [32, 64, 16]
        self.flat_size = functools.reduce(operator.mul, (gru_hidden_dims[-1], *obs_space.shape))

        # This is the old (defunct) lib I pulled... keeping for posterity
        # from pytorch_convgru.convgru import ConvGRU
        # self.gru = ConvGRU(1, hidden_sizes=[32, 64, 16],
        #                    kernel_sizes=[3, 5, 3], n_layers=3)

        from convlstmgru.convgru import ConvGRU
        self.gru = ConvGRU(
            input_size=self.obs_size,
            input_dim=1,
            hidden_dim=gru_hidden_dims,
            kernel_size=(3, 3),
            num_layers=len(gru_hidden_dims),
            batchnorm=False,
            batch_first=True,
            activation=F.tanh
        )

        self.action_branch = nn.Linear(self.flat_size, num_outputs)
        self.value_branch = nn.Linear(self.flat_size, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [s.squeeze() for s in self.gru.get_init_states(1, cuda=False)]

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(ModelV2)
    def forward(self, input_dict,
                state,
                seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn()."""

        # Copy from RecurrentNetwork code
        # The default RecurrentNetwork flattens the observation first.
        # We want to preserve spatials.
        obs = input_dict["obs"].float()
        obs = obs.unsqueeze(dim=1).float()  # Add channel dim
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
        output, new_state = self.forward_rnn(obs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the ConvGRU Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        # TODO: Feed through a CNN first?
        self._features, h = self.gru(inputs, state)
        self._features = self._features.view(-1, self.flat_size)
        action_out = self.action_branch(self._features)
        new_state = h
        return action_out, new_state


class MyRNNModel(RecurrentNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = 64
        self.lstm_state_size = 256

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(
            self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),
                torch.unsqueeze(state[1], 0)])
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


class MyCNNModel(TorchModelV2, nn.Module):
    """PyTorch custom model which encodes the observation with a CNN before passing it to the policy/value network.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, *args, **kwargs)
        nn.Module.__init__(self)

        # This is Ray's default FC network. Eventually observations will pass from CNN -> CGRU -> LSTM -> FC.
        self.fcnet = FullyConnectedNetwork(
            self.obs_space,
            self.action_space,
            num_outputs,
            model_config,
            name="fcnet")
        
        self.unet = Unet(obs_space,num_outputs)

    def encode_observation(self,observation):
        return self.unet.forward(observation)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Delegate to our FCNet."""
        obs = input_dict["obs"].float()
        conv_out = self.encode_observation(obs)
        return self.fcnet(input_dict, state, seq_lens)

    @override(TorchModelV2)
    def value_function(self):
        return self.fcnet.value_function()
