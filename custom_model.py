"""A custom RLLib model. It acts as a Torch module which takes observations and outputs actions and values."""
import torch
import torch.nn.functional as F
from ray.rllib.models import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from torch import nn


class MyRNNModel(RecurrentNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 fc_size=64,
                 lstm_state_size=256):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

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


class MyFCForwardModel(TorchModelV2, nn.Module):
    """PyTorch custom model.
    It just forwards a fully connected net (and should act the same as the default for non-image observations).
    Taken from example: https://github.com/ray-project/ray/blob/master/rllib/examples/models/custom_loss_model.py

    See https://docs.ray.io/en/master/rllib-models.html#custom-pytorch-models for how it works and how to extend it.
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

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Delegate to our FCNet."""
        return self.fcnet(input_dict, state, seq_lens)

    @override(TorchModelV2)
    def value_function(self):
        return self.fcnet.value_function()
