"""A custom RLLib model. It acts as a Torch module which takes observations and outputs actions and values."""

from ray.rllib.models import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from torch import nn


class MyFCForwardModel(TorchModelV2, nn.Module):
    """PyTorch custom model.
    It just forwards a fully connected net (and should act the same as the default for non-image observations).
    Taken from example: https://github.com/ray-project/ray/blob/master/rllib/examples/models/custom_loss_model.py

    See https://docs.ray.io/en/master/rllib-models.html#custom-pytorch-models for how it works and how to extend it.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
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
