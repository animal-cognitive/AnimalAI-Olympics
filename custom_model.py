"""A custom RLLib model. It acts as a Torch module which takes observations and outputs actions and values."""

from ray.rllib.models import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from torch import nn


class MyFCForwardModel(TorchModelV2, nn.Module):
    """PyTorch custom model.
    It just forwards a fully connected net (same as the default for non-image observations)
    Taken from example: https://github.com/ray-project/ray/blob/3bdcca7ee56b60178517308711b4384561bba94d/rllib/examples/models/custom_loss_model.py
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        nn.Module.__init__(self)

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
