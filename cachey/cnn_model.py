import gym
import ray
import torch
import torch.nn.functional as F
from animalai.envs.arena_config import ArenaConfig
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelV2, ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.tune import register_env
from torch import nn


# from cognitive.primitive_arena import RayAIIGym


class CNNModel(TorchModelV2, nn.Module):
    """
    obs -> CNN -> linear -> action/value
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        assert len(obs_space.shape) == 2 or len(obs_space.shape) == 3, \
            f'Must take image observations!'

        if len(obs_space.shape) == 2:
            w, h = obs_space.shape
            in_channels = 1  # Assume BW input
        else:
            w, h, in_channels = obs_space.shape

        self._setup_config(model_config)

        dummy_cnn_output = self._setup_cnn(in_channels, obs_space.shape)
        cnn_output_flat_size = self.cnn_flatten(dummy_cnn_output).shape[-1]
        self.linear = nn.Linear(cnn_output_flat_size, self.linear_size)
        n_logits = self.linear.out_features

        self.action_branch = nn.Linear(n_logits, num_outputs)
        self.value_branch = nn.Linear(n_logits, 1)

        # Holds the last "base" output (before logits layer) for calls to value_function.
        self._features = None

    def _setup_config(self, model_config):
        if not model_config.get("linear_size"):
            model_config["linear_size"] = 64
        self.linear_size = model_config["linear_size"]

    def _setup_cnn(self, in_channels, input_shape):
        # Nature CNN (Mnih et. al 2015)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.cnn_flatten = nn.Flatten()
        # Do a forward pass of dummy data to get the shape of the CNN output
        with torch.no_grad():
            dummy_input = torch.zeros(input_shape)
            if len(input_shape) == 2:
                dummy_input = dummy_input.unsqueeze(2)
            dummy_input = dummy_input.permute(2, 0, 1)
            dummy_input = dummy_input.unsqueeze(0)  # Add batch dim
            dummy_output = self.cnn(dummy_input)
        return dummy_output

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
        x = input_dict["obs"].float()
        if len(x.shape[1:]) < 3:
            x = x.unsqueeze(dim=3)  # Add channel dim
        x = x.permute(0, 3, 1, 2)  # Permute to (B x C x W x H)
        x = self.cnn(x)
        x = self.cnn_flatten(x)
        x = F.relu(self.linear(x))
        self._features = x
        action_out = self.action_branch(self._features)
        action_out = torch.reshape(action_out, [-1, self.num_outputs])  # Remove time dim if present

        return action_out, state


def test_sanity():
    """Test simple passing observations."""
    from ray.rllib.env.atari_wrappers import wrap_deepmind
    env = wrap_deepmind(gym.make('Pong-v4'))
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    obs = torch.stack([torch.from_numpy(prep.transform(env.observation_space.sample())) for _ in range(100)], dim=0)
    input_dict = {
        "obs": obs,
    }
    model = CNNModel(env.observation_space, env.action_space, env.action_space.n, {}, 'test')
    model(input_dict, None, None)
    env.close()


def test_trainonebatch():
    """Test hooking into an RLLib trainer."""
    from cognitive.primitive_arena import GymFactory

    ray.shutdown()
    ray.init(include_dashboard=False, local_mode=True)
    register_env("my_env", GymFactory(ArenaConfig('examples/configurations/curriculum/0.yml')))
    ModelCatalog.register_custom_model("cnn_model", CNNModel)
    trainer = PPOTrainer(env="my_env", config={
        "model": {
            "custom_model": 'cnn_model',
            "custom_model_config": {},
        },
        "num_workers": 0,
        "framework": 'torch',
    })
    trainer.train()
