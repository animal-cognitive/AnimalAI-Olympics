"""
Custom models specific to Cache architecture.
This is the only tested implementation, see bottom.
"""

import gym
import ray
import torch
import torch.nn.functional as F
from animalai.envs.arena_config import ArenaConfig
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelV2, ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils import override
from ray.tune import register_env
from torch import nn


# from cognitive.primitive_arena import RayAIIGym


class LSTMModel(RecurrentNetwork, nn.Module):
    """
    obs -> linear -> LSTM -> action/value
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        assert len(obs_space.shape) == 2 or len(obs_space.shape) == 3, \
            f'Must take image observations!'
        self._setup_config(model_config)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.linear = nn.Linear(self.obs_size, self.linear_size)
        self.lstm = nn.LSTM(self.linear_size, self.lstm_state_size, batch_first=True)
        n_logits = self.lstm_state_size

        self.action_branch = nn.Linear(n_logits, num_outputs)
        self.value_branch = nn.Linear(n_logits, 1)
        self._features = None

    def _setup_config(self, model_config):
        if "enable_lstm" not in model_config:
            model_config["enable_lstm"] = True
        self.use_lstm = model_config["enable_lstm"]
        if not model_config.get("lstm_state_size"):
            model_config["lstm_state_size"] = 256
        self.lstm_state_size = model_config["lstm_state_size"]
        if not model_config.get("linear_size"):
            model_config["linear_size"] = 64
        self.linear_size = model_config["linear_size"]

    @override(ModelV2)
    def get_initial_state(self):
        lstm_h = [
            self.action_branch.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.action_branch.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return lstm_h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(RecurrentNetwork)
    def forward_rnn(self, x, state, seq_lens):
        x = F.relu(self.linear(x))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),
                torch.unsqueeze(state[1], 0)])
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


def test_sanity():
    """Test simple passing observations."""
    from ray.rllib.env.atari_wrappers import wrap_deepmind
    env = wrap_deepmind(gym.make('Pong-v4'))
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    obs = torch.stack([torch.from_numpy(prep.transform(env.observation_space.sample())) for _ in range(100)], dim=0)
    input_dict = {
        "obs": obs,
    }
    model = LSTMModel(env.observation_space, env.action_space, env.action_space.n, {}, 'test')
    state = model.get_initial_state()
    state = [s.unsqueeze(0) for s in state]
    seq_lens = torch.tensor([input_dict["obs"].shape[0]])  # One sequence of the whole thing
    model(input_dict, state, seq_lens)
    env.close()


def test_trainonebatch():
    """Test hooking into an RLLib trainer."""
    from cognitive.primitive_arena import GymFactory

    ray.shutdown()
    ray.init(include_dashboard=False, local_mode=True)
    register_env("my_env", GymFactory(ArenaConfig('examples/configurations/curriculum/0.yml')))
    ModelCatalog.register_custom_model("lstm_model", LSTMModel)
    trainer = PPOTrainer(env="my_env", config={
        "model": {
            "custom_model": 'lstm_model',
            "custom_model_config": {},
        },
        "num_workers": 0,
        "framework": 'torch',
    })
    trainer.train()
