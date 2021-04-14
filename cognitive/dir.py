import copy
from random import random

from animalai.envs.arena_config import ArenaConfig
from torch.utils.data import Dataset


class DIRWrapper:
    def __init__(self, model):
        self.model = model
        self.model.lstm.register_forward_hook(self.save_dir)
        self.dir = None

    def get_dir(self):
        return self.dir

    def save_dir(self, module, input, output):
        """Example hook that just prints the input/output shapes."""
        # in_features, in_state = input
        # out_features, out_state = output
        # # Expect (1 x 1 x 64) coming from FC layer into LSTM
        # print('input features:', in_features.shape, 'previous state:', [s.shape for s in in_state])
        # Expect (1 x 1 x 256) coming out of LSTM. This is the DIR.
        out_features, out_state = output
        self.dir = (out_features, out_state)
        # print('DIR output:', out_features.shape, 'current state:', [s.shape for s in out_state])


def rotate_180():
    # env._flattener.action_lookup
    # todo test this
    if random() > 0.5:
        return [1] * 10
    else:
        return [3] * 10


def remove_goal(arena_config) -> ArenaConfig:
    arena_config = copy.deepcopy(arena_config)
    items = arena_config.arenas[0].items
    gg_idx = None
    for i in range(len(items)):
        if items[i].name == "GoodGoal":
            gg_idx = i
            break
    assert gg_idx is not None
    arena_config.arenas[0].items.pop(gg_idx)
    return arena_config


class DIRDataset(Dataset):
    def __init__(self, dir, label):
        super(DIRDataset, self).__init__()
        self.dir = dir
        self.label = label

    def __getitem__(self, item):
        return self.dir[item], self.label[item]

    def __len__(self):
        return len(self.dir)
