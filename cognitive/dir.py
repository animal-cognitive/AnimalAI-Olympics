import copy
import random

from animalai.envs.arena_config import ArenaConfig
from torch.utils.data import Dataset
import copy

from cachey.cnn_model import CNNModel


class DIRWrapper:
    def __init__(self, model):
        self.model = model
        if not isinstance(self.model, CNNModel):
            self.model.lstm.register_forward_hook(self.save_dir)
        self.dir = None

    def get_dir(self):
        if not isinstance(self.model, CNNModel):
            return self.dir
        else:
            return self.model._features

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
    if random.random() > 0.5:
        return [4] * 20
    else:
        return [5] * 20


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
        return self.dir[item][0].detach(), self.label[item]

    def __len__(self):
        return len(self.dir)

    def split(self, ratio=0.1):
        self.clean()
        self.balance()

        indices = list(range(len(self)))
        random.shuffle(indices)
        val_size = int(len(self) * ratio)
        val, train = indices[:val_size], indices[val_size:]
        val_dir, val_label = [self.dir[v] for v in val], [self.label[v] for v in val]
        train_dir, train_label = [self.dir[t] for t in train], [self.label[t] for t in train]
        a, b = DIRDataset(train_dir, train_label), DIRDataset(val_dir, val_label)
        a.balance()
        b.balance()
        return a, b

    def balance(self):
        pos_idx = [i for i in range(len(self)) if self.label[i]]*2
        neg_idx = [i for i in range(len(self)) if not self.label[i]]*2
        balanced = len(self) // 2
        pos = random.sample(pos_idx, k=balanced)
        neg = random.sample(neg_idx, k=balanced)
        dir = []
        label = []
        for idx in pos + neg:
            dir.append(self.dir[idx])
            label.append(self.label[idx])
        self.dir = dir
        self.label = label

    def clean(self):
        for idx, d in enumerate(self.dir):
            if isinstance(d, tuple):
                if d[0].shape[0] != 1:
                    self.dir[idx][0] = d[0][0].unsqueeze(0)
            elif d is None:
                self.dir[idx] = self.dir[idx - 1]
                self.label[idx] = self.label[idx - 1]
            else:
                if d.shape[0] != 1:
                    self.dir[idx] = d[0].unsqueeze(0)

        for i in range(len(self.dir)):
            if isinstance(self.dir[i], tuple):
                assert self.dir[0][0].shape == self.dir[i][0].shape
            else:
                assert self.dir[0].shape == self.dir[i].shape

    def get_balance(self):
        return sum(self.label) / len(self.label)
