from animalai.envs.arena_config import ArenaConfig
from torch.utils.data import Dataset


class DIRWrapper:
    def __init__(self, model):
        self.model = model

    def get_dir(self):
        return None

def no_action():
    return None

def rotate_180():
    return None

def remove_goal(arena_config) -> ArenaConfig:
    return None

class DIRDataset(Dataset):
    def __init__(self, dir, label):
        super(DIRDataset, self).__init__()
        self.dir = dir
        self.label = label

    def __getitem__(self, item):
        return self.dir[item], self.label[item]

    def __len__(self):
        return len(self.dir)
