"""
Take the DIR datasets, train MLP and evaluate the representation
"""
import argparse
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cognitive.primitive_arena import BeforeOrBehind, Occlusion, Rotation
import torch.nn.functional as F
from cognitive.logger import Logger, MultiAverageMeter


class MLP(nn.Module):
    def __init__(self, hidden=32, layers=4):
        super(MLP, self).__init__()
        self.hidden = hidden
        self.layers = layers
        self.pipe = nn.ModuleList()
        for i in range(layers):
            self.pipe.append(nn.LazyLinear(hidden))
            self.pipe.append(nn.ReLU())
            if i % 3 == 1:
                self.pipe.append(nn.Dropout(0.1))
        self.pipe.append(nn.LazyLinear(1))

    def forward(self, input):
        x = input
        for layer in self.pipe:
            x = layer(x)
        return x


def load_dir_dataset():
    Arenas = {BeforeOrBehind: "BeforeOrBehind",
              Occlusion: "Occlusion",
              Rotation: "Rotation"}

    datasets = {}
    for Arena, name in Arenas.items():
        with open('cognitive/dataset/' + name + ".dir", "rb") as f:
            ds = pickle.load(f)
        datasets[name] = ds
    return datasets


def load_data(dataset, batch_size=32, num_workers=1):
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                    persistent_workers=True if num_workers else False, pin_memory=True,
                    shuffle=True)
    return dl


def main(args):
    datasets = load_dir_dataset()
    start_with = 0
    for name, ds in datasets.items():
        meter = MultiAverageMeter()
        train_ds, val_ds = ds.split()
        if start_with:
            start_with -= 1
            continue
        logger = Logger(set_name="cognitive", version=name)
        model = MLP(args.hidden, args.layers)
        model = model.to(args.device)
        loss_fn = F.binary_cross_entropy_with_logits
        optim = torch.optim.Adadelta(model.parameters(), lr=0.001)
        train(args, train_ds, model, loss_fn, optim, meter, logger)
        valid(args, val_ds, model, loss_fn, logger)


def train(args, train_ds, model, loss_fn, optim, meter, logger):
    epochs = args.epochs
    device = args.device
    dl = load_data(train_ds, args.batch_size, args.num_workers)
    for epoch in range(epochs):
        for x, y in dl:
            x = x[0].squeeze()
            y = y.unsqueeze(-1).float()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
            acc = ((pred > 0.5) == y).float().mean()
            meter.update(loss=loss.item(), accuracy=acc.item())
        if epoch % 20 == 0:
            logger.auto_log("training", epoch=epoch, **meter.get())
    return model


def valid(args, val_ds, model, loss_fn, logger):
    device = args.device
    dl = load_data(val_ds, args.batch_size, args.num_workers)
    with torch.no_grad():
        for x, y in dl:
            x = x[0].squeeze()
            y = y.unsqueeze(-1).float()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            acc = ((pred > 0.5) == y).float().mean()
            logger.auto_log("validation", loss=loss.item(), accuracy=acc.item())


def get_args():
    parser = argparse.ArgumentParser(description='Train MLP for DIR.')
    parser.add_argument("--epochs", default=141, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--layers", default=4, type=int)
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
