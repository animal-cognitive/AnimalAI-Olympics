"""
Take the DIR datasets, train MLP and evaluate the representation
"""
import argparse
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

import torch.nn.functional as F

from cognitive.dir import DIRDataset
from cognitive.logger import Logger, MultiAverageMeter
from cognitive.primitive_arena import BeforeOrBehind, Occlusion, Rotation


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


def load_data(dataset: DIRDataset, batch_size=32, num_workers=1):
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True if num_workers else False,
                    persistent_workers=True if num_workers else False, shuffle=True)
    return dl


def load_dir_dataset(model_str):
    Arenas = {BeforeOrBehind: "BeforeOrBehind",
              Occlusion: "Occlusion",
              # Rotation: "Rotation" # none of the agents are able to find the ball even.
              }

    datasets = {}
    for Arena, name in Arenas.items():
        with open(f'cognitive/{model_str}_dataset/{name}.dir', "rb") as f:
            ds = pickle.load(f)
        datasets[name] = ds
    return datasets


def ttest_dataset():
    mnames = ["cnn", "lstm", "reduced", "whole"]
    for mname in mnames:
        datasets = load_dir_dataset(mname)
        for name, ds in datasets.items():
            train_ds, val_ds = ds.split()
            for ds in (train_ds, val_ds):
                assert (ds.get_balance() == 1 / 2)


def main_helper(args):
    datasets = load_dir_dataset(args.model_str)
    start_with = 0
    for name, ds in datasets.items():
        args.arena_name = name
        train_ds, val_ds = ds.split()
        if start_with:
            start_with -= 1
            continue
        logger = Logger(set_name="cognitive", version=f"{name}_{args.model_str}")
        model = MLP(args.hidden, args.layers)
        model = model.to(args.device)
        loss_fn = F.binary_cross_entropy_with_logits
        optim = torch.optim.Adadelta(model.parameters(), lr=0.01)
        val_dl = load_data(val_ds, args.batch_size, 0)
        train_dl = load_data(train_ds, args.batch_size, args.num_workers)

        for epoch in range(args.epochs):
            train(args, epoch, train_dl, model, loss_fn, optim, logger)
            if epoch % 20 == 0:
                valid(args, epoch, val_dl, model, loss_fn, logger)
        logger.close()


def train(args, epoch, train_dl, model, loss_fn, optim, logger):
    device = args.device
    meter = MultiAverageMeter()
    for x, y in train_dl:
        if args.model_str not in ("cnn",):
            x = x.squeeze()
        y = y.unsqueeze(-1).float()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        pred = pred.squeeze()
        y = y.squeeze()
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()
        p = torch.sigmoid(pred)
        acc = ((p > 0.5) == y).float().mean()
        meter.update(loss=loss.item(), accuracy=acc.item())
        if args.dry_run:
            return model
    if epoch % 20 == 0:
        logger.auto_log(f"training", model=args.model_str, arena=args.arena_name, epoch=epoch,
                        **meter.get())
    return model


def valid(args, epoch, val_dl, model, loss_fn, logger):
    device = args.device
    with torch.no_grad():
        losses = []
        accs = []
        for x, y in val_dl:
            if args.model_str not in ("cnn",):
                x = x.squeeze()
            y = y.unsqueeze(-1).float()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred = pred.squeeze()
            y = y.squeeze()
            loss = loss_fn(pred, y)
            p = torch.sigmoid(pred)
            acc = ((p > 0.5) == y).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())
            if args.dry_run:
                return

        logger.auto_log(f"validation", model=args.model_str, arena=args.arena_name,
                        epoch=epoch,
                        loss=sum(losses) / len(losses),
                        accuracy=sum(accs) / len(accs),
                        )


def get_args():
    parser = argparse.ArgumentParser(description='Train MLP for DIR.')
    parser.add_argument("--epochs", default=81, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--hidden", default=64, type=int)
    parser.add_argument("--layers", default=6, type=int)
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model_strs = ["cnn", "lstm", "whole", "reduced"]
    # model_strs = ["lstm", "whole", "reduced"]
    for dry in (False,):
        args.dry_run = dry
        for ms in model_strs:
            args.model_str = ms
            main_helper(args)


if __name__ == '__main__':
    main()
