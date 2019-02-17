#!/bin/env python3
# -*- coding: utf8 -*-
import math

import click

import torch
from model import SimpleCifar10CNN
from torch import nn
from torchvision import transforms
from torchvision.datasets import cifar


def _xavier_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=math.sqrt(3))
        fan_in = fan_out = m.bias.shape[0]
        m.bias.data = nn.init.xavier_uniform_(
            torch.empty((fan_in, fan_out)), gain=math.sqrt(3)
        )[0]
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=1 / 0.87962566103423978)
        if m.bias:
            fan_in = fan_out = m.bias.shape[0]
            m.bias.data = nn.init.xavier_normal_(
                torch.empty((fan_in, fan_out)), gain=1 / 0.87962566103423978
            )[0]


@click.command()
@click.help_option("-h", "--help")
@click.option(
    "--batch-size", default=50, show_default=True, help="the image batch size", type=int
)
@click.option(
    "--lr",
    default=0.9,
    show_default=True,
    help="the learning rate of the optimizer",
    type=float,
)
@click.option(
    "--epochs", default=10, show_default=True, help="the number of epochs", type=int
)
@click.option(
    "--keep-prob",
    default=0.9,
    show_default=True,
    help="the dropout layer keep probability",
    type=float,
)
@click.option(
    "--output",
    help="output model file name",
    default="cifar10_cnn.ckpt",
    show_default=True,
)
@click.option("--ckpt-file", default=None, show_default=True)
def train(batch_size, lr, epochs, keep_prob, output, ckpt_file):
    click.echo(
        click.style(
            "lr: {}, keep_prob: {}, output: {}".format(lr, keep_prob, output),
            fg="cyan",
            bold=True,
        )
    )
    trans_train = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trans_test = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    cifar10_train = cifar.CIFAR10(
        "./cifar10_data", transform=trans_train, download=True, train=True
    )
    cifar10_test = cifar.CIFAR10(
        "./cifar10_data", transform=trans_test, download=True, train=False
    )
    train_loader = torch.utils.data.DataLoader(
        cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2
    )
    eval_loader = torch.utils.data.DataLoader(
        cifar10_test, batch_size=len(cifar10_test), shuffle=False, num_workers=2
    )
    model = SimpleCifar10CNN(keep_prob=keep_prob)
    model.apply(_xavier_init)
    if ckpt_file is not None:
        with open(ckpt_file, "rb") as fid:
            state = torch.load(fid)
            model.load_state_dict(state)
            click.echo("{} loaded".format(ckpt_file))
    cross_loss = nn.CrossEntropyLoss()
    device = torch.cuda.is_available() and "gpu" or "cpu"
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.95, eps=1e-7)
    best_acc, best_state = 0, None
    for epoch in range(1, epochs + 1):
        for i, (img_batch, label_batch) in enumerate(train_loader, 1):
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            logits = model(img_batch)
            loss = cross_loss(logits, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i % 100) == 0:
                click.echo(
                    click.style(
                        "[{}/{}: {}] train loss: {:0.4f}".format(
                            epoch, epochs, i, loss.item()
                        ),
                        fg="yellow",
                        bold=True,
                    )
                )
            if (i % 500) == 0:
                model.eval()
                img_batch, label_batch = next(iter(eval_loader))
                logits = model(img_batch)
                _, pred_label = torch.max(logits, 1)
                # fmt: off
                accuracy = (
                    (label_batch == pred_label).sum().item() /
                    label_batch.shape[0]
                )
                # fmt: on
                click.echo(
                    click.style(
                        "eval acc: {:0.2f}%".format(accuracy * 100),
                        fg="green",
                        bold=True,
                    )
                )
                if accuracy >= best_acc:
                    best_acc = accuracy
                    best_state = model.state_dict()
                model.train()
    click.echo(
        click.style(
            "saving best model to {} (best acc: {:0.2f})".format(
                output, best_acc * 100
            ),
            fg="white",
            bold=True,
        )
    )
    torch.save(best_state, output)


if __name__ == "__main__":
    train()
