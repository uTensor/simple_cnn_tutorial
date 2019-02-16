# -*- coding: utf8 -*-
from collections import OrderedDict

import click

import torch
from torch import nn
from torch.onnx import export
from torchvision import transforms
from torchvision.datasets import cifar


class SimpleCifar10CNN(nn.Module):
    def __init__(self, keep_prob=0.9):
        super(SimpleCifar10CNN, self).__init__()
        self.layer1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 16, 2)),
                    ("conv2", nn.Conv2d(16, 32, 3)),
                    ("relu1", nn.ReLU()),
                    ("pool1", nn.MaxPool2d(2, stride=2)),
                ]
            )
        )
        self.layer2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv3", nn.Conv2d(32, 32, 3, stride=2)),
                    ("conv4", nn.Conv2d(32, 32, 3, stride=2)),
                    ("relu2", nn.ReLU()),
                    ("drop1", nn.Dropout(p=keep_prob)),
                    ("pool2", nn.MaxPool2d(2, stride=2)),
                ]
            )
        )
        self.layer3 = nn.Sequential(
            OrderedDict(
                [
                    ("conv5", nn.Conv2d(32, 64, 1)),
                    ("relu3", nn.ReLU()),
                    ("conv6", nn.Conv2d(64, 128, 1)),
                    ("relu4", nn.ReLU()),
                ]
            )
        )
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(128, 128)),
                    ("relu5", nn.ReLU()),
                    ("drop2", nn.Dropout(p=keep_prob)),
                    ("fc2", nn.Linear(128, 64)),
                    ("relu6", nn.ReLU()),
                    ("fc3", nn.Linear(64, 10)),
                ]
            )
        )

    def forward(self, img_batch):
        out = self.layer1(img_batch)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 128)
        out = self.fc(out)
        return out


@click.command()
@click.help_option("-h", "--help")
@click.option(
    "--batch-size", default=50, show_default=True, help="the image batch size", type=int
)
@click.option(
    "--lr",
    default=0.001,
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
    trans = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]
    )
    cifar10_train = cifar.CIFAR10(
        "./cifar10_data", transform=trans, download=True, train=True
    )
    cifar10_test = cifar.CIFAR10(
        "./cifar10_data", transform=transforms.ToTensor(), train=False, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        cifar10_train, batch_size=batch_size, shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        cifar10_test, batch_size=len(cifar10_test), shuffle=False
    )
    model = SimpleCifar10CNN(keep_prob=keep_prob)
    if ckpt_file is not None:
        with open(ckpt_file, "rb") as fid:
            state = torch.load(fid)
            model.load_state_dict(state)
            click.echo("{} loaded".format(ckpt_file))
    cross_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        for i, (img_batch, label_batch) in enumerate(train_loader, 1):
            logits = model(img_batch)
            optimizer.zero_grad()
            loss = cross_loss(logits, label_batch)
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
                model.train()
    click.echo(click.style("saving model to {}".format(output), fg="white", bold=True))
    torch.save(model.state_dict(), output)


if __name__ == "__main__":
    train()
