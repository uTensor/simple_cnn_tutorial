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
    "--batch-size", default=64, show_default=True, help="the image batch size", type=int
)
@click.option(
    "--lr",
    default=1.0,
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
def train(batch_size, lr, epochs, keep_prob, output):
    cifar10 = cifar.CIFAR10(
        "./cifar10_data", transform=transforms.ToTensor(), download=True, train=True
    )
    train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True)
    model = SimpleCifar10CNN(keep_prob=keep_prob)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    cross_loss = nn.CrossEntropyLoss()
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
    click.echo(click.style("saving model to {}".format(output), fg="white", bold=True))
    torch.save(model.state_dict(), output)


if __name__ == "__main__":
    train()
