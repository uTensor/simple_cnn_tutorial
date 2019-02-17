from collections import OrderedDict

import torch
from torch import nn


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
                    ("drop1", nn.Dropout(p=1 - keep_prob)),
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
                    ("fc1", nn.Linear(128, 128, bias=False)),
                    ("relu5", nn.ReLU()),
                    ("drop2", nn.Dropout(p=1 - keep_prob)),
                    ("fc2", nn.Linear(128, 64, bias=False)),
                    ("relu6", nn.ReLU()),
                    ("fc3", nn.Linear(64, 10, bias=False)),
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
