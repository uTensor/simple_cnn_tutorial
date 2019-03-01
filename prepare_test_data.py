import os
from random import random

import idx2numpy
import torch
from torchvision import transforms
from torchvision.datasets import cifar
from utensor_cgen.utils import save_idx

cifar10_train = cifar.CIFAR10("./cifar10_data", download=True, train=True)
cifar10_test = cifar.CIFAR10("./cifar10_data", download=True, train=False)


mean = (cifar10_train.train_data.astype("float32") / 255.0).mean(axis=(0, 1, 2))
std = (cifar10_train.train_data.astype("float32") / 255.0).std(axis=(1, 2)).mean(axis=0)


idx_map = {}
while len(idx_map) < 10:
    for i, data in enumerate(cifar10_train.train_data):
        label = cifar10_train.train_labels[i]
        if random() < 0.1 and label not in idx_map:
            idx_map[label] = i

if not os.path.exists("idx_data"):
    os.mkdir("idx_data")
for label, idx in idx_map.items():
    data = cifar10_train.train_data[idx].astype("float32") / 255.0
    norm_data = (data - mean) / std
    save_idx(norm_data, "idx_data/{}.idx".format(label))
