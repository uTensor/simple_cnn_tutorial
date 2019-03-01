import os
from random import random

import click
import idx2numpy
import torch
from torchvision import transforms
from torchvision.datasets import cifar


@click.command()
@click.help_option('-h', '--help')
@click.argument('target_label', default=3, type=int)
def prepare_test_data(target_label):
    cifar10_train = cifar.CIFAR10("./cifar10_data", download=True, train=True)
    mean = (cifar10_train.train_data.astype("float32") / 255.0).mean(axis=(0, 1, 2))
    std = (cifar10_train.train_data.astype("float32") / 255.0).std(axis=(1, 2)).mean(axis=0)
    done = False
    while not done:
        for label, data in zip(cifar10_train.train_labels, cifar10_train.train_data):
            if label == target_label and random() < 0.1:
                data = data.astype('float32') / 255.0
                norm_data = (data - mean) / std
                with open('img_data.h', 'w') as fid:
                    fid.write(
                        '#ifndef __IMG_DATA_H\n'
                        '#define __IMG_DATA_H\n'
                    )
                    fid.write('static const int label_true = {};\n'.format(target_label))
                    fid.write(
                        'static const float img_data[{}] = {};\n'.format(
                            len(norm_data.ravel()),
                            str(norm_data.ravel().tolist()).replace('[', '{\n').replace(']', '\n}')
                        )
                    ) 
                    fid.write('#endif // __IMG_DATA_H\n')
                    done = True
                    break

if __name__ == '__main__':
    prepare_test_data()
