#!/bin/env python3
# -*- coding: utf8 -*-
import math

import click
import numpy as np
import tensorflow as tf

import torch
from model import build_graph
from torch import nn
from torchvision import transforms
from torchvision.datasets import cifar
from utensor_cgen.utils import prepare_meta_graph


def one_hot(labels, n_class=10):
    return np.eye(n_class)[labels]


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
    "--chkp-dir",
    default="chkp/cifar_cnn",
    show_default=True,
    help="directory where to save check point files",
)
@click.option(
    "--output-pb",
    help="output model file name",
    default="cifar10_cnn.pb",
    show_default=True,
)
def train(batch_size, lr, epochs, keep_prob, chkp_dir, output_pb):
    click.echo(
        click.style(
            "lr: {}, keep_prob: {}, output pbfile: {}".format(lr, keep_prob, output_pb),
            fg="cyan",
            bold=True,
        )
    )
    cifar10_train = cifar.CIFAR10("./cifar10_data", download=True, train=True)
    cifar10_test = cifar.CIFAR10("./cifar10_data", download=True, train=False)
    mean = (
        (cifar10_train.train_data.astype("float32") / 255.0)
        .mean(axis=(0, 1, 2))
        .tolist()
    )
    std = (
        (cifar10_train.train_data.astype("float32") / 255.0)
        .std(axis=(1, 2))
        .mean(axis=0)
        .tolist()
    )
    cifar10_train.transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    cifar10_test.transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    train_loader = torch.utils.data.DataLoader(
        cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2
    )
    eval_loader = torch.utils.data.DataLoader(
        cifar10_test, batch_size=len(cifar10_test), shuffle=False, num_workers=2
    )
    graph = tf.Graph()
    with graph.as_default():
        tf_image_batch = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        tf_labels = tf.placeholder(tf.float32, shape=[None, 10])
        tf_keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        tf_pred, train_op, tf_total_loss, saver = build_graph(
            tf_image_batch, tf_labels, tf_keep_prob, lr=lr
        )
    best_acc = 0.0
    chkp_cnt = 0
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for epoch in range(1, epochs + 1):
            for i, (img_batch, label_batch) in enumerate(train_loader, 1):
                np_img_batch = img_batch.numpy().transpose((0, 2, 3, 1))
                np_label_batch = label_batch.numpy()
                _ = sess.run(
                    train_op,
                    feed_dict={
                        tf_image_batch: np_img_batch,
                        tf_labels: one_hot(np_label_batch),
                        tf_keep_prob: keep_prob,
                    },
                )
                if (i % 100) == 0:
                    img_batch, label_batch = next(iter(eval_loader))
                    np_img_batch = img_batch.numpy().transpose((0, 2, 3, 1))
                    np_label_batch = label_batch.numpy()
                    pred_label = sess.run(
                        tf_pred,
                        feed_dict={tf_image_batch: np_img_batch, tf_keep_prob: 1.0},
                    )
                    acc = (pred_label == np_label_batch).sum() / np_label_batch.shape[0]
                    click.echo(
                        click.style(
                            "[epoch {}: {}], accuracy {:0.2f}%".format(
                                epoch, i, acc * 100
                            ),
                            fg="yellow",
                            bold=True,
                        )
                    )
                    if acc >= best_acc:
                        best_acc = acc
                        chkp_cnt += 1
                        click.echo(
                            click.style(
                                "[epoch {}: {}] saving checkpoint, {} with acc {:0.2f}%".format(
                                    epoch, i, chkp_cnt, best_acc * 100
                                ),
                                fg="white",
                                bold=True,
                            )
                        )
                        best_chkp = saver.save(sess, chkp_dir, global_step=chkp_cnt)
    best_graph_def = prepare_meta_graph(
        "{}.meta".format(best_chkp), output_nodes=[tf_pred.op.name]
    )

    with open(output_pb, "wb") as fid:
        fid.write(best_graph_def.SerializeToString())
        click.echo(click.style("{} saved".format(output_pb), fg="blue", bold=True))


if __name__ == "__main__":
    train()
