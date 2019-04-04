from random import random

import click
from torchvision.datasets import cifar


def _random_sample(accept_prob):
    cifar10_train = cifar.CIFAR10("./cifar10_data", download=True, train=False)
    img_data = cifar10_train.data
    labels = cifar10_train.targets
    mean = (img_data.astype("float32") / 255.0).mean(axis=(0, 1, 2))
    std = (img_data.astype("float32") / 255.0).std(axis=(1, 2)).mean(axis=0)
    label_data_map = {}
    while len(label_data_map) < 10:
        for label, data in zip(labels, img_data):
            if label not in label_data_map and random() < accept_prob:
                norm_data = (data.astype("float32") / 255.0 - mean) / std
                label_data_map[label] = norm_data.ravel()
    return label_data_map


@click.command()
@click.help_option("-h", "--help")
@click.option("--accept-prob", default=0.3, show_default=True)
@click.option(
    "--labels",
    multiple=True,
    show_default=True,
    type=int,
    help="labels to add to testing data",
)
def prepare_test_data(accept_prob, labels):
    labels = list(labels) or [i for i in range(10)]
    data_map = _random_sample(accept_prob)
    with open("img_data.h", "w") as fid:
        fid.write("#ifndef __IMG_DATA_H\n" "#define __IMG_DATA_H\n")
        num_elems = data_map[0].shape[0]
        fid.write("static float imgs_data[%s][%s] = {\n" % (len(labels), num_elems))
        for label in labels:
            data = data_map[label]
            fid.write("    {\n")
            buffer = []
            for elem in data:
                if len(buffer) == 5:
                    fid.write("     " + ", ".join(buffer) + ",\n")
                    buffer = []
                buffer.append("{:0.6f}".format(elem))
            if buffer:
                fid.write("     " + ", ".join(buffer) + ",\n")
            fid.write("    },\n")
        fid.write("};\n")
        fid.write("#endif // __IMG_DATA_H\n")
    click.secho("img_data.h generated", fg='yellow', bold=True)


if __name__ == "__main__":
    prepare_test_data()
