import sys

import tensorflow as tf

if sys.version_info.major > 2:
    from functools import reduce


def get_conv_filter(
    width,
    height,
    in_channels,
    out_channels,
    dtype=tf.float32,
    initializer=None,
    seed=None,
    name=None,
):
    """
    arguments
    =========
    - width: int, filter width
    - height: int, filter height
    - in_channels: int, input channel
    - out_channels: int, output channel
    - dtype: data type
    - initializer: filter initializer
    - seed: random seed of the initializer
    """
    if initializer is None:
        initializer = tf.glorot_uniform_initializer(seed=seed, dtype=dtype)
    filter_shape = [width, height, in_channels, out_channels]
    return tf.Variable(initializer(shape=filter_shape), name=name, dtype=dtype)


def get_bias(shape, dtype=tf.float32, name=None, initializer=None, seed=None):
    if initializer is None:
        initializer = tf.glorot_uniform_initializer(seed=seed, dtype=dtype)
    return tf.Variable(initializer(shape=shape), name=name, dtype=dtype)


def conv_layer(in_fmap, w_shape, padding="SAME", stride=1, act_fun=None, name=None):
    width, height, in_channel, out_channel = w_shape
    strides = [1, stride, stride, 1]
    with tf.name_scope(name, "conv"):
        w_filter = get_conv_filter(width, height, in_channel, out_channel)
        out_fmap = tf.nn.conv2d(
            in_fmap, w_filter, padding=padding, strides=strides, name="feature_map"
        )
        bias = get_bias(w_filter.shape.as_list()[-1:], dtype=in_fmap.dtype, name="bias")
        act = tf.add(out_fmap, bias, name="logits")
        if act_fun:
            act = act_fun(act, name="activation")
    return act


def fc_layer(in_tensor, out_dim, act_fun=None, initializer=None, name=None):
    """Fully conneted layer
    """
    if initializer is None:
        initializer = tf.glorot_normal_initializer(dtype=in_tensor.dtype)
    w_shape = [in_tensor.shape.as_list()[-1], out_dim]
    with tf.name_scope(name, "fully_connect"):
        w_fc = tf.Variable(
            initializer(shape=w_shape, dtype=in_tensor.dtype), name="weight"
        )
        bias = get_bias(
            (out_dim,), dtype=in_tensor.dtype, name="bias", initializer=initializer
        )
        act = tf.add(tf.matmul(in_tensor, w_fc), bias, name="logits")
        if act_fun:
            act = act_fun(act, name="activation")
    return act


def cross_entropy_loss(logits, labels, name=None, axis=-1):
    """https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py#L3171
    """
    with tf.name_scope(name, "cross_entropy"):
        prob = tf.nn.softmax(logits=logits, axis=axis)
        prob = tf.clip_by_value(prob, 1e-7, 1 - 1e-7)
        loss = tf.reduce_sum(-labels * tf.log(prob), name="total_loss")
    return loss


def build_graph(tf_image_batch, tf_labels, tf_keep_prob, lr=1.0):
    """
    tf_image_batch: None x 32 x 32 x 3
    tf_labels: None x 10
    tf_keep_prob: None
    """
    graph = tf_image_batch.graph

    with graph.as_default():
        conv1 = conv_layer(tf_image_batch, [2, 2, 3, 16], padding="VALID")
        conv2 = conv_layer(conv1, [3, 3, 16, 32], padding="VALID", act_fun=tf.nn.relu)
        pool1 = tf.nn.max_pool(
            conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
        )
        conv3 = conv_layer(pool1, [3, 3, 32, 32], stride=2, padding="VALID")
        conv4 = conv_layer(
            conv3, [3, 3, 32, 32], padding="VALID", stride=2, act_fun=tf.nn.relu
        )
        drop1 = tf.nn.dropout(conv4, keep_prob=tf_keep_prob)
        pool2 = tf.nn.max_pool(
            drop1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
        )
        conv5 = conv_layer(pool2, [1, 1, 32, 64], padding="VALID", act_fun=tf.nn.relu)
        conv6 = conv_layer(conv5, [1, 1, 64, 128], act_fun=tf.nn.relu)
        flat_conv6 = tf.reshape(
            conv6, shape=[-1, reduce(lambda x, y: x * y, conv6.shape.as_list()[1:], 1)]
        )
        fc1 = fc_layer(flat_conv6, 128, act_fun=tf.nn.relu)
        drop_2 = tf.nn.dropout(fc1, keep_prob=tf_keep_prob)
        fc2 = fc_layer(drop_2, 64, act_fun=tf.nn.relu)
        logits = fc_layer(fc2, 10)
        tf_pred = tf.argmax(logits, axis=-1, name="pred")
        total_loss = cross_entropy_loss(logits=logits, labels=tf_labels)

        train_op = tf.train.AdadeltaOptimizer(learning_rate=lr, epsilon=1e-7).minimize(
            total_loss
        )
        saver = tf.train.Saver(max_to_keep=5)

        return tf_pred, train_op, total_loss, saver
