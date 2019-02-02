"""
Some code motivated by:
    Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurelien Geron
"""

import numpy as np
import numpy.random as rnd
import tensorflow as tf
from .utils import *
from .timeseries_datasets import *


def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)


def dnn_layer(x, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(x.get_shape()[1])
        stddev = 2. / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="W")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(x, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


class DenseDNN(object):
    def __init__(self, layers):
        self.layers = layers  # list of all layers

    def output(self):
        return self.layers[len(self.layers)-1]

    def logit_loss(self, labels):
        logits = self.output()
        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")
        return loss

    def mse_loss(self, y):
        yhat = self.output()
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.square(yhat - y), name="loss")
        return loss

    def eval(self, logits, labels):
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

    def training_op(self, loss, learning_rate=0.01):
        with tf.name_scope("train"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
        return training_op


def dnn_construct(x, n_neurons, names, activations):
    layer_input = x
    layers = list()
    with tf.name_scope("dnn"):
        for i, name in enumerate(names):
            hidden = dnn_layer(layer_input, n_neurons=n_neurons[i], name=names[i], activation=activations[i])
            layers.append(hidden)
            layer_input = hidden
    return DenseDNN(layers)


def get_train_batches(x, y=None, batch_size=-1, shuffle=False):
    n = x.shape[0]
    if batch_size < 0:
        batch_size = n
    indxs = np.arange(n)
    if shuffle:
        rnd.shuffle(indxs)
    for i in range(0, n, batch_size):
        et = min(i+batch_size, n)
        yield x[indxs[i:et], :], None if y is None else y[indxs[i:et]]


class MLPRegressor_TF(object):
    def __init__(self, n_inputs, n_neurons, n_outputs, n_epochs=100, batch_size=20,
                 learning_rate=0.01, l2_penalty=0.001, shuffle=False):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.shuffle = shuffle

        tf.set_random_seed(42)
        self.x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
        self.y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")
        self.dnn = dnn_construct(self.x, [n_neurons, n_outputs], names=["hidden", "output"],
                                 activations=[leaky_relu, None])

        with tf.name_scope("loss"):
            self.output = self.dnn.output()
            self.mse_loss = self.dnn.mse_loss(self.y)
            if self.l2_penalty > 0:
                vars = tf.trainable_variables()
                loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.l2_penalty
                self.mse_loss = self.mse_loss + loss_l2

        with tf.name_scope("training"):
            self.training_op = self.dnn.training_op(self.mse_loss, learning_rate=self.learning_rate)

        self.session = None

    def fit(self, x, y):
        self.session = tf.Session()
        init = tf.global_variables_initializer()

        y_ = np.reshape(y, newshape=(-1, self.n_outputs))
        self.session.run(init)
        for epoch in range(self.n_epochs):
            for x_batch, y_batch in get_train_batches(x, y, self.batch_size, shuffle=self.shuffle):
                y_batch_ = np.reshape(y_batch, newshape=(-1, self.n_outputs))
                self.session.run(self.training_op, feed_dict={self.x: x_batch, self.y: y_batch_})
            if False:
                # debug only
                mse = self.session.run(self.mse_loss, feed_dict={self.x: x, self.y: y_})
                logger.debug("epoch %d: MSE: %f" % (epoch, mse))

    def predict(self, x):
        pred = self.session.run(self.output, feed_dict={self.x: x})
        # logger.debug("pred: %s" % str(pred))
        return pred[:, 0]


class Autoencoder(object):
    """ A very simple autoencoder framework

    Consider denoising=True for over-complete representations i.e.,
    having higher dimension than input data. In such case will avoid trivial identity function.

    TODO: Step-by-step pre-training
    """
    def __init__(self, n_inputs, n_neurons, activations=None,
                 n_epochs=200, batch_size=20,
                 denoising=False, noise_level=1.0,
                 learning_rate=0.01, l2_penalty=0.001, shuffle=False):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activations = activations
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.denoising = denoising
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.shuffle = shuffle

        self.noise_level = noise_level
        self.session = None

        tf.set_random_seed(42)

        self.x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
        if denoising:
            self.x_noisy = self.x + self.noise_level * tf.random_normal(tf.shape(self.x))
        else:
            self.x_noisy = None

        names = ["hidden"] * len(n_neurons)
        names.append("output")
        n_hiddens = n_neurons
        n_hiddens.append(n_inputs)
        if activations is None:
            activations = [None] * len(n_hiddens)
        self.dnn = dnn_construct(self.x_noisy if denoising else self.x,
                                 n_hiddens, names=names, activations=activations)

        with tf.name_scope("loss"):
            self.output = self.dnn.output()
            self.mse_loss = self.dnn.mse_loss(self.x)
            if self.l2_penalty > 0:
                vars = tf.trainable_variables()
                loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.l2_penalty
                self.mse_loss = self.mse_loss + loss_l2

        with tf.name_scope("training"):
            self.training_op = self.dnn.training_op(self.mse_loss, learning_rate=self.learning_rate)

    def fit(self, x):
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.session.run(init)
        for epoch in range(self.n_epochs):
            for x_batch, _ in get_train_batches(x, None, self.batch_size, shuffle=self.shuffle):
                self.session.run(self.training_op, feed_dict={self.x: x_batch})
            if True:
                # debug only
                mse = self.session.run(self.mse_loss, feed_dict={self.x: x})
                logger.debug("epoch %d: MSE: %f" % (epoch, mse))

    def transform(self, x, layer_id=0):
        coding_layer = self.dnn.layers[layer_id]
        codings = self.session.run(coding_layer, feed_dict={self.x: x})
        return codings


class PCA_TF(Autoencoder):
    def __init__(self, n_inputs, n_dims=2, n_epochs=200, batch_size=20,
                 learning_rate=0.01, l2_penalty=0.001, shuffle=False):
        Autoencoder.__init__(self,
                             n_inputs=n_inputs,
                             n_neurons=[n_dims],
                             activations=None,
                             n_epochs=n_epochs, batch_size=batch_size,
                             denoising=False, noise_level=1.0,
                             learning_rate=learning_rate,
                             l2_penalty=l2_penalty, shuffle=shuffle)


class AutoencoderAnomalyDetector(object):
    def __init__(self, n_inputs, n_neurons, activations=None,
                 normalize_scale=False, n_epochs=200, batch_size=20,
                 denoising=False, noise_level=1.0,
                 learning_rate=0.01, l2_penalty=0.001, shuffle=False):
        self.autoencoder = Autoencoder(n_inputs, n_neurons, activations=activations,
                 n_epochs=n_epochs, batch_size=batch_size,
                 denoising=denoising, noise_level=noise_level,
                 learning_rate=learning_rate, l2_penalty=l2_penalty, shuffle=shuffle)
        self.normalizer = None
        if normalize_scale:
            self.normalizer = DiffScale()

    def fit(self, x):
        x_ = x if self.normalizer is None else self.normalizer.fit_transform(x)
        self.autoencoder.fit(x_)

    def decision_function(self, x):
        """ Returns smaller for more anomalous so that API is similar to other detectors """
        x_ = x if self.normalizer is None else self.normalizer.scale(x)
        decoded = self.autoencoder.transform(x_, layer_id=-1)  # output layer
        # logger.debug("x: %s, decoded: %s" % (str(x.shape), str(decoded.shape)))
        recons_errs = np.sum(np.square(decoded - x_), axis=1)
        # logger.debug("recons_errs: %s\n%s" % (str(decoded.shape), str(recons_errs)))
        return -recons_errs
