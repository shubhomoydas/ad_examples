"""
Some code motivated by:
    Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurelien Geron

To execute:
python -m dnn.dnn_classifier
"""

import numpy as np
import numpy.random as rnd
import tensorflow as tf
from common.utils import *


def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)


def dnn_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2. / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="W")
        b = tf.Variable(tf.zeros([n_neurons]), name="b")
        Z = tf.matmul(X, W) + b
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

    def eval(self, logits, labels):
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

    def training_op(self, loss, learning_rate=0.01):
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
        return training_op


def dnn_construct(X, n_neurons, names, activations):
    layer_input = X
    layers = list()
    with tf.name_scope("dnn"):
        for i, name in enumerate(names):
            hidden = dnn_layer(layer_input, n_neurons=n_neurons[i], name=names[i], activation=activations[i])
            layers.append(hidden)
            layer_input = hidden
    return DenseDNN(layers)


if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/dnn/dnn.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    rnd.seed(42)

    mnist = input_data.read_data_sets("../datasets/mnist/")

    n_inputs = 28*28  # MNIST
    n_epochs = 40
    batch_size = 50
    saved_file = "./temp/dnn/mnist_final_%d_%d.ckpt" % (n_epochs, batch_size)

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    dnn = dnn_construct(X, [300, 100, 10], names=["hidden1", "hidden2", "output"],
                        # activations=[tf.nn.relu, tf.nn.relu, None]
                        activations = [leaky_relu, leaky_relu, None])

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    logit_loss = dnn.logit_loss(y)
    training_op = dnn.training_op(logit_loss)
    accuracy = dnn.eval(dnn.output(), y)

    logger.debug("Constructed DNN")

    if False:
        # If previous session was *not* saved and we want to retrain
        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                for iteration in range(mnist.train.num_examples // batch_size):
                    X_batch, y_batch = mnist.train.next_batch(batch_size)
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                                    y: mnist.test.labels})
                logger.debug("epoch %d: Test accuracy: %f" % (epoch, acc_test))
            save_path = saver.save(sess, saved_file)
    else:
        # Only if previous session was saved
        with tf.Session() as sess:
            saver.restore(sess, saved_file)
            logger.debug("Loaded saved session")
            acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
            logger.debug("Test accuracy: %f" % (acc_test))
            # Z = dnn.output().eval(feed_dict={X: mnist.test.images})
            # y_pred = np.argmax(Z, axis=1)