import random
import numpy as np
import numpy.random as rnd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from common.utils import *
from common.timeseries_datasets import *
from common.data_plotter import *

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from timeseries.simulate_timeseries import *
from sklearn import manifold

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Simple time-series modeling with Tensorflow RNN and LSTM cells

To execute:
pythonw -m timeseries.activity_model --log_file=temp/timeseries/activity_model.log --debug --n_epochs=100 --n_lags=20 --algo=lstm

Supported algo(s): lstm, basic
"""


class ActivityRNN(object):
    """
    Models activity sequences
    """
    def __init__(self, n_lags, n_neurons, n_classes, n_epochs=1, batch_size=-1, learning_rate=0.001, use_lstm=True):
        self.n_lags = n_lags
        self.n_neurons = n_neurons
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_inputs = None
        self.n_outputs = None
        self.use_lstm = use_lstm

        self.X = None

        self.training_op = None
        self.predict_op = None
        self.states = None
        self.session = None

    def fit(self, ts):
        n_data = ts.series_len
        self.n_inputs = ts.dim
        self.n_outputs = ts.y.shape[1]
        batch_size = n_data if self.batch_size < 0 else self.batch_size
        logger.debug("n_inputs: %d, n_outputs: %d, state_size: %d, n_lag: %d; batch_size: %d" %
                     (self.n_inputs, self.n_outputs, self.n_neurons, self.n_lags, batch_size))

        X_batch_ts = Y_batch_ts = None
        for X_batch_ts, Y_batch_ts in ts.get_batches(self.n_lags, batch_size):
            pass

        tf.set_random_seed(42)

        self.X = tf.placeholder(tf.float32, shape=[None, self.n_lags, self.n_inputs])
        self.Y = tf.placeholder(tf.int32, shape=[None, self.n_lags, self.n_outputs])

        init = tf.zeros((self.n_classes, self.n_neurons), dtype=tf.float32)
        self.L = tf.Variable(init, name="L")

        if self.use_lstm:
            logger.debug("Using LSTM in RNN")
            cell = tf.contrib.rnn.LSTMCell(num_units=self.n_neurons)
        else:
            cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=tf.nn.relu)
        rnn_outputs, self.states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.n_classes)
        # outputs = tf.reshape(stacked_outputs, [-1, self.n_lags, n_classes])
        _, self.predict_op = tf.nn.top_k(stacked_outputs)
        self.predict_op = tf.reshape(self.predict_op, [-1, self.n_lags, self.n_outputs])
        stacked_labels = tf.reshape(self.Y, [-1])
        if False:
            one_hot_labels = tf.one_hot(indices=stacked_labels, depth=self.n_classes, axis=-1)
            label_encs = tf.matmul(one_hot_labels, self.L)
        else:
            label_probs = tf.nn.softmax(stacked_outputs)
            label_encs = tf.matmul(label_probs, self.L)

        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=stacked_labels,
                                                                      logits=stacked_outputs)
            label_loss = tf.reduce_mean(xentropy)

            # minimizing squared loss between label encoding and the corresponding
            # activities leads to a simplistic label encoding: labels are represented
            # as the mean of the related activities in the latent (hidden) space.
            encoding_loss = tf.reduce_mean(tf.square(stacked_rnn_outputs - label_encs))

            loss = label_loss + encoding_loss
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        self.session = tf.Session()
        self.session.run(init)
        for epoch in range(self.n_epochs):
            for i, batch in enumerate(ts.get_batches(self.n_lags, batch_size, single_output_only=False)):
                X_batch, Y_batch = batch
                self.session.run([training_op], feed_dict={self.X: X_batch, self.Y: Y_batch})
            mse = self.session.run(loss, feed_dict={self.X: X_batch_ts, self.Y: Y_batch_ts})
            logger.debug("epoch: %d, mse: %f" % (epoch, mse))

    def get_label_encodings(self):
        encs = self.session.run(self.L, feed_dict={})
        return encs

    def predict(self, start_ts, n_preds=1, true_preds=None):
        seq = list(np.reshape(start_ts, newshape=(-1,)))
        logger.debug("seq: %s" % str(seq))
        preds = list()
        for i in range(n_preds):
            ts = seq[-self.n_lags:]
            X_batch = np.array(ts).reshape(1, self.n_lags, self.n_inputs)
            y_preds = self.session.run(self.predict_op, feed_dict={self.X: X_batch})
            yhat = y_preds[0, -1, 0]
            logger.debug("pred: %d %s" % (i, str(yhat)))
            preds.append(yhat)
            if true_preds is not None:
                seq.append(true_preds[i])
            else:
                seq.append(yhat)
        return np.array(preds)

    def transform(self, ts):
        """ Transforms the activity observations into context state """
        states = None
        x = y = None
        for i, batch in enumerate(ts.get_batches(self.n_lags, -1, single_output_only=False)):
            x, y = batch
            # logger.debug("X_batch: %s" % str(X_batch.shape))
            states = self.session.run(self.states, feed_dict={self.X: x, self.Y: y})
        return x, y, states


def plot_original_feature_tsne(ts, n_lags):
    """ plot t-SNE for original feature space """
    x = y = None
    for x, y in ts.get_batches(n_lags, -1, single_output_only=True):
        x = np.reshape(x, newshape=(x.shape[0], -1))
        y = np.reshape(y, newshape=(y.shape[0], -1))
    logger.debug("computing t-SNE for original space...")
    embed = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tr = embed.fit_transform(x)
    y_tr = y[:, -1]
    pdfpath = "temp/timeseries/activity_tsne_orig_%s.pdf" % (args.algo)
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    dp.plot_points(x_tr, pl, labels=y_tr, marker='o',
                   lbl_color_map={0: "blue", 1: "red", 2: "green", 3: "orange"}, s=12)
    dp.close()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/timeseries")  # for logging and plots

    args = get_command_args(debug=False, debug_args=["--n_lags=20",
                                                     "--n_epochs=100",
                                                     "--algo=lstm",
                                                     "--debug",
                                                     "--plot",
                                                     "--log_file=temp/timeseries/activity_model.log"])

    # print "log file: %s" % args.log_file
    configure_logger(args)

    random.seed(42)
    rnd.seed(42)

    acts_full = read_activity_data()
    logger.debug("samples: %s, activities: %s, starts: %s" %
                 (str(acts_full.samples.shape), str(acts_full.activities.shape), str(acts_full.starts.shape)))

    # Just take a short sequence so that processing does not take long on laptop
    # and can easily illustrate the idea behind the algorithm for sequence modeling ...
    acts_sub = TSeries(acts_full.samples[0:2000, :], y=acts_full.y[0:2000, :])
    logger.debug("y: %s" % str(acts_sub.y.shape))

    n_lags = args.n_lags
    n_classes = 3
    n_neurons = 100
    use_lstm = True if args.algo == "lstm" else False
    n_epochs = args.n_epochs
    batch_size = 30

    if False:
        # t-SNE with i.i.d. 'window' based features with same n_lags as sequence
        # model is *not* as good as t-SNE of hidden states for sequence model.
        # Therefore, the sequence model indeed models the data in a more meaningful way.
        plot_original_feature_tsne(acts_sub, n_lags)
        exit(0)

    rnn = ActivityRNN(n_lags=n_lags, n_neurons=n_neurons, n_classes=n_classes,
                      n_epochs=n_epochs, batch_size=batch_size,
                      learning_rate=0.001, use_lstm=use_lstm)

    rnn.fit(acts_sub)

    x, y, states = rnn.transform(acts_sub)
    # logger.debug("states: %s, x: %s, y: %s" % (str(states.shape), str(x.shape), str(y.shape)))
    logger.debug("x: %s, y: %s" % (str(x.shape), str(y.shape)))
    y = np.reshape(y, newshape=(y.shape[0], -1))
    y_tr = y[:, -1]  # only the final activity label
    logger.debug("x: %s, y: %s" % (str(x.shape), str(y_tr.shape)))

    label_encodings = rnn.get_label_encodings()
    logger.debug("Encodings: %s\n%s" % (str(label_encodings.shape), str(label_encodings)))

    if use_lstm:
        exit(0)  # TODO: extract the state

    states_a = np.vstack([states, label_encodings])
    y_tr_a = np.asarray(np.hstack([y_tr, np.arange(n_classes, dtype=np.int32)]), dtype=int)

    logger.debug("computing t-SNE...")
    embed = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tr_a = embed.fit_transform(states_a)

    # logger.debug("Label encodings:\n%s\n%s" % (str(x_tr_a[-n_classes:, :]), y_tr_a[-n_classes:]))

    pdfpath = "temp/timeseries/activity_tsne_%s.pdf" % (args.algo)
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    dp.plot_points(x_tr_a, pl, labels=y_tr_a, marker='o',
                   lbl_color_map={0: "blue", 1: "red", 2: "green", 3: "orange"}, s=12)
    dp.plot_points(x_tr_a[-n_classes:, :], pl, labels=y_tr_a[-n_classes:], marker='o',
                   lbl_color_map={0: "blue", 1: "red", 2: "green", 3: "orange"}, s=25)
    dp.plot_points(x_tr_a[-n_classes:, :], pl, labels=y_tr_a[-n_classes:], marker='+',
                   defaultcol="black", s=55, linewidths=3.0)
    dp.close()

