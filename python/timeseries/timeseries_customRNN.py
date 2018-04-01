import random
import numpy as np
import numpy.random as rnd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from common.utils import *
from common.timeseries_datasets import *
from common.data_plotter import *


"""
Simple time-series modeling with custom RNN

Some code motivated by:
    Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurelien Geron
    https://machinelearningmastery.com/time-train_series-forecasting-long-short-term-memory-network-python/
    https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
"""


class TsRNNCustom(object):
    """
    Does not use any RNN cells. Simply approximates what goes on inside the RNN cells.
    The loss is only based on the final output, and not the intermediate time outputs.

    NOTE(s):
        (1) Should consider this for only lag-1 time series although the API suports more than 1 lags.
    """

    def __init__(self, n_lag, state_size, n_epochs=1, batch_size=-1, learning_rate=0.01, l2_penalty=0.001):
        self.state_size = state_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.n_lag = n_lag
        self.n_inputs = None

        self.init_state = None
        self.X = None
        self.Y = None

        self.err_loss = None
        self.training_op = None
        self.predict_op = None

    def rnn_cell(self, rnn_input, hidden_state):
        with tf.variable_scope('rnn_cell', reuse=True):
            W = tf.get_variable('W', shape=[self.state_size, self.state_size], dtype=np.float32)
            b = tf.get_variable('b', shape=[1, self.state_size], dtype=np.float32,
                                initializer=tf.constant_initializer(0.0))
            U = tf.get_variable('U', shape=[self.n_inputs, self.state_size], dtype=np.float32)
            c = tf.get_variable('c', shape=[1, self.n_inputs], dtype=np.float32,
                                initializer=tf.constant_initializer(0.0))
            V = tf.get_variable('V', shape=[self.state_size, self.n_inputs], dtype=np.float32)
        new_state = tf.tanh(tf.matmul(hidden_state, W) + tf.matmul(rnn_input, U) + b)
        # output = tf.matmul(new_state, V) + c
        return new_state  # , output

    def fit(self, ts, n_predict=0):

        n_data = ts.series_len
        self.n_inputs = ts.dim
        batch_size = n_data if self.batch_size < 0 else self.batch_size
        logger.debug("n_inputs: %d, state_size: %d, n_lag: %d; batch_size: %d" %
                     (self.n_inputs, self.state_size, self.n_lag, batch_size))

        tf.set_random_seed(42)

        self.init_state = tf.placeholder(tf.float32, shape=(None, self.state_size))

        self.X = tf.placeholder(tf.float32, shape=(None, self.n_lag, self.n_inputs))
        self.Y = tf.placeholder(tf.float32, shape=(None, self.n_inputs))

        # rnn_inputs is a list of n_lag tensors with shape [batch_size, n_inputs]
        rnn_inputs = tf.unstack(self.X, axis=1)

        with tf.variable_scope('rnn_cell'):
            W = tf.get_variable('W', shape=[self.state_size, self.state_size], dtype=np.float32)
            b = tf.get_variable('b', shape=[1, self.state_size], dtype=np.float32,
                                initializer=tf.constant_initializer(0.0))
            U = tf.get_variable('U', shape=[self.n_inputs, self.state_size], dtype=np.float32)
            c = tf.get_variable('c', shape=[1, self.n_inputs], dtype=np.float32,
                                initializer=tf.constant_initializer(0.0))
            V = tf.get_variable('V', shape=[self.state_size, self.n_inputs], dtype=np.float32)

        state = self.init_state
        for rnn_input in rnn_inputs:
            state = self.rnn_cell(rnn_input, state)
        final_y = tf.matmul(state, V) + c

        self.predict_op = final_y

        with tf.name_scope("loss"):
            self.err_loss = tf.reduce_mean(tf.square(final_y - self.Y))
            if self.l2_penalty > 0:
                l2_loss = self.l2_penalty * (tf.nn.l2_loss(U) + tf.nn.l2_loss(V) + tf.nn.l2_loss(W))
                reg_loss = tf.add(self.err_loss, l2_loss, name="l2loss")
            else:
                reg_loss = self.err_loss
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.training_op = optimizer.minimize(reg_loss)

        return self.train(ts, n_predict=n_predict)

    def train(self, ts, n_predict=0):
        n_data = ts.series_len
        preds = None
        x_train = y_train = None
        for x_train, y_train in ts.get_batches(self.n_lag, self.batch_size, single_output_only=True):
            pass
        z_train = np.zeros(shape=(x_train.shape[0], self.state_size))
        zero_state = np.zeros(shape=(self.batch_size, self.state_size))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.n_epochs):
                for i, batch in enumerate(ts.get_batches(self.n_lag, self.batch_size, single_output_only=True)):
                    x, y = batch
                    sess.run([self.training_op],
                             feed_dict={self.X: x, self.Y: y,
                                        self.init_state: zero_state[0:x.shape[0], :]})

                mse = self.err_loss.eval(feed_dict={self.X: x_train, self.Y: y_train,
                                                    self.init_state: z_train})
                logger.debug("epoch: %d, mse: %f" % (epoch, mse))

            if n_predict > 0:
                preds = self.predict(ts.samples[-self.n_lag:, :], n=n_predict)

        return preds

    def predict(self, start_ts, n=1):
        """
        Predict with fixed model.
        NOTE: Assumes that each time input is one-dimensional.
        """
        if self.n_inputs != 1:
            raise ValueError("Currently only supports univariate input per time-step")
        seq = list(np.reshape(start_ts, newshape=(-1,)))
        logger.debug("seq: %s" % str(seq))
        preds = list()
        init_state = np.zeros(shape=(1, self.state_size))
        for i in range(n):
            ts = seq[-self.n_lag:]
            X_batch = np.array(ts).reshape(1, self.n_lag, self.n_inputs)
            yhat = self.predict_op.eval(feed_dict={self.X: X_batch,
                                                   self.init_state: init_state})
            logger.debug("pred: %d %s" % (i, str(yhat)))
            preds.append(yhat[0, 0])
            seq.append(yhat)
        return np.array(preds)
