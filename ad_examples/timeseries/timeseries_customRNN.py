import numpy as np
import tensorflow as tf
from ..common.utils import logger


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

    def __init__(self, n_lags, state_size, n_epochs=1, batch_size=-1, learning_rate=0.01, l2_penalty=0.001):
        self.state_size = state_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.n_lags = n_lags
        self.n_inputs = None

        self.init_state = None
        self.X = None
        self.Y = None

        self.err_loss = None
        self.training_op = None
        self.predict_op = None
        self.session = None

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

    def fit(self, ts):

        n_data = ts.series_len
        self.n_inputs = ts.dim
        batch_size = n_data if self.batch_size < 0 else self.batch_size
        logger.debug("n_inputs: %d, state_size: %d, n_lag: %d; batch_size: %d" %
                     (self.n_inputs, self.state_size, self.n_lags, batch_size))

        tf.set_random_seed(42)

        self.init_state = tf.placeholder(tf.float32, shape=(None, self.state_size))

        self.X = tf.placeholder(tf.float32, shape=(None, self.n_lags, self.n_inputs))
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

        return self.train(ts)

    def train(self, ts):
        x_train = y_train = None
        for x_train, y_train in ts.get_batches(self.n_lags, self.batch_size, single_output_only=True):
            pass
        z_train = np.zeros(shape=(x_train.shape[0], self.state_size))
        zero_state = np.zeros(shape=(self.batch_size, self.state_size))
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        for epoch in range(self.n_epochs):
            for i, batch in enumerate(ts.get_batches(self.n_lags, self.batch_size, single_output_only=True)):
                x, y = batch
                self.session.run([self.training_op],
                                 feed_dict={self.X: x, self.Y: y,
                                            self.init_state: zero_state[0:x.shape[0], :]})

            mse = self.session.run(self.err_loss,
                                   feed_dict={self.X: x_train, self.Y: y_train,
                                              self.init_state: z_train})
            logger.debug("epoch: %d, mse: %f" % (epoch, mse))

    def predict(self, start_ts, n_preds=1, true_preds=None):
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
        for i in range(n_preds):
            ts = seq[-self.n_lags:]
            X_batch = np.array(ts).reshape(1, self.n_lags, self.n_inputs)
            yhat = self.session.run(self.predict_op,
                                    feed_dict={self.X: X_batch,
                                               self.init_state: init_state})
            logger.debug("pred: %d %s" % (i, str(yhat)))
            preds.append(yhat[0, 0])
            if true_preds is not None:
                seq.append(true_preds[i])
            else:
                seq.append(yhat[0, 0])
        return np.array(preds)
