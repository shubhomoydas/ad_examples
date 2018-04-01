import random
import numpy as np
import numpy.random as rnd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from common.utils import *
from common.timeseries_datasets import *
from common.data_plotter import *
from timeseries_customRNN import TsRNNCustom

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
Simple time-series modeling with Tensorflow RNN and LSTM cells

To execute:
pythonw -m timeseries.timeseries_rnn
"""


class TsRNN(object):
    """
    Uses Tensorflow's Basic RNN cell
    """
    def __init__(self, n_lag, n_neurons, n_epochs=1, batch_size=-1, learning_rate=0.001, use_lstm=True):
        self.n_neurons = n_neurons
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_lag = n_lag
        self.n_inputs = None
        self.n_outputs = None
        self.use_lstm = use_lstm

        self.X = None

        self.training_op = None
        self.predict_op = None

    def fit(self, ts, n_predict=0):
        """
        Borrowed from:
            Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurelien Geron
        """
        n_data = ts.series_len
        self.n_inputs = ts.dim
        self.n_outputs = ts.y.shape[1]
        batch_size = n_data if self.batch_size < 0 else self.batch_size
        logger.debug("n_inputs: %d, n_outputs: %d, state_size: %d, n_lag: %d; batch_size: %d" %
                     (self.n_inputs, self.n_outputs, self.n_neurons, self.n_lag, batch_size))

        X_batch_ts = Y_batch_ts = None
        for X_batch_ts, Y_batch_ts in ts.get_batches(self.n_lag, batch_size):
            pass

        tf.set_random_seed(42)

        self.X = tf.placeholder(tf.float32, shape=[None, self.n_lag, self.n_inputs])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.n_lag, self.n_outputs])

        if self.use_lstm:
            logger.debug("Using LSTM in RNN")
            cell = tf.contrib.rnn.LSTMCell(num_units=self.n_neurons)
        else:
            cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=tf.nn.relu)
        # basic_cell = tf.contrib.rnn.OutputProjectionWrapper(
        #     tf.contrib.rnn.BasicRNNCell(num_units=self.state_size, activation=tf.nn.relu),
        #     output_size=self.n_outputs
        # )
        rnn_outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, self.n_lag, self.n_outputs])
        self.predict_op = outputs

        preds = None

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.square(outputs - self.Y))
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            init.run()
            for epoch in range(self.n_epochs):
                for i, batch in enumerate(ts.get_batches(self.n_lag, batch_size, single_output_only=False)):
                    X_batch, Y_batch = batch
                    # logger.debug("X_batch.shape: %s" % str(X_batch.shape))
                    sess.run([training_op], feed_dict={self.X: X_batch, self.Y: Y_batch})
                mse = loss.eval(feed_dict={self.X: X_batch_ts, self.Y: Y_batch_ts})
                logger.debug("epoch: %d, mse: %f" % (epoch, mse))

            if n_predict > 0:
                # predict while keeping the model fixed
                preds = self.predict(ts.samples[-n_lag:, :], n=n_predict)

        return preds

    def predict(self, start_ts, n=1):
        seq = list(np.reshape(start_ts, newshape=(-1,)))
        logger.debug("seq: %s" % str(seq))
        preds = list()
        for i in range(n):
            ts = seq[-self.n_lag:]
            X_batch = np.array(ts).reshape(1, self.n_lag, self.n_inputs)
            y_preds = self.predict_op.eval(feed_dict={self.X: X_batch})
            yhat = y_preds[0, -1, 0]
            logger.debug("pred: %d %s" % (i, str(yhat)))
            preds.append(yhat)
            seq.append(yhat)
        return np.array(preds)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/timeseries/timeseries_rnn.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    dir_create("./temp/timeseries")  # for logging and plots

    random.seed(42)
    rnd.seed(42)

    dataset = "airline"
    # dataset = "shampoo"
    df = get_univariate_timeseries_data(dataset)
    # logger.debug(df)
    all_series =  np.array(df.values, dtype=np.float32)  # training

    n = all_series.shape[0]
    n_training = int(2. * n / 3.)
    train_series = all_series[0:n_training]
    test_series = all_series[n_training:n]
    logger.debug("Dataset: %s, n_training: %d, n_test: %d" % (dataset, n_training, test_series.shape[0]))
    # logger.debug("train_series.shape: %s\n%s" % (str(train_series.shape), str(train_series)))
    # logger.debug("test_series.shape: %s\n%s" % (str(test_series.shape), str(test_series)))

    # remove trend by differencing
    diff_series = difference_series(train_series, interval=1)
    # logger.debug("diff_series.shape: %s\n%s" % (str(diff_series.shape), str(diff_series)))

    # normalize by mean and variance
    scaler = MinMaxScaler(feature_range=(-1, 1))  # since output is tanh
    scaler = scaler.fit(diff_series)
    scld_series = scaler.transform(diff_series)
    # logger.debug("scld_series.shape: %s\n%s" % (str(scld_series.shape), str(scld_series)))

    use_custom = True
    use_lstm = False
    batch_size = 10
    n_lag = 4
    n_neurons = 100
    n_epochs = 100

    train_ts = prepare_tseries(scld_series)
    n_preds = test_series.shape[0]  # 12

    if False:
        # check if the series iterates correctly
        logger.debug("len(scld_series): %d" % len(scld_series))
        logger.debug("scld_series:\n%s" % str(list(np.round(scld_series[:, 0], 3))))
        train_ts.log_batches(n_lag, 200, single_output_only=False)

    # if True: exit(0)

    if use_custom:
        # if using the custom RNN
        rnn_type = "custom"
        tsrnn = TsRNNCustom(n_lag=n_lag, state_size=n_neurons,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=0.001, l2_penalty=0.0)
    else:
        # if using the RNN/LSTM cells
        if use_lstm:
            rnn_type = "lstm"
            batch_size = 1
        else:
            rnn_type = "basic"
        tsrnn = TsRNN(n_lag=n_lag, n_neurons=n_neurons,
                      n_epochs=n_epochs, batch_size=batch_size,
                      learning_rate=0.001, use_lstm=use_lstm)

    preds = tsrnn.fit(train_ts, n_predict=n_preds)
    # logger.debug(preds)

    n_tr = scld_series.shape[0]

    final_preds = None
    if preds is not None:
        # convert the predictions into original ranges through inverse transforms
        pred_series = preds.reshape((len(preds), 1))
        inv_pred_series = scaler.inverse_transform(pred_series)
        final_preds = invert_difference_series(
            inv_pred_series, initial=train_series[n_training - 1, :], interval=1)
        logger.debug("final_preds.shape: %s\n%s" % (str(final_preds.shape), str(final_preds)))

    pdfpath = "temp/timeseries/timeseries_rnn_%s_%s.pdf" % (rnn_type, dataset)
    dp = DataPlotter(pdfpath=pdfpath, rows=3, cols=1)

    pl = dp.get_next_plot()
    plt.title("Time train_series %s" % dataset, fontsize=8)
    pl.set_xlim([0, n])
    pl.plot(np.arange(0, n_training), train_series[:, 0], 'b-')
    pl.plot(np.arange(n_training, n), test_series[:, 0], 'r-')

    if final_preds is not None:
        pl = dp.get_next_plot()
        plt.title("unscaled predictions %s" % dataset, fontsize=8)
        pl.set_xlim([0, n])
        pl.plot(np.arange(1, n_tr+1), scld_series[:, 0], 'b-')
        pl.plot(np.arange(n_tr+1, n_tr+1+n_preds), preds, 'r-')

        pl = dp.get_next_plot()
        plt.title("final predictions %s" % dataset, fontsize=8)
        pl.set_xlim([0, n])
        pl.plot(np.arange(0, n_training), train_series[:, 0], 'b-')
        pl.plot(np.arange(n_training, n), test_series[:, 0], 'b-')
        pl.plot(np.arange(n_training, n_training+final_preds.shape[0]), final_preds[:, 0], 'r-')
    dp.close()
