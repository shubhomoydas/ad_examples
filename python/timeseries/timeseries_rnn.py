import random
import numpy as np
import numpy.random as rnd
from pandas import concat
import tensorflow as tf
from common.utils import *
from common.timeseries_datasets import *
from common.data_plotter import *
from timeseries.timeseries_customRNN import TsRNNCustom

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
Anomaly detection by simple time-series modeling with Tensorflow RNN and LSTM cells.

After modeling the timeseries, we predict values for new time points
and flag the time points with the most deviating values as anomalies.

To execute:
pythonw -m timeseries.timeseries_rnn --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_rnn.log --normalize_trend --algo=lstm --n_lags=12 --dataset=airline
pythonw -m timeseries.timeseries_rnn --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_rnn.log --normalize_trend --algo=basic --n_lags=12 --dataset=airline

pythonw -m timeseries.timeseries_rnn --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_rnn.log --normalize_trend --algo=lstm --n_lags=5 --dataset=shampoo
pythonw -m timeseries.timeseries_rnn --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_rnn.log --normalize_trend --algo=basic --n_lags=4 --dataset=lynx
pythonw -m timeseries.timeseries_rnn --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_rnn.log --normalize_trend --algo=lstm --n_lags=4 --dataset=aus_beer
pythonw -m timeseries.timeseries_rnn --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_rnn.log --normalize_trend --algo=lstm --n_lags=12 --dataset=us_accident
pythonw -m timeseries.timeseries_rnn --n_epochs=20 --debug --log_file=temp/timeseries/timeseries_rnn.log --normalize_trend --algo=lstm --n_lags=50 --dataset=wolf_sunspot

The below does not work well...need longer dependencies
pythonw -m timeseries.timeseries_rnn --dataset=fisher_temp --algo=lstm --n_lags=200 --n_epochs=10 --debug --log_file=temp/timeseries/timeseries_rnn.log
"""


class TsRNN(object):
    """
    Uses Tensorflow's Basic RNN cell
    """
    def __init__(self, n_lags, n_neurons, n_epochs=1, batch_size=-1, learning_rate=0.001, use_lstm=True):
        self.n_neurons = n_neurons
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_lags = n_lags
        self.n_inputs = None
        self.n_outputs = None
        self.use_lstm = use_lstm

        self.X = None

        self.training_op = None
        self.predict_op = None
        self.session = None

    def fit(self, ts):
        """
        Borrowed from:
            Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurelien Geron
        """
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
        self.Y = tf.placeholder(tf.float32, shape=[None, self.n_lags, self.n_outputs])

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
        outputs = tf.reshape(stacked_outputs, [-1, self.n_lags, self.n_outputs])
        self.predict_op = outputs

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.square(outputs - self.Y))
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
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


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/timeseries")  # for logging and plots

    args = get_command_args(debug=False,
                            debug_args=["--dataset=airline", "--algo=lstm", "--n_lags=12",
                                        "--n_anoms=10", "--debug", "--plot",
                                        "--log_file=temp/timeseries/timeseries_rnn.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    random.seed(42)
    rnd.seed(42)

    rnn_type = args.algo  # custom, basic, lstm
    n_anoms = args.n_anoms
    n_lags = args.n_lags
    n_epochs = args.n_epochs
    normalize_trend = args.normalize_trend
    batch_size = 10
    n_neurons = 100  # number of nodes in hidden state

    allowed_algos = {'custom': 'Custom implementation of the basic RNN cell',
                     'basic': 'RNN using the basic cell',
                     'lstm': 'RNN using LSTM Cell'}
    if args.algo not in allowed_algos.keys():
        print ("Invalid algo: %s. Allowed algos:" % args.algo)
        for key, val in allowed_algos.iteritems():
            print ("  %s: %s" % (key, val))
        exit(0)

    dataset = args.dataset
    logger.debug("dataset: %s, algo: %s" % (dataset, args.algo))
    if not dataset in univariate_timeseries_datasets:
        print ("Invalid dataset: %s. Supported datasets: %s" %
              (dataset, str(univariate_timeseries_datasets.keys())))
        exit(0)
    df = get_univariate_timeseries_data(dataset)

    all_series = np.array(df.values, dtype=np.float32)  # training

    n = all_series.shape[0]
    n_training = int(2. * n / 3.)
    train_series = all_series[0:n_training]
    test_series = all_series[n_training:n]
    logger.debug("Dataset: %s, n_training: %d, n_test: %d" % (dataset, n_training, test_series.shape[0]))
    # logger.debug("train_series.shape: %s\n%s" % (str(train_series.shape), str(train_series)))
    # logger.debug("test_series.shape: %s\n%s" % (str(test_series.shape), str(test_series)))

    normalizer = DiffScale()
    train_input = normalizer.fit_transform(train_series, normalize_trend=normalize_trend)

    train_ts = prepare_tseries(train_input)
    n_preds = test_series.shape[0]

    if False:
        # check if the series iterates correctly
        logger.debug("len(scld_series): %d" % len(train_input))
        logger.debug("scld_series:\n%s" % str(list(np.round(train_input[:, 0], 3))))
        train_ts.log_batches(n_lags, 200, single_output_only=False)

    if rnn_type == "custom":
        # if using the custom RNN
        tsrnn = TsRNNCustom(n_lags=n_lags, state_size=n_neurons,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=0.001, l2_penalty=0.0)
    else:
        # if using the RNN/LSTM cells
        if rnn_type == "lstm":
            use_lstm = True
        elif rnn_type == "basic":
            use_lstm = False
        else:
            raise ValueError("Invalid RNN type: %s" % rnn_type)
        tsrnn = TsRNN(n_lags=n_lags, n_neurons=n_neurons,
                      n_epochs=n_epochs, batch_size=batch_size,
                      learning_rate=0.001, use_lstm=use_lstm)

    tsrnn.fit(train_ts)
    preds = tsrnn.predict(train_ts.samples[-n_lags:, :], n_preds=n_preds)
    # logger.debug("preds:\n%s" % str(preds))

    final_preds = None
    if preds is not None:
        pred_series = np.reshape(preds, newshape=(len(preds), 1))
        final_preds = normalizer.inverse_transform(pred_series,
                                                   train_series[n_training - 1, :] if normalize_trend else None)

    pdfpath = "temp/timeseries/timeseries_rnn_%s_%s.pdf" % (rnn_type, dataset)
    dp = DataPlotter(pdfpath=pdfpath, rows=3, cols=1)

    pl = dp.get_next_plot()
    plt.title("Time train_series %s" % dataset, fontsize=8)
    pl.set_xlim([0, n])
    pl.plot(np.arange(0, n_training), train_series[:, 0], 'b-')
    pl.plot(np.arange(n_training, n), test_series[:, 0], 'r-')

    if final_preds is not None:
        scores = np.abs(test_series[:, 0] - final_preds[:, 0])
        n_anoms = min(n_anoms, final_preds.shape[0])
        top_anoms = np.argsort(-scores)[0:n_anoms]
        logger.debug("top scores:\n%s\n%s" % (str(top_anoms), str(scores[top_anoms])))

        pl = dp.get_next_plot()
        plt.title("unscaled predictions %s" % dataset, fontsize=8)
        pl.set_xlim([0, n])
        pl.plot(np.arange(0, n_training), train_input[:, 0], 'b-')
        pl.plot(np.arange(n_training, n_training+n_preds), preds, 'r-')

        pl = dp.get_next_plot()
        plt.title("final predictions %s" % dataset, fontsize=8)
        pl.set_xlim([0, n])
        pl.plot(np.arange(0, n_training), train_series[:, 0], 'b-')
        pl.plot(np.arange(n_training, n), test_series[:, 0], 'b-')
        pl.plot(np.arange(n_training, n_training+final_preds.shape[0]), final_preds[:, 0], 'r-')
        for i in top_anoms:
            plt.axvline(n_training+i, color='g', linewidth=0.5)

    dp.close()
