import random
import numpy as np
import numpy.random as rnd
from sklearn.preprocessing import MinMaxScaler
from common.utils import *
from common.data_plotter import *
from common.nn_utils import *

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from common.timeseries_datasets import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
Anomaly detection in time series by breaking the series into windows ('shingles')
and then treating each window as i.i.d feature vector.

pythonw -m timeseries.timeseries_shingles --debug --plot --log_file=temp/timeseries/timeseries_shingles.log --n_lags=20 --algo=autoenc --dataset=synthetic
pythonw -m timeseries.timeseries_shingles --debug --plot --log_file=temp/timeseries/timeseries_shingles.log --n_lags=6 --algo=ifor --normalize_trend --log_transform --dataset=airline
"""


def find_anomalies_with_shingles(dataset, data, window_size=5, skip_size=None, ad_type="ifor",
                                 normalize_trend=False, n_top=10, outliers_fraction=0.1, log_transform=False):
    """ Finds anomalous regions in time series using standard unsupervised detectors

    First the time series is chopped up into windows ('shingles').
    Then, a standard anomaly detector is run.
    """
    x = w = None
    n = 0
    ts_data = data

    if log_transform:
        # log-transform now since the values are positive (in context of
        # many real-world datasets line airline); otherwise, values become
        # negative after de-trending
        ts_data = log_transform_series(ts_data, eps=1.0)

    if normalize_trend:
        # remove trend from series
        ts_data = difference_series(ts_data)

    ts = TSeries(ts_data, y=None)
    for x_, _, w in ts.get_shingles(window_size, skip_size=skip_size, batch_size=-1):
        x = np.reshape(x_, newshape=(x_.shape[0], -1))
        n = x.shape[0]
        logger.debug("Total instances: %d" % n)
        # logger.debug("Windows:\n%s" % str(w))

    if False:
        feature_ranges = get_sample_feature_ranges(x)
        logger.debug("feature_ranges:\n%s" % str(feature_ranges))

    scores = None
    if ad_type == "ocsvm":
        ad = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)
        ad.fit(x)
        scores = -ad.decision_function(x).reshape((n,))
    elif ad_type == "ifor":
        ad = IsolationForest(max_samples=min(256, x.shape[0]), contamination=outliers_fraction, random_state=None)
        ad.fit(x)
        scores = -ad.decision_function(x)
    elif ad_type == "lof":
        ad = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
        ad.fit(x)
        scores = -ad._decision_function(x)
    elif ad_type == "autoenc":
        n_hiddens = max(1, window_size//2)
        ad = AutoencoderAnomalyDetector(n_inputs=x.shape[1], n_neurons=[300, n_hiddens, 300],
                                        normalize_scale=True,
                                        activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None])
        ad.fit(x)
        scores = -ad.decision_function(x)

    top_anoms = np.argsort(-scores)[0:n_top]
    logger.debug("top scores (%s):\n%s\n%s" % (ad_type, str(top_anoms), str(scores[top_anoms])))

    pdfpath = "temp/timeseries/timeseries_shingles_%s_w%d%s_%s.pdf" % \
              (dataset, window_size, "" if not log_transform else "_log", ad_type)
    dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=1)

    # plot the timeseries anomalies with the detrended series
    pl = dp.get_next_plot()
    pl.set_xlim([0, ts.samples.shape[0]])
    pl.plot(np.arange(0, ts.samples.shape[0]), ts.samples, 'b-', linewidth=0.5)

    for i in top_anoms:
        if w[i] + window_size <= len(ts.samples):
            pl.plot(np.arange(w[i], w[i] + window_size), ts.samples[w[i]:(w[i] + window_size)], 'r-')

    if normalize_trend:
        # plot the original series with anomalous windows
        pl = dp.get_next_plot()
        pl.set_xlim([0, data.shape[0]])
        pl.plot(np.arange(0, data.shape[0]), data, 'b-', linewidth=0.5)

        for i in top_anoms:
            if w[i] + window_size <= len(data):
                pl.plot(np.arange(w[i], w[i] + window_size), data[w[i]:(w[i] + window_size)], 'r-')

    dp.close()


def read_ts(dataset):
    if not (dataset == 'synthetic' or dataset in univariate_timeseries_datasets):
        datasets = univariate_timeseries_datasets.keys()
        datasets.append('synthetic')
        print ("Invalid dataset: %s. Supported datasets: %s" %
              (dataset, datasets))
        return None
    if dataset == 'synthetic':
        samples = pd.read_csv("../datasets/simulated_timeseries/samples_2000.csv",
                              header=None, sep=",", usecols=[1]
                              )
        samples = np.asarray(samples, dtype=np.float32)
        samples = samples[:2000, :]
    else:
        samples = get_univariate_timeseries_data(dataset)
        samples = np.array(samples, dtype=float)
    return samples


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/timeseries")  # for logging and plots

    args = get_command_args(debug=False,
                            debug_args=["--dataset=airline", "--debug",
                                        "--plot", "--n_lags=20", "--algo=autoenc",
                                        "--log_file=temp/timeseries/timeseries_shingles.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    random.seed(42)
    rnd.seed(42)

    allowed_algos = {'autoenc': 'Auto-encoder',
                     'ifor': 'Isolation Forest',
                     'ocsvm': 'One-class SVM',
                     'lof': 'Local Outlier Factor'}
    if args.algo not in allowed_algos.keys():
        print ("Invalid algo: %s. Allowed algos:" % args.algo)
        for key, val in allowed_algos.iteritems():
            print ("  %s: %s" % (key, val))
        exit(0)

    skip_size = None
    n_anoms = 10
    i = 0
    data = read_ts(args.dataset)
    if data is None:
        exit(0)

    if False:
        logger.debug("samples:\n%s" % str(data.samples[:, 0]))
        for x, _, w in data.get_shingles(window_size, skip_size=skip_size, batch_size=200):
            logger.debug("batch:\n%s" % str(np.reshape(x, newshape=(-1, window_size))))

    find_anomalies_with_shingles(args.dataset, data, window_size=args.n_lags, skip_size=None,
                                 ad_type=args.algo, normalize_trend=args.normalize_trend,
                                 n_top=n_anoms, outliers_fraction=0.1, log_transform=args.log_transform)
