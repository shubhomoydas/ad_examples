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

pythonw -m timeseries.timeseries_shingles
"""


def find_anomalies_with_shingles(ts, window_size=5, skip_size=None, ad_type="ifor", n_top=10, outliers_fraction=0.1):
    """ Finds anomalous regions in time series using standard unsupervised detectors

    First the time series is chopped up into windows ('shingles').
    Then, a standard anomaly detector is run.
    """
    x = w = None
    n = 0
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
        ad = IsolationForest(max_samples=256, contamination=outliers_fraction, random_state=None)
        ad.fit(x)
        scores = -ad.decision_function(x)
    elif ad_type == "lof":
        ad = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
        ad.fit(x)
        scores = -ad._decision_function(x)
    elif ad_type == "autoenc":
        ad = AutoencoderAnomalyDetector(n_inputs=x.shape[1], n_neurons=[300, 10, 300],
                                        normalize_scale=True,
                                        activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None])
        ad.fit(x)
        scores = -ad.decision_function(x)

    top_anoms = np.argsort(-scores)[0:n_top]
    logger.debug("top scores (%s):\n%s\n%s" % (ad_type, str(top_anoms), str(scores[top_anoms])))

    pdfpath = "temp/timeseries/timeseries_shingles_w%d_%s.pdf" % (window_size, ad_type)
    dp = DataPlotter(pdfpath=pdfpath, rows=3, cols=1)
    pl = dp.get_next_plot()
    pl.set_xlim([0, ts.samples.shape[0]])
    pl.plot(np.arange(0, ts.samples.shape[0]), ts.samples, 'b-', linewidth=0.5)

    for i in top_anoms:
        if w[i] + window_size <= len(ts.samples):
            pl.plot(np.arange(w[i], w[i] + window_size), ts.samples[w[i]:(w[i] + window_size)], 'r-')
    dp.close()


def read_ts():
    samples = pd.read_csv("../datasets/simulated_timeseries/samples_2000.csv",
                          header=None, sep=",", usecols=[1]
                          )
    samples = np.asarray(samples, dtype=np.float32)
    return TSeries(samples[:2000, :], y=None)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/timeseries/timeseries_shingles.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    dir_create("./temp/timeseries")  # for logging and plots

    random.seed(42)
    rnd.seed(42)

    ad_type = "autoenc"
    window_size = 20
    skip_size = None
    n_anoms = 10
    i = 0
    ts = read_ts()

    if False:
        logger.debug("samples:\n%s" % str(ts.samples[:, 0]))
        for x, _, w in ts.get_shingles(window_size, skip_size=skip_size, batch_size=200):
            logger.debug("batch:\n%s" % str(np.reshape(x, newshape=(-1, window_size))))

    find_anomalies_with_shingles(ts, window_size=window_size, skip_size=None,
                                 ad_type=ad_type, n_top=n_anoms, outliers_fraction=0.1)
