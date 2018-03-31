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

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Simple time-series modeling with Tensorflow RNN and LSTM cells

To execute:
pythonw -m timeseries.simulate_timeseries_anomalies
"""


def find_anomalies_with_shingles(actions, n_lags=5, skip_size=None, ad_type="ifor", n_top=10, outliers_fraction=0.1):
    """ Finds anomalous regions in time series using standard unsupervised detectors

    First the time series is chopped up into windows ('shingles').
    Then, a standard anomaly detector is run.
    """
    x = y = w = None
    n = 0
    for x_, y, w in actions.get_shingles(n_lags, skip_size=skip_size, batch_size=-1):
        x = np.reshape(x_, newshape=(x_.shape[0], -1))
        n = x.shape[0]
        logger.debug("Total instances: %d" % n)
        # logger.debug("Windows:\n%s" % str(w))

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

    top_anoms = np.argsort(-scores)[0:n_top]
    logger.debug("top scores (%s):\n%s\n%s" % (ad_type, str(top_anoms), str(scores[top_anoms])))

    pdfpath = "temp/timeseries/timeseries_simulation_anomalies_%s.pdf" % ad_type
    dp = DataPlotter(pdfpath=pdfpath, rows=3, cols=1)
    pl = dp.get_next_plot()
    pl.set_xlim([0, actions.samples.shape[0]])
    pl.plot(np.arange(0, actions.samples.shape[0]), actions.samples, 'b-', linewidth=0.5)

    for i in top_anoms:
        #  the first windows contain insufficient lag info (past values are zero)
        if i - n_lags < 0: continue
        pl.plot(np.arange((w[i] - n_lags), w[i]), actions.samples[(w[i] - n_lags):w[i]], 'r-')
    dp.close()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/timeseries/simulate_timeseries_anomalies.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    random.seed(42)
    rnd.seed(42)

    acts_full = read_activity_data()
    logger.debug("samples: %s, activities: %s, starts: %s" %
                 (str(acts_full.samples.shape), str(acts_full.activities.shape), str(acts_full.starts.shape)))

    acts_sub = ActivityData(acts_full.samples[0:2000, :], y=acts_full.y[0:2000, :])
    # logger.debug("acts_sub:\n%s" % str(np.hstack([acts_sub.samples, acts_sub.y])))

    n_lags = 5
    skip_size = None
    i = 0
    if False:
        for x, y in acts_sub.get_batches(n_lags, 30):
            # logger.debug("batch[%d]:\n%s" % (i, str(x)))
            # logger.debug("batch[%d]:\n%s" % (i, str(y)))
            logger.debug("batch[%d]:\n%s" % (i, str(np.reshape(x, newshape=(x.shape[0], -1)))))
            i += 1
    else:
        # ad_type = "ocsvm"
        ad_type = "ifor"
        # ad_type = "lof"
        find_anomalies_with_shingles(acts_sub, n_lags=n_lags, skip_size=skip_size,
                                     ad_type=ad_type, n_top=10)

    # logger.debug("y:\n%s" % str(acts_sub.y))
