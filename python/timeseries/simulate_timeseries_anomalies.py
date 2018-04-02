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

    acts_sub = TSeries(acts_full.samples[0:2000, :], y=acts_full.y[0:2000, :])
    # logger.debug("acts_sub:\n%s" % str(np.hstack([acts_sub.samples, acts_sub.y])))

    n_lags = 5
    skip_size = None
    i = 0
    for x, y in acts_sub.get_batches(n_lags, 30):
        # logger.debug("batch[%d]:\n%s" % (i, str(x)))
        # logger.debug("batch[%d]:\n%s" % (i, str(y)))
        logger.debug("batch[%d]:\n%s" % (i, str(np.reshape(x, newshape=(x.shape[0], -1)))))
        i += 1
    # logger.debug("y:\n%s" % str(acts_sub.y))
