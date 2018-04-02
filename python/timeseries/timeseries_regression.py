import random
import numpy as np
import numpy.random as rnd
from sklearn.preprocessing import MinMaxScaler
from common.utils import *
from common.data_plotter import *

from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

from common.timeseries_datasets import *


"""
pythonw -m timeseries.timeseries_regression
"""


def find_anomalies_with_regression(ts, n_lags=5, reg_type="svr", n_top=10, tr_frac=0.8):
    """ Finds anomalous time points in time series using standard regression algorithms

    SVR might be fine even if trend is not removed, but when using
    RandomForestRegressor, make sure that the trend is removed.
    Ideally, remove trend and standardize for all timeseries.
    """
    x = y = None
    n = 0
    for x_, y_ in ts.get_batches(n_lags, batch_size=-1, single_output_only=True):
        x = np.reshape(x_, newshape=(x_.shape[0], -1))
        y = np.reshape(y_, newshape=(-1,))
        n = x.shape[0]

    if False:
        logger.debug("Total instances: %d" % n)
        logger.debug("y:\n%s" % str(y))

    if reg_type == "svr":
        mdl = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
                      kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    elif reg_type == "rfor":
        mdl = RandomForestRegressor(n_estimators=10, criterion="mse", max_depth=None,
                                    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                    max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
                                    random_state=None, verbose=0, warm_start=False)
    else:
        raise ValueError("unsupported regression type: %s" % reg_type)

    n_tr = int(n * tr_frac)
    mdl.fit(x[:n_tr], y[:n_tr])

    # Predict on test set. Assumes that all previous values at a time point as known
    preds = mdl.predict(x[n_tr:])
    scores = np.abs(y[n_tr:] - preds)

    top_anoms = np.argsort(-scores)[0:n_top]
    logger.debug("top scores (%s):\n%s\n%s" % (reg_type, str(top_anoms), str(scores[top_anoms])))

    pdfpath = "temp/timeseries/timeseries_regression_%s_%d_%s.pdf" % (ts.name, n_lags, reg_type)
    dp = DataPlotter(pdfpath=pdfpath, rows=3, cols=1)
    pl = dp.get_next_plot()
    pl.set_xlim([0, ts.samples.shape[0]])
    pl.plot(np.arange(0, x.shape[0]), y, 'b-', linewidth=0.5)
    pl.plot(np.arange(n_tr, x.shape[0]), preds, 'r-', linewidth=0.5)
    for i in top_anoms:
        plt.axvline(i+n_tr, color='g', linewidth=0.5)
    dp.close()


def read_ts():
    samples = pd.read_csv("../datasets/simulated_timeseries/samples_2000.csv",
                          header=None, sep=",", usecols=[1]
                          )
    samples = np.asarray(samples, dtype=np.float32)
    return TSeries(samples[:2000, :], y=None)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True,
                            debug_args=["--debug",
                                        "--plot",
                                        "--log_file=temp/timeseries/timeseries_regression.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    dir_create("./temp/timeseries")  # for logging and plots

    random.seed(42)
    rnd.seed(42)

    reg_type = "svr"  # "rfor" # "svr"
    n_lags = 5
    skip_size = None
    n_anoms = 10

    # datasets = univariate_timeseries_datasets.keys()
    dataset = "airline"
    logger.debug("dataset: %s, reg_type: %s" % (dataset, reg_type))
    data = get_univariate_timeseries_data(dataset)
    ts = prepare_tseries(np.array(data, dtype=float), name=dataset)

    if False:
        # debug only
        logger.debug("samples:(%s)\n%s" % (str(ts.samples.shape), str(ts.samples[:, 0])))
        for x, y in ts.get_batches(n_lags=n_lags, batch_size=-1, single_output_only=True):
            batch = np.hstack([y, np.reshape(x, newshape=(x.shape[0], -1))])
            logger.debug("batch:\n%s" % str(batch))
    else:
        find_anomalies_with_regression(ts, n_lags=n_lags, reg_type=reg_type, n_top=n_anoms)
