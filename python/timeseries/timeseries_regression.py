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


def find_anomalies_with_regression(data, dataset, n_lags=5, reg_type="svr", n_top=10, tr_frac=0.8):
    """ Finds anomalous time points in time series using standard regression algorithms

    SVR might be fine even if trend is not removed, but when using
    RandomForestRegressor, make sure that the trend is removed.
    Ideally, remove trend and standardize for all timeseries.
    """

    n = data.shape[0]
    n_tr = int(n * tr_frac)

    # remove trend from training series
    diff_data = difference_series(data, interval=1)

    diff_tr = diff_data[:(n_tr-1)]  # we loose one time point due to differencing
    diff_test = diff_data[(n_tr-1):]  # will be used to score predictions

    # normalize by mean and variance
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(diff_tr)
    scaled_tr = scaler.transform(diff_tr)
    scaled_test = scaler.transform(diff_test)  # will be used to score predictions later

    ts = prepare_tseries(scaled_tr, name=dataset)

    if False:
        # debug only
        logger.debug("samples:(%s)\n%s" % (str(ts.samples.shape), str(ts.samples[:, 0])))
        for x, y in ts.get_batches(n_lags=n_lags, batch_size=-1, single_output_only=True):
            batch = np.hstack([y, np.reshape(x, newshape=(x.shape[0], -1))])
            logger.debug("batch:\n%s" % str(batch))
        return

    x = y = None
    for x_, y_ in ts.get_batches(n_lags, batch_size=-1, single_output_only=True):
        x = np.reshape(x_, newshape=(x_.shape[0], -1))
        y = np.reshape(y_, newshape=(-1,))

    if False:
        logger.debug("Total train instances: %d" % x.shape[0])
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

    # remember: x, y are only the train series
    mdl.fit(x, y)

    # Predict on test set. Assumes that all previous values at a time point as known
    pred_points = list(scaled_tr[:, 0])
    prev_points = list(scaled_tr[:, 0])
    for i in range(len(scaled_test)):
        yhat = mdl.predict(np.reshape(np.array(prev_points[-n_lags:]), newshape=(1, n_lags)))[0]
        logger.debug("%d yhat: %f" % (i, yhat))
        pred_points.append(yhat)
        prev_points.append(scaled_test[i, 0])  # to predict just one step ahead of known values
        # prev_points.append(yhat)  # to predict far into future

    logger.debug("pred_points:(%d)\n%s" % (len(pred_points), str(pred_points)))

    # now invert all transformations
    pred_points = np.reshape(pred_points, newshape=(-1, ts.y.shape[1]))
    pred_points = scaler.inverse_transform(pred_points)
    pred_points = np.vstack([np.zeros(shape=(1, ts.y.shape[1]), dtype=np.float32), pred_points])
    pred_points = invert_difference_series(pred_points, data[[0], :])
    logger.debug("inv_diffs:(%s)\n%s" % (str(pred_points.shape), str(pred_points)))

    scores = np.abs(data[n_tr:, 0] - pred_points[n_tr:, 0])

    top_anoms = np.argsort(-scores)[0:n_top]
    logger.debug("top scores (%s):\n%s\n%s" % (reg_type, str(top_anoms), str(scores[top_anoms])))

    pdfpath = "temp/timeseries/timeseries_regression_%s_%d_%s.pdf" % (ts.name, n_lags, reg_type)
    dp = DataPlotter(pdfpath=pdfpath, rows=3, cols=1)
    pl = dp.get_next_plot()
    pl.set_xlim([0, data.shape[0]])
    pl.plot(np.arange(0, n), data[:, 0], 'b-', linewidth=0.5)
    # pl.plot(np.arange(0, n), pred_points[:, 0], 'g-', linewidth=0.5)
    pl.plot(np.arange(n_tr, n), pred_points[n_tr:, 0], 'r-', linewidth=0.5)
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
    n_lags = 12
    skip_size = None
    n_anoms = 10

    # datasets = univariate_timeseries_datasets.keys()
    dataset = "airline"
    logger.debug("dataset: %s, reg_type: %s" % (dataset, reg_type))
    data = get_univariate_timeseries_data(dataset)

    find_anomalies_with_regression(np.array(data, dtype=float), dataset, n_lags=n_lags, reg_type=reg_type, n_top=n_anoms)
