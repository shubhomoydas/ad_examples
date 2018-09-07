import random
import numpy as np
import numpy.random as rnd
from common.utils import *
from common.data_plotter import *
from common.nn_utils import *

from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor as MLPRegressor_SK

from common.timeseries_datasets import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
To execute:
pythonw -m timeseries.timeseries_regression --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_regression.log --normalize_trend --algo=nnsk --n_lags=12 --dataset=airline
pythonw -m timeseries.timeseries_regression --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_regression.log --normalize_trend --algo=nntf --n_lags=5 --dataset=shampoo
pythonw -m timeseries.timeseries_regression --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_regression.log --normalize_trend --algo=nntf --n_lags=12 --dataset=lynx
pythonw -m timeseries.timeseries_regression --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_regression.log --normalize_trend --algo=nntf --n_lags=12 --dataset=aus_beer
pythonw -m timeseries.timeseries_regression --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_regression.log --normalize_trend --algo=nntf --n_lags=12 --dataset=us_accident
pythonw -m timeseries.timeseries_regression --n_epochs=200 --debug --log_file=temp/timeseries/timeseries_regression.log --normalize_trend --algo=nnsk --n_lags=50 --dataset=wolf_sunspot

pythonw -m timeseries.timeseries_regression --dataset=fisher_temp --algo=nntf --n_lags=20 --n_epochs=100 --debug --log_file=temp/timeseries/timeseries_regression.log

"""


def find_anomalies_with_regression(data, dataset, n_lags=5, reg_type="svr", n_anoms=10,
                                   normalize_trend=False, batch_size=10, n_epochs=200, tr_frac=0.8):
    """ Finds anomalous time points in time series using standard regression algorithms

    SVR might be fine even if trend is not removed, but when using
    RandomForestRegressor, make sure that the trend is removed.
    Ideally, remove trend and standardize for all timeseries.
    """

    n = data.shape[0]
    n_tr = int(n * tr_frac)

    if normalize_trend:
        # remove trend from training series
        diff_data = difference_series(data)
    else:
        diff_data = data
    diff_tr = diff_data[:n_tr]
    diff_test = diff_data[n_tr:]  # will be used to score predictions

    # normalize to range (-1, 1)
    normalizer = DiffScale()
    scaled_tr = normalizer.fit_transform(diff_tr, normalize_trend=False)
    scaled_test = normalizer.scale(diff_test)  # will be used to score predictions later

    ts = prepare_tseries(scaled_tr, name=dataset)

    if False:
        logger.debug("diff_data:(%s, %s, %s, %s)\n%s\n%s" %
                     (str(data.shape), str(diff_data.shape), str(diff_tr.shape), str(diff_test.shape),
                      str(scaled_tr[:, 0]), str(scaled_test[:, 0])))

    if False:
        # debug only
        logger.debug("samples:(%s)\n%s" % (str(ts.samples.shape), str(ts.samples[:, 0])))
        for x, y in ts.get_batches(n_lags=n_lags, batch_size=-1, single_output_only=True):
            batch = np.hstack([y, np.reshape(x, newshape=(x.shape[0], -1))])
            logger.debug("batch:\n%s" % str(batch))
        return

    x = y = None
    # read the entire timeseries in one batch (batch_size=-1)
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
    elif reg_type == "nntf":
        # use tensorflow
        mdl = MLPRegressor_TF(x.shape[1], 100, 1, batch_size=batch_size, shuffle=True,
                              n_epochs=n_epochs, l2_penalty=0.01)
    elif reg_type == "nnsk":
        # use scikit-learn
        mdl = MLPRegressor_SK(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                              alpha=0.0001, batch_size='auto', learning_rate='constant',
                              learning_rate_init=0.001, power_t=0.5, max_iter=n_epochs, shuffle=True,
                              random_state=None, tol=0.0001, verbose=False, warm_start=False,
                              momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                              validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
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
        # prev_points.append(yhat)  # to predict far into the future

    logger.debug("pred_points:(%d)\n%s" % (len(pred_points[scaled_tr.shape[0]:]), str(list(pred_points[scaled_tr.shape[0]:]))))

    # now invert all transformations
    pred_points = np.reshape(pred_points, newshape=(-1, ts.y.shape[1]))
    pred_points = normalizer.inverse_transform(pred_points)
    if normalize_trend:
        pred_points = invert_difference_series(pred_points, data[[0], :])

    scores = np.abs(data[n_tr:, 0] - pred_points[n_tr:, 0])

    top_anoms = np.argsort(-scores)[0:n_anoms]
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


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/timeseries")  # for logging and plots

    args = get_command_args(debug=False,
                            debug_args=["--dataset=airline", "--algo=nntf", "--n_lags=12",
                                        "--n_anoms=10", "--debug", "--plot",
                                        "--log_file=temp/timeseries/timeseries_regression.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    random.seed(42)
    rnd.seed(42)

    reg_type = args.algo  # "nntf" # "nnsk" # "rfor" # "svr"
    n_anoms = args.n_anoms
    n_lags = args.n_lags
    n_epochs = args.n_epochs
    normalize_trend = args.normalize_trend
    batch_size = 20

    allowed_algos = {'nnsk': 'Multilayer Perceptron based Regression',
                     'nntf': 'Linear Regression with TensorFlow Neural Net',
                     'rfor': 'Random Forest Regression',
                     'svr':  'Support Vector Regression'}
    if args.algo not in allowed_algos.keys():
        print ("Invalid algo: %s. Allowed algos:" % args.algo)
        for key, val in allowed_algos.iteritems():
            print ("  %s: %s" % (key, val))
        exit(0)

    dataset = args.dataset
    # dataset = "airline"
    logger.debug("dataset: %s, reg_type: %s" % (dataset, reg_type))
    if not dataset in univariate_timeseries_datasets:
        print ("Invalid dataset: %s. Supported datasets: %s" %
              (dataset, str(univariate_timeseries_datasets.keys())))
        exit(0)
    data = get_univariate_timeseries_data(dataset)

    find_anomalies_with_regression(np.array(data, dtype=float), dataset, n_lags=n_lags,
                                   reg_type=reg_type, normalize_trend=normalize_trend,
                                   batch_size=batch_size,
                                   n_epochs=n_epochs, n_anoms=n_anoms)
