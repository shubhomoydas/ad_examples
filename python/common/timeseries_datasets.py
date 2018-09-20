import numpy as np
import pandas as pd
# from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
from common.utils import *


class TsFileDef(object):
    def __init__(self, path, usecols, date_columns, index_column, ARIMA_order):
        self.path = path
        self.usecols = usecols
        self.date_columns = date_columns
        self.index_column = index_column
        self.ARIMA_order = ARIMA_order


# def date_parser(x):
#     # date parser for shampoo dataset
#     return datetime.strptime('190'+x, '%Y-%m')


class DiffScale(object):
    def __init__(self):
        self.scaler = None

    def fit_transform(self, series, normalize_trend=False):
        if normalize_trend:
            # remove trend by differencing
            series = difference_series(series)
        # normalize to range (-1, 1); helps when output is tanh
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = scaler.fit(series)
        # logger.debug("scaler: (%f, %f), %f, %f" % (scaler.data_min_, scaler.data_max_, scaler.data_range_, scaler.scale_))
        scld_series = self.scaler.transform(series)
        # logger.debug("scld_series.shape: %s\n%s" % (str(scld_series.shape), str(scld_series)))
        return scld_series

    def scale(self, series):
        return self.scaler.transform(series)

    def inverse_transform(self, series, initial=None):
        # logger.debug("preds before inverse scale(%s):\n%s" % (str(series.shape), str(series[:, 0])))
        final_preds = self.scaler.inverse_transform(series)
        # logger.debug("final_preds after inverse scale(%s):\n%s" % (str(final_preds.shape), str(final_preds[:, 0])))
        if initial is not None and False:
            logger.debug("initial(%s):\n%s" % (str(initial.shape), str(initial)))
        if initial is not None:
            final_preds = invert_difference_series(final_preds, initial=initial)
        # logger.debug("final_preds.shape: %s\n%s" % (str(final_preds.shape), str(final_preds)))
        return final_preds


univariate_timeseries_datasets = {
    # "name": (path, use_cols, date_columns, index_column, ARIMA_order)
    # ARIMA_order: (AR, differences, MA, seasonality)
    "airline": TsFileDef("AirlinePassengers/international-airline-passengers.csv", [1], False, None, (0, 1, 1, 12)),
    "aus_beer": TsFileDef("AustralianBeerProduction/quarterly-beer-production-in-aus.csv", [2], False, None, (3, 1, 0, 4)),
    "lynx": TsFileDef("CanadianLynx/lynx_trappings.csv", [1], False, None, (4, 1, 1, 12)),
    "fisher_temp": TsFileDef("FisherRiver/mean-daily-temperature-fisher-ri.csv", [1], False, None, (4, 1, 0)),
    "shampoo": TsFileDef("ShampooSales/sales-of-shampoo-over-a-three-ye.csv", [1], [0], None, (5, 1, 0)),
    "us_accident": TsFileDef("USAccidentalDeaths/accidental-deaths-in-usa-monthly.csv", [1], False, None, (4, 1, 0, 12)),
    "wolf_sunspot": TsFileDef("WolfSunSpot/wolfs-sunspot-numbers-1700-1988.csv", [1], False, None, (4, 1, 1, 12))
}


def get_univariate_timeseries_data(dataset):
    if not dataset in univariate_timeseries_datasets:
        raise ValueError("Allowed datasets: %s" % str(univariate_timeseries_datasets.keys()))
    dataset_def = univariate_timeseries_datasets[dataset]
    data = pd.read_csv("../datasets/%s" % dataset_def.path,
                       header=0, sep=",", usecols=dataset_def.usecols,
                       # parse_dates=dataset_def.date_columns, index_col=dataset_def.index_column,
                       # squeeze=True, date_parser=date_parser
                       )
    return data


def log_transform_series(series, eps=1.0):
    """ Transform a series by element-wise log
    transformed_series = log(series + eps)
    :param series: np.ndarray
    :param eps: real
    :return: np.ndarray
    """
    new_series = np.log(series + eps)
    return new_series


def inverse_log_transform_series(series, eps=1.0):
    """ Invert a previous element-wise log-transform
    inverted_series = exp(series) - eps
    :param series: np.ndarray
    :param eps: real
    :return: np.ndarray
    """
    new_series = np.exp(series) - eps
    return new_series


def difference_series(series):
    n, m = series.shape
    diffs = np.zeros((n, m), dtype=series.dtype)
    for i in range(1, n):
        diffs[i, :] = series[i, :] - series[i-1, :]
    return diffs


def invert_difference_series_old(history, series):
    n, m = series.shape
    inv = np.zeros((n, m), dtype=series.dtype)
    for i in range(n):
        inv[i, :] = series[i, :] + history[i, :]
    return inv


def invert_difference_series(series, initial):
    n, m = series.shape
    inv = np.zeros((n, m), dtype=series.dtype)
    inv[0, :] = initial + series[0, :]
    for i in range(1, n):
        inv[i, :] = inv[i-1, :] + series[i, :]
    return inv


class TSeries(object):
    """ Provides simple APIs for iterating over a timeseries and activities """
    def __init__(self, samples, y=None, activities=None, starts=None, name=None):
        self.samples = samples
        self.y = y
        self.series_len = self.samples.shape[0]
        self.dim = self.samples.shape[1]
        self.activities = activities
        self.starts = starts
        self.name = name

        if self.y is None and self.activities is not None and self.starts is not None:
            # populate the activity labels
            n = self.samples.shape[0]
            n_acts = self.activities.shape[0]
            self.y = np.zeros(shape=(n, 1), dtype=int)
            for i in range(n_acts):
                s = self.starts[i, 0]
                e = n if i == n_acts-1 else self.starts[i+1, 0]
                self.y[s:e, 0] = self.activities[i, 0]

    def get_batches(self, n_lags, batch_size, single_output_only=False):
        """ Iterate over timeseries where current values are functions of previous values

        The assumption is that the model is:
            (y_{t-n_lag+1}, y_{t}, y_{t+1},...) = f(x_{t-n_lag+1}, x_{t-1}, x_t,...)

        Args:
            n_lags: int
            batch_size: int
            single_output_only: boolean
                True:
                    Return only the current y-value for each time point.
                    This is useful when predicting only the last y-value
                False:
                    Return n_lags y-values for each time point

        returns: np.ndarray(shape=(batch_size, n_lags, d))
            where d = samples.shape[1]
            The data is ordered in increasing time
        """
        n = self.samples.shape[0]
        d_in = self.samples.shape[1]
        d_out = 0 if self.y is None else self.y.shape[1]
        batch_size = n if batch_size < 0 else batch_size
        for i in range(0, n, batch_size):
            x = np.zeros(shape=(batch_size, n_lags, d_in), dtype=np.float32)
            y = None
            if self.y is not None and not single_output_only:
                y = np.zeros(shape=(batch_size, n_lags, d_out), dtype=np.float32)
            e = min(n, i + batch_size)
            sz = e - i
            # logger.debug("i, e, sz: %d, %d, %d" % (i, e, sz))
            for t in range(n_lags):
                st = max(0, i - t)  # maximum time we can go back in the past
                et = e - t
                # logger.debug("st, et: %d, %d" % (st, et))
                if et >= st:
                    x[(sz-(et-st)):sz, n_lags - 1 - t, :] = self.samples[st:et, :]
                    if y is not None and not single_output_only:
                        y[(sz - (et - st)):sz, n_lags - 1 - t, :] = self.y[st:et, :]
                else:
                    break
            if self.y is not None and single_output_only:
                y = self.y[i:e, :]
            elif y is not None:
                y = y[0:sz, :, :]
            yield x[0:sz, :, :], y

    def get_shingles(self, window_size, skip_size=None, batch_size=100):
        """ Creates feature vectors out of windows of data and iterates over these

        The instances are of the form:
            (x_{t-window_size+1}, ..., x_t)

        returns: np.ndarray(shape=(batch_size, n_lags, d))
            where d = samples.shape[1]
        """
        skip_size = window_size if skip_size is None else skip_size
        n = self.samples.shape[0]
        d = self.samples.shape[1]
        if batch_size < 0:
            batch_size = 1 + n // skip_size
        x = np.zeros(shape=(batch_size, window_size, d), dtype=np.float32)
        w = np.zeros(batch_size, dtype=np.int)  # window start time
        y = None
        if self.y is not None: y = np.zeros(batch_size, dtype=np.int)
        l = 0
        for i in range(0, n, skip_size):
            et = min(n - i, window_size)
            x[l, 0:et, :] = self.samples[i:(i+et), :]
            w[l] = i
            if self.y is not None:
                y[l] = self.y[i]
            l += 1
            if l == batch_size or i + skip_size >= n:
                x_, y_, w_ = x, y, w
                if l < batch_size:
                    x_, y_, w_ = x[0:l, :, :], None if y is None else y[0:l], w[0:l]
                yield x_, y_, w_
                if i + skip_size < n:
                    l = 0
                    x = np.zeros(shape=(batch_size, window_size, d), dtype=np.float32)
                    w = np.zeros(batch_size, dtype=np.int)
                    if self.y is not None: y = np.zeros(batch_size, dtype=np.int)

    def log_batches(self, n_lags, batch_size, single_output_only=False):
        """ Debug API """
        logger.debug("Logging timeseries (nlags: %d, batch_size: %d)" % (n_lags, batch_size))
        i = 0
        for x, y in self.get_batches(n_lags, batch_size, single_output_only=single_output_only):
            logger.debug("x: (%s), y: (%s)" % (str(x.shape), "-" if y is None else str(y.shape)))
            if y is not None:
                # logger.debug("y[%d](%d)\n%s" % (i, y.shape[0], str(np.reshape(y, newshape=(-1,)))))
                if single_output_only:
                    batch = np.hstack([y, np.reshape(x, newshape=(x.shape[0],-1))])
                else:
                    batch = np.hstack([np.reshape(y, newshape=(y.shape[0],-1)), np.reshape(x, newshape=(x.shape[0], -1))])
            else:
                batch = x
            logger.debug("batch[%d](%d)\n%s" % (i, x.shape[0], str(np.round(batch, 3))))
            i += 1


def prepare_tseries(series, name=""):
    x = series[:-1]
    y = series[1:]
    return TSeries(x, y=y, name=name)
