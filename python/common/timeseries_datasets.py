import numpy as np
import pandas as pd
# from pandas import datetime


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


univariate_timeseries_datasets = {
    # "name": (path, use_cols, date_columns, index_column, ARIMA_order)
    "airline": TsFileDef("AirlinePassengers/international-airline-passengers.csv", [1], False, None, (4, 1, 1)),
    "aus_beer": TsFileDef("AustralianBeerProduction/quarterly-beer-production-in-aus.csv", [2], False, None, (4, 1, 0)),
    "lynx": TsFileDef("CanadianLynx/lynx_trappings.csv", [1], False, None, (4, 1, 0)),
    "fisher_temp": TsFileDef("FisherRiver/mean-daily-temperature-fisher-ri.csv", [1], False, None, (4, 1, 0)),
    "shampoo": TsFileDef("ShampooSales/sales-of-shampoo-over-a-three-ye.csv", [1], [0], None, (5, 1, 0)),
    "us_accident": TsFileDef("USAccidentalDeaths/accidental-deaths-in-usa-monthly.csv", [1], False, None, (4, 1, 0)),
    "wolf_sunspot": TsFileDef("WolfSunSpot/wolfs-sunspot-numbers-1700-1988.csv", [1], False, None, (4, 1, 0))
}


def get_univariate_timeseries_data(dataset):
    dataset_def = univariate_timeseries_datasets[dataset]
    data = pd.read_csv("../datasets/%s" % dataset_def.path,
                       header=0, sep=",", usecols=dataset_def.usecols,
                       # parse_dates=dataset_def.date_columns, index_col=dataset_def.index_column,
                       # squeeze=True, date_parser=date_parser
                       )
    return data


def difference_series(series, interval=1):
    n, m = series.shape
    diffs = np.zeros((n-interval, m), dtype=series.dtype)
    for i in range(interval, n):
        diffs[i-interval, :] = series[i, :] - series[i-interval, :]
    return diffs


def invert_difference_series_old(history, series, interval=1):
    n, m = series.shape
    inv = np.zeros((n, m), dtype=series.dtype)
    for i in range(n):
        inv[i, :] = series[i, :] + history[i, :]
    return inv


def invert_difference_series(series, initial, interval=1):
    n, m = series.shape
    inv = np.zeros((n, m), dtype=series.dtype)
    prev = initial
    for i in range(n):
        prev = series[i, :] + prev
        inv[i, :] = prev
    return inv


def prepare_n_lag_samples(series, n_lag=1):
    new = [series]
    tmp = series
    for i in range(n_lag):
        tmp = np.roll(tmp, 1, axis=0)
        tmp[0, :] = 0
        new.append(tmp)
    new.reverse()
    return new


class TSeries(object):
    def __init__(self, tseries):
        self.tseries = tseries
        self.dim = self.tseries[0].shape[1]
        self.series_len = self.tseries[0].shape[0]
        self.n_lag = len(self.tseries) - 1

    def iter_series_batch(self, i, batch_size):
        n = self.series_len
        start = i * batch_size
        end = min(n, start + batch_size)
        if start >= self.series_len:
            return None, None, None
        # arrs = list()
        # for arr in self.tseries:
        #     tmp = arr[start:end, :]
        return [arr[start:end, :] for arr in self.tseries], start, end

    def iter_series_batch_rnn(self, i, batch_size):
        batch, start, end = self.iter_series_batch(i, batch_size)
        # return x, y
        return np.stack(batch[0:(self.n_lag)], axis=1), batch[self.n_lag], start, end

