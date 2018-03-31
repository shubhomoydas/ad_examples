import warnings
from statsmodels.tsa import stattools
from statsmodels.tsa.arima_model import ARIMA
from common.timeseries_datasets import *
from common.data_plotter import *
from common.utils import *


'''
Before we try to identify anomalies, it is important to first model the timeseries
and understand how to forecast/predict. This file first performs very simple model 
fitting with ARIMA (single univariate timeseries) on the initial 2/3 data, and then 
detects time points with largest forecasting errors on the last 1/3 data. Are these
'anomalies'? - depends on the application context.

Some examples motivated by:
    https://machinelearningmastery.com/arima-for-time-tr_series-forecasting-with-python/

pythonw -m timeseries.timeseries_arima
'''


def time_lag_diff(series):
    tmp = np.zeros(series.shape, dtype=series.dtype)
    n = len(tmp)
    for i in xrange(1, n):
        tmp[i] = series[i] - series[i-1]
    return tmp


def fit_ARIMA(series, order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # hack the time index, else ARIMA will not run
        model = ARIMA(series, dates=pd.to_datetime(np.arange(0, len(series))), order=order)
        model_fit = model.fit(disp=0)
        residuals = model_fit.resid
    return residuals


def rolling_forecast_ARIMA(train, test, order, nsteps=1):
    tseries = [x for x in train]
    rets = []
    errors = []
    tindex = pd.to_datetime(np.arange(0, len(train) + nsteps))
    for i in range(nsteps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # hack the time index, else ARIMA will not run
            model = ARIMA(tseries, dates=tindex[0:len(tseries)], order=order)
            model_fit = model.fit(disp=-1)
            forecasts = model_fit.forecast()
            val = forecasts[0][0]
            rets.append(val)
            errors.append(test[i] - val)
            tseries.append(test[i])
    return np.array(rets, dtype=float), np.array(errors, dtype=float)


def plot_lag_difference():
    dataset = "airline"
    data = get_univariate_timeseries_data(dataset)
    tseries = np.array(data.iloc[:, 0], dtype=float)
    diffs = time_lag_diff(tseries)
    pdfpath = "temp/timeseries_lag_diff_%s.pdf" % dataset
    dp = DataPlotter(pdfpath=pdfpath, rows=3, cols=1)

    pl = dp.get_next_plot()
    plt.title("Time tr_series %s" % dataset, fontsize=8)
    pl.plot(np.arange(0, len(diffs)), diffs, 'b-')
    dp.close()


def forecast_and_report_anomalies():

    # datasets = univariate_timeseries_datasets.keys()
    datasets = ["airline"]

    for dataset in datasets:
        logger.debug("dataset: %s" % dataset)
        data = get_univariate_timeseries_data(dataset)
        tseries = np.array(data.iloc[:, 0], dtype=float)
        logger.debug("timeseries[%d]:\n%s" % (len(tseries), str(list(tseries))))

        acf = stattools.acf(tseries, nlags=40)
        logger.debug("acf:\n%s" % str(list(acf)))

        pacf = stattools.pacf(tseries, nlags=40)
        logger.debug("pacf:\n%s" % str(list(pacf)))

        dataset_def = univariate_timeseries_datasets[dataset]
        order = dataset_def.ARIMA_order

        residuals = fit_ARIMA(tseries, order=order)
        logger.debug("residuals[%d]:\n%s" % (len(residuals), str(list(residuals))))

        train_sz = int(len(tseries) * 0.66)
        train = tseries[0:train_sz]
        test = tseries[train_sz:len(tseries)]
        forecasts, errors = rolling_forecast_ARIMA(train, test, order=order, nsteps=len(tseries) - train_sz)
        logger.debug("forecasts[%d]:\n%s" % (len(forecasts), str(list(forecasts))))
        logger.debug("errors[%d]:\n%s" % (len(errors), str(list(errors))))

        # identify the times at which we had largest forecasting errors
        err_ordered = np.argsort(-np.abs(errors))[0:5]
        logger.debug("largest errors[%d]:\n%s" % (len(err_ordered), str(list(errors[err_ordered]))))

        if True:
            pdfpath = "temp/timeseries_plot_%s.pdf" % dataset
            dp = DataPlotter(pdfpath=pdfpath, rows=3, cols=1)

            pl = dp.get_next_plot()
            plt.title("Time tr_series %s" % dataset, fontsize=8)
            pl.plot(np.arange(0, len(tseries)), tseries, 'b-')

            pl = dp.get_next_plot()
            plt.title("ACF %s" % dataset, fontsize=8)
            pl.plot(np.arange(0, len(acf)), acf, 'b-')

            pl = dp.get_next_plot()
            plt.title("PACF %s" % dataset, fontsize=8)
            pl.plot(np.arange(0, len(pacf)), pacf, 'b-')

            pl = dp.get_next_plot()
            plt.title("Residuals %s" % dataset, fontsize=8)
            pl.plot(np.arange(0, len(residuals)), residuals, 'b-')

            pl = dp.get_next_plot()
            plt.title("Forecast %s" % dataset, fontsize=8)
            pl.plot(np.arange(0, len(tseries)), tseries, 'b-')
            pl.plot(np.arange(train_sz, train_sz + len(forecasts)), forecasts, 'r-')
            # mark anomalous time points
            for x in err_ordered:
                plt.axvline(x + train_sz, color='g')
            dp.close()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--log_file=temp/timeseries_explore.log", "--debug"])
    configure_logger(args)

    np.random.seed(42)

    if True:
        forecast_and_report_anomalies()
    else:
        plot_lag_difference()