import warnings
from statsmodels.tsa import stattools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    https://machinelearningmastery.com/arima-for-time-train_series-forecasting-with-python/

pythonw -m timeseries.timeseries_arima --debug --plot --log_file=temp/timeseries/timeseries_arima.log --log_transform --dataset=airline
'''


def time_lag_diff(series):
    tmp = np.zeros(series.shape, dtype=series.dtype)
    n = len(tmp)
    for i in range(1, n):
        tmp[i] = series[i] - series[i-1]
    return tmp


def fit_ARIMA(series, dates=None, order=(0, 0, 1)):
    """Fits either an ARIMA or a SARIMA model depending on whether order is 3 or 4 dimensional

    :param series:
    :param dates:
    :param order: tuple
        If this has 3 elements, an ARIMA model will be fit
        If this has 4 elements, the fourth is the seasonal factor and SARIMA will be fit
    :return: fitted model, array of residuals
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # hack the time index, else ARIMA will not run
        if dates is None:
            dates = pd.to_datetime(np.arange(1, len(series)+1))
        if len(order) > 3:
            seasonal_order = (0, 0, 0, order[3])
            arima_order = (order[0], order[1], order[2])
            model = SARIMAX(series, dates=dates, order=arima_order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=0)
            residuals = model_fit.resid
        else:
            model = ARIMA(series, dates=dates, order=order)
            model_fit = model.fit(disp=0)
            residuals = model_fit.resid
    return model_fit, residuals


def rolling_forecast_ARIMA(train, test, order, nsteps=1):
    tseries = [x for x in train]
    rets = []
    errors = []
    tindex = pd.to_datetime(np.arange(1, len(train) + nsteps + 1))
    for i in range(nsteps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # hack the time index, else ARIMA will not run
            model_fit, residuals = fit_ARIMA(tseries, dates=tindex[0:len(tseries)], order=order)
            if len(order) == 3:
                # ARIMA forecast
                forecasts = model_fit.forecast()
                val = forecasts[0]
            else:
                # SARIMA forecast
                val = model_fit.forecast()
            val = val[0]
            rets.append(val)
            errors.append(test[i] - val)
            tseries.append(test[i])
    return np.array(rets, dtype=float), np.array(errors, dtype=float)


def plot_lag_difference(args):
    dataset = args.dataset
    data = get_univariate_timeseries_data(dataset)
    tseries = np.array(data.iloc[:, 0], dtype=float)
    diffs = time_lag_diff(tseries)
    if args.plot:
        pdfpath = "temp/timeseries/timeseries_lag_diff_%s.pdf" % dataset
        dp = DataPlotter(pdfpath=pdfpath, rows=3, cols=1)

        pl = dp.get_next_plot()
        plt.title("Time train_series %s" % dataset, fontsize=8)
        pl.plot(np.arange(0, len(diffs)), diffs, 'b-')
        dp.close()


def forecast_and_report_anomalies(args):

    if not args.dataset in univariate_timeseries_datasets:
        print ("Invalid dataset: %s. Supported datasets: %s" %
              (args.dataset, str(univariate_timeseries_datasets.keys())))
        return

    datasets = [args.dataset]
    n_anoms = 10

    for dataset in datasets:
        logger.debug("dataset: %s" % dataset)
        data = get_univariate_timeseries_data(dataset)
        tseries = np.array(data.iloc[:, 0], dtype=float)
        logger.debug("timeseries[%d]:\n%s" % (len(tseries), str(list(tseries))))

        if args.log_transform:
            # log-transform now since the values are positive (in context of
            # many real-world datasets line airline); otherwise, values become
            # negative after de-trending
            tseries = log_transform_series(tseries, eps=1.0)
            logger.debug("log transformed timeseries[%d]:\n%s" % (len(tseries), str(list(tseries))))

        tseries_diff = time_lag_diff(tseries)

        acf = stattools.acf(tseries_diff, nlags=40)
        logger.debug("acf:\n%s" % str(list(acf)))

        pacf = stattools.pacf(tseries_diff, nlags=40)
        logger.debug("pacf:\n%s" % str(list(pacf)))

        residuals = train_sz = forecasts = err_ordered = None
        if not args.explore_only:
            dataset_def = univariate_timeseries_datasets[dataset]
            order = dataset_def.ARIMA_order

            model_fit, residuals = fit_ARIMA(tseries, order=order)
            logger.debug("residuals[%d]:\n%s" % (len(residuals), str(list(residuals))))

            train_sz = int(len(tseries) * 0.66)
            train = tseries[0:train_sz]
            test = tseries[train_sz:len(tseries)]
            forecasts, errors = rolling_forecast_ARIMA(train, test, order=order, nsteps=len(tseries) - train_sz)
            logger.debug("forecasts[%d]:\n%s" % (len(forecasts), str(list(forecasts))))
            logger.debug("errors[%d]:\n%s" % (len(errors), str(list(errors))))

            # identify the times at which we had largest forecasting errors
            err_ordered = np.argsort(-np.abs(errors))[0:n_anoms]
            logger.debug("largest errors[%d]:\n%s" % (len(err_ordered), str(list(errors[err_ordered]))))

        if args.plot:
            pdfpath = "temp/timeseries/timeseries_plot_%s%s.pdf" % ("log_" if args.log_transform else "", dataset)
            dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=1)

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)

            pl = dp.get_next_plot()
            plt.title("Time train_series %s%s" % ("log " if args.log_transform else "", dataset), fontsize=10)
            pl.plot(np.arange(0, len(tseries)), tseries, 'b-')

            pl = dp.get_next_plot()
            plt.title("Difference(1) %s%s" % ("log " if args.log_transform else "", dataset), fontsize=10)
            pl.plot(np.arange(0, len(tseries_diff)), tseries_diff, 'b-')

            pl = dp.get_next_plot()
            plt.title("ACF %s%s" % ("log " if args.log_transform else "", dataset), fontsize=10)
            pl.plot(np.arange(0, len(acf)), acf, 'b-')

            pl = dp.get_next_plot()
            plt.title("PACF %s%s" % ("log " if args.log_transform else "", dataset), fontsize=10)
            pl.plot(np.arange(0, len(pacf)), pacf, 'b-')

            if not args.explore_only:
                pl = dp.get_next_plot()
                plt.title("Residuals %s%s" % ("log " if args.log_transform else "", dataset), fontsize=10)
                pl.plot(np.arange(0, len(residuals)), residuals, 'b-')

                pl = dp.get_next_plot()
                plt.title("Forecast %s%s" % ("log " if args.log_transform else "", dataset), fontsize=10)
                pl.plot(np.arange(0, len(tseries)), tseries, 'b-')
                pl.plot(np.arange(train_sz, train_sz + len(forecasts)), forecasts, 'r-')
                # mark anomalous time points
                for x in err_ordered:
                    plt.axvline(x + train_sz, color='g')

            dp.close()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/timeseries")  # for logging and plots

    args = get_command_args(debug=False,
                            debug_args=["--dataset=airline", "--debug", "--plot", "--explore_only",
                                        "--log_file=temp/timeseries/timeseries_arima.log"])
    configure_logger(args)

    np.random.seed(42)

    forecast_and_report_anomalies(args)
    # plot_lag_difference(args)
