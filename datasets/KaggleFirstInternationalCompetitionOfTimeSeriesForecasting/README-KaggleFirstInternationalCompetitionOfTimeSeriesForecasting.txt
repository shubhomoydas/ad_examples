https://www.kaggle.com/c/FirstInternationalCompetitionofTimeSeriesForecasting2/data

The International Competition of Time Series Forecasting is a forecasting competition which aims to encourage the development of new models to evaluate the accuracy of computational intelligence (CI) methods in time series forecasting with multiple frequencies. The contestants will use a unique  consistent methodology for all time series.

Forecast a dataset of 8 time series from a web bookshop data as accurately as possible, using methods from computational intelligence and applying a consistent methodology. The data consists of 8 time series with different time frequencies, that could include yearly, quarterly, monthly, weekly, daily and hourly data.

The prediction competition is open to all methods of computational intelligence, incl. feed-forward and recurrent neural networks, fuzzy predictors, evolutionary & genetic algorithms, decision & regression tress, support vector regression, hybrid approaches etc. used in all areas of forecasting, prediction & time series analysis, etc.

You will be able to find in data an example_submission.csv file.  It currently contains the last observed value for each time series as the prediction for each of the remaining values, update this value with your own predictions and submit it.


In Data folder participants will be able to see different files:

    TRAIN_CSV_ICTSF_Datasets.csv: This file contains all the known Data values so participants can train their models or systems to try to improve them and make them as accurately as possible (time_series_id,time_point,value).
    number_of_predictions.csv: This file specify the number of future unkown values that have to be forecasted for each time series (time_series_id,number_of_predictions).
    sample_entry.csv: Is an example file of how a submission should be done. As you will be able to see the forecasted values are the same known last value for each time series from the TRAIN_CSV_ICTSF_Datasets.csv file. 

To carry out a submission, participants just have to open sample_entry.csv file, change the last column for the values they obtain as prediction for each time series, go to Submission folder and see how good your system is.

Careful!!, you will be able to check the accuracy of your system only once per day. GOOD LUCK!!



 The global performance of a forecasting model is evaluated by an error measure. Historically, Mean Absolute Error (MAE) or (Root) Mean Squared Error ((R)MSE) are very popular error measures. However, Mean Square Error is too sensitive to outliers [1] and furthermore, both (R)MSE and MAE are scale-dependent measures and hence, it can be hardly used for a comparison across more time series since every single time series has a different impact on the overall results [2]. For example, it has been shown that five of the 1001 series from the M-competition dominated the RMSE ranking of the forecasting methods and the remaining 996 series had only little impact on the ranking [1].

Symmetric Mean Absolute Percentage Error (SMAPE) [3]:

 

 

where et = yt-Ft for t = T +1, ..., T +h, it belongs to scale independent error measures, thus can be more easily used to compare methods across different time series. Although the SMAPE was originally proposed in [4] in a different form, formula (10) adopts the variant used in [5] since it does not lead to negative values (ranging from 0% to 200%).

For the sake of correct evaluation, we choose SMAPE to measure the errors of the competition.

 

Selected references

    J.S. Armstrong, F. Collopy, Error measures for generalizing about forecasting methods: Empirical comparisons, International Journal of Forecasting 8 (1992) 69-80.

    J.S. Armstrong, Evaluating methods, in: J.S. Armstrong (Eds.), Principles of Forecasting: A handbook for reasearchers and practitioners, Chap. 14, Kluwer, Boston/Dordrecht/London, 2001, pp. 443-473.

    R. J. Hyndman, A. Koehler, Another look at measures of forecast accuracy, International Journal of Forecasting 22(4) (2006) 679-688.

    J.S. Armstrong, Long-range forecasting, Wiley New York ETC., 1985.

    R. Andrawis, A. Atiya, A new Bayesian formulation for Holt's exponential smoothing, Journal of Forecasting 28(3) (2009) 218-234.

