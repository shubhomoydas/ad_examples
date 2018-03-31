2016 CIF International Time Series Competition

url: http://irafm.osu.cz/cif/main.php?c=Static&page=download

Competition Data Format
Data file containing time series to be predicted is a text file having the following format:

    Each row contains a single time series data record;
    items in the row are delimited with semicolon (";");
    the first item is an ID of the time series;
    the second item determines the forecasting horizon, i.e., the number of values to be forecasted;
    the third item determines the frequency of the time series (this year "monthly" only);
    the rest of the row contains numeric data of the time series;
    the number of values in each row may differ because each time series is of different length.

Example of the competition data format:

ts1;4;yearly;26.5;38.2;5.3
ts2;12;monthly;1;2;4;5;5;6;8;9;10
...
ts72;12;daily;1;2;4;5;5;6;8;9;10


Download

Download the competition dataset (http://irafm.osu.cz/cif/cif-dataset.txt).
Download the testing dataset (uncovered values that had to be forecasted) (http://irafm.osu.cz/cif/cif-results.txt).
Download the results provided by contestants (http://irafm.osu.cz/cif/results.zip).
