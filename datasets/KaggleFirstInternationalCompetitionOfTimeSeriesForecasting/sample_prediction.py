"""
Example entry for Time Series Forecasting Competition

Author: Ben Hamner
Date:   11/18/2011

Copyright (C) 2011 Ben Hamner

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

# Read in the training data, saving the last point for each time series
f = open("TRAIN_CSV_ICTSF_Datasets.csv")
f.readline()
last_entry = {}
for line in f:
    line = line.split(",")
    time_series = int(line[0])
    time_point = int(line[1])
    value = float(line[2])
    last_entry[time_series] = (time_point,value)
f.close()

# Determine the number of predictions we need to make, and then make them
f_num = open("number_of_predictions.csv")
f_num.readline()

f_out = open("sample_entry.csv","w")
f_out.write("time_series_id,time_point,predicted_value\n")

for line in f_num:
    line = line.split(",")
    time_series = int(line[0])
    number_of_predictions = int(line[1])
    last_time_point = last_entry[time_series][0]
    for i in range(last_time_point+1,last_time_point+1+number_of_predictions):
        f_out.write("%i,%i,%d\n" % (time_series,i,last_entry[time_series][1]))

f_num.close()
f_out.close()
