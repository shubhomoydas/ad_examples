Beijing PM2.5 Data Data Set

source: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data


Data pre-processing
-------------------
Motivated by: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
There are NA values for pm2.5 for the first 24 hours and a few others. Pre-process as:
  - Discard the first 24 hrs rows
  - Set other NA values to 0
  - Drop the 'No' column
  - wind speed feature is one-hot encoded


From the source:
----------------
Abstract: This hourly data set contains the PM2.5 data of US Embassy in Beijing. Meanwhile, meteorological data from Beijing Capital International Airport are also included. 

Multivariate, Time-Series
Number of Instances: 43824
Attribute Characteristics: Integer, Real
Number of Attributes: 13
Associated Tasks: Regression
Missing Values? Yes

Source:
Song Xi Chen, csx '@' gsm.pku.edu.cn, Guanghua School of Management, Center for Statistical Science, Peking University.

Data Set Information:

The dataâ€™s time period is between Jan 1st, 2010 to Dec 31st, 2014. Missing data are denoted as â€œNAâ€.

Attribute Information:

No: row number
year: year of data in this row
month: month of data in this row
day: day of data in this row
hour: hour of data in this row
pm2.5: PM2.5 concentration (ug/m^3)
DEWP: Dew Point (â„ƒ)
TEMP: Temperature (â„ƒ)
PRES: Pressure (hPa)
cbwd: Combined wind direction
Iws: Cumulated wind speed (m/s)
Is: Cumulated hours of snow
Ir: Cumulated hours of rain

Relevant Papers:
Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H. and Chen, S. X. (2015). Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating. Proceedings of the Royal Society A, 471, 20150257.


Citation Request:
Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H. and Chen, S. X. (2015). Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating. Proceedings of the Royal Society A, 471, 20150257.

