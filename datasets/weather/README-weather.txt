url: http://users.rowan.edu/~polikar/research/NSE/weather_data.zip

Main homepage for the dataset:
http://users.rowan.edu/~polikar/research/nse/

To cite this dataset:
---------------------
Incremental learning of concept drift from streaming imbalanced data,
in IEEE Transactions on Knowledge & Data Engineering, 2013, vol. 25, no. 10, pp. 2283-2301.
by G. Ditzler and R. Polikar,  

Incremental Learning of Concept Drift in Nonstationary Environments, 
by Ryan Elwell and Robi Polikar.


From original README of the dataset:
====================================

:::::::::::::::::::::::::::::::::
:: Weather Dataset ::::::::::::::
:::::::::::::::::::::::::::::::::

The National Oceanic and Atmospheric Administration (NOAA), part of 
the United States Department of Commerce (USDC), has compiled a database 
of weather measurements from over 7,000 weather stations worldwide. 
Records date back to the mid-1900ís providing a wide scope of weather 
trends. Daily measurements include a variety of features (temperature, 
pressure, wind speed, etc.) as well as a series of indicators for 
precipitation and other weather-related events. Offutt Air Force Base 
in Bellevue, Nebraska is selected for experimentation based on its 
extensive range of over 50 years (1949-1999) as well as its full feature 
set. 

For the selected feature set, missing values were synthetically generated 
beforehand using an average of the instances before and after the missing 
one. Class labels are determined based on the binary indicator(s) provided 
for each daily reading. Using rain as the class label yields the most 
balanced dataset consisting of 18,159 daily readings, 5,698 (31%) of which
are positive (ìrainî) while the remaining 12,461 (69%) are negative 
(ìno rainî).

For more information on the NOAA weather database, visit:
ftp://ftp.ncdc.noaa.gov/pub/data/gsod/readme.txt


::: Dataset Information ::::::::::::::::::::::::::::::::::::::

 Features: 8 Daily weather measurements
   - Temperature
   - Dew Point
   - Sea Level Pressure
   - Visibility
   - Average Wind Speed
   - Maximum Sustained Wind Speed
   - Maximum Temperature
   - Minimum Temperature
 Classes: 2 ("rain" or "no rain")
 Total Data Instances: 18,159
 Training instances per time step: 30
 Testing instances per time step: 30
 Time steps: 400


::: Training/Testing Procedure (Batch Learning) ::::::::::::::
 
 At each time step, read in a window of 30 training samples from 
 "NEweather_data.csv" (18,159 x 8) with class labels from 
 "NEweather_class.csv" 18,159 x 1). For testing, use the following 30 
 samples.  At the following time step,  shift both training and testing 
 windows by 30 (train on the testing data from the previous time step).  
 Repeat until data is depleted.


==========================================================
Ryan Elwell, Robi Polikar
Signal Processing & Pattern Recognition Laboratory (SPPRL)
Department of Electrical & Computer Engineering
Rowan University
===========================================