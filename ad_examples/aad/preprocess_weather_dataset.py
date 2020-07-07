import os
import logging
import numpy as np
import numpy.random as rnd

from ..common.gen_samples import configure_logger, get_command_args, read_csv

"""
We preprocess the weather dataset to construct a streaming anomaly dataset.
This type of preprocessing is used in (for example):

RS-Forest: A Rapid Density Estimator for Streaming Anomaly Detection
by Ke Wu, Kun Zhang, Wei Fan, Andrea Edwards and Philip S. Yu

pythonw -m ad_examples.aad.preprocess_weather_dataset
"""


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/preprocess_weather_dataset.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    rnd.seed(42)

    x = read_csv("../datasets/weather/weather_data/NEweather_data.csv")
    y = read_csv("../datasets/weather/weather_data/NEweather_class.csv")
    y = np.array(y.iloc[:, 0], dtype=int)

    logger.debug("x: %s, y: %s" % (str(x.shape), str(y.shape)))

    CLASS_NO_RAIN = 1
    CLASS_RAIN = 2

    # retain all 'no rain' instances
    no_rain_indexes = np.where(y == CLASS_NO_RAIN)[0]
    logger.debug("#no rain: %d" % len(no_rain_indexes))

    rain_indexes = np.where(y == CLASS_RAIN)[0]
    logger.debug("#rain: %d" % len(rain_indexes))

    # subsample 656 rain indexes such that 'rain' can be treated as 'anomaly' with 5% of total
    n_anoms = 656
    rnd.shuffle(rain_indexes)
    rain_subsample = rain_indexes[0:n_anoms]

    # sort the combined indexes so that original order is maintained
    combined = np.sort(np.append(no_rain_indexes, rain_subsample))
    logger.debug("#new combined: %d" % len(combined))

    x_new = x.iloc[combined, :]
    y_new = y[combined]

    logger.debug("x_new: %s, y_new: %s" % (str(x_new.shape), str(y_new.shape)))

    y_labels = ["nominal" if cls == CLASS_NO_RAIN else "anomaly" for cls in y_new]
    logger.debug("#y_labels: %d" % len(y_labels))

    f_data = open("./temp/weather_1.csv", 'w')
    f_data.write("label,%s" % ",".join(["x%d" % (i+1) for i in range(x_new.shape[1])]))
    f_data.write(os.linesep)

    f_labels = open("./temp/weather_1_orig_labels.csv", 'w')
    f_labels.write("ground.truth,label")
    f_labels.write(os.linesep)

    for i in range(x_new.shape[0]):
        f_data.write("%s,%s" % (y_labels[i], ",".join(["%0.1f" % x_new.iloc[i, j] for j in range(x_new.shape[1])])))
        f_data.write(os.linesep)

        f_labels.write("%s,%d" % (y_labels[i], y_new[i]))
        f_labels.write(os.linesep)

    f_data.close()
    f_labels.close()
