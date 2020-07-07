import numpy as np
import numpy.random as rnd
import logging

from ..common.gen_samples import get_command_args, configure_logger, read_csv

"""
We preprocess the weather dataset to construct a streaming anomaly dataset.
This type of preprocessing is used in (for example):

Incremental Learning of Concept Drift from Streaming Imbalanced Data by
Gregory Ditzler, Robi Polikar

pythonw -m ad_examples.aad.preprocess_electricity_dataset
"""


def get_start_row_in_arff(filepath):
    with open(filepath, 'r') as fp:
        i = 0
        for line in fp:
            if "@data" == line.strip():
                return i+1
            i += 1
    return -1


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/preprocess_electricity_dataset.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    rnd.seed(42)

    datapath = "../datasets/NSWElectricityPricing/elecNormNew.arff"
    skiprows = get_start_row_in_arff(datapath)
    # logger.debug("skiprows: %d" % skiprows)
    if skiprows < 0:
        raise ValueError("No data found in ARFF")

    x = read_csv(datapath, skiprows=skiprows)
    y = x.iloc[:, x.shape[1]-1]

    if False:
        # DEBUG
        y_up = np.sum([1 if v == "UP" else 0 for v in y])
        y_down = np.sum([1 if v == "DOWN" else 0 for v in y])

        logger.debug("UP: %d, DOWN: %d" % (y_up, y_down))  # UP: 19237, DOWN: 26075
        if True:
            exit(0)

    # ignore the first 'date' column and the last class column
    x = x.iloc[:, 1:(x.shape[1]-1)]
    y = np.array([1 if v == "DOWN" else 2 for v in y], dtype=int)
    logger.debug("x: %s, y: %s" % (str(x.shape), str(y.shape)))

    CLASS_DOWN = 1  # nominal
    CLASS_UP = 2  # anomaly

    # retain all 'DOWN' instances
    down_indexes = np.where(y == CLASS_DOWN)[0]
    logger.debug("#DOWN: %d" % len(down_indexes))

    up_indexes = np.where(y == CLASS_UP)[0]
    logger.debug("#UP: %d" % len(up_indexes))

    # subsample 1372 UP indexes such that 'UP' can be treated as 'anomaly' with 5% of total
    n_anoms = 1372
    rnd.shuffle(up_indexes)
    up_subsample = up_indexes[0:n_anoms]

    # sort the combined indexes so that original order is maintained
    combined = np.sort(np.append(down_indexes, up_subsample))
    logger.debug("#new combined: %d" % len(combined))

    x_new = x.iloc[combined, :]
    y_new = y[combined]

    logger.debug("x_new: %s, y_new: %s" % (str(x_new.shape), str(y_new.shape)))

    y_labels = ["nominal" if cls == CLASS_DOWN else "anomaly" for cls in y_new]
    logger.debug("#y_labels: %d" % len(y_labels))

    day_codes = np.eye(7, dtype=int)
    days = np.array(x_new.iloc[:, 0], dtype=int)
    days = days - 1
    day_tr = day_codes[days, :]
    x_tmp = np.array(x_new.iloc[:, 1:x_new.shape[1]], dtype=np.float32)
    x_new = np.hstack([day_tr, x_tmp])

    f_data = open("./temp/electricity_1.csv", 'w')
    f_data.write("label,%s" % ",".join(["x%d" % (i+1) for i in range(x_new.shape[1])]))
    f_data.write(os.linesep)

    f_labels = open("./temp/electricity_1_orig_labels.csv", 'w')
    f_labels.write("ground.truth,label")
    f_labels.write(os.linesep)

    for i in range(x_new.shape[0]):
        f_data.write("%s,%s" % (y_labels[i], ",".join([("%d" if j < 7 else "%0.6f") % x_new[i, j] for j in range(x_new.shape[1])])))
        f_data.write(os.linesep)

        f_labels.write("%s,%d" % (y_labels[i], y_new[i]))
        f_labels.write(os.linesep)

    f_data.close()
    f_labels.close()
