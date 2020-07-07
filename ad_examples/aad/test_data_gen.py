import logging
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from ..common.utils import get_command_args, configure_logger
from ..common.data_plotter import DataPlotter
from ..common.gen_samples import read_anomaly_dataset

"""
pythonw -m ad_examples.aad.test_data_gen
"""


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/test_data_gen.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    rnd.seed(42)

    x, y = read_anomaly_dataset("toy2")

    # logger.debug("\n%s" % str(x))

    orig_labels = np.zeros(len(y), dtype=int)  # nominal classes labeled 1
    anoms = np.where(y == 1)[0]
    # logger.debug(anoms)
    anoms1 = anoms[np.where(x[anoms, 0] < 2.)[0]]
    anoms2 = anoms[np.where(x[anoms, 0] >= 2.)[0]]
    # logger.debug(anoms1)
    # logger.debug(anoms2)
    orig_labels[anoms1] = 1  # class label 1
    orig_labels[anoms2] = 2  # class label 2

    # logger.debug("\n%s" % str(np.transpose([orig_labels])))
    if True:
        cls_cols = {0: "grey", 1: "blue", 2: "red"}
        dp = DataPlotter(pdfpath="./temp/test_data_gen.pdf", rows=1, cols=1)
        pl = dp.get_next_plot()
        plt.xlim([np.min(x[:, 0]), np.max(x[:, 0])])
        plt.ylim([np.min(x[:, 1]), np.max(x[:, 1])])
        for cls in np.unique(orig_labels):
            X = x[np.where(orig_labels == cls)[0], :]
            pl.scatter(X[:, 0], X[:, 1], c=cls_cols[cls], marker='x',
                       linewidths=2.0, s=24, label="class %d (%s)" % (cls, "nominal" if cls == 0 else "anomaly"))
        pl.legend(loc='lower right', prop={'size': 8})
        dp.close()

    if False:
        f = open("./temp/toy2_1_orig_labels.csv", 'w')
        f.write("ground.truth,label")
        f.write(os.linesep)
        for i in range(x.shape[0]):
            file_data = "%s,%d" % ("nominal" if orig_labels[i] == 0 else "anomaly", orig_labels[i])
            f.write(file_data)
            f.write(os.linesep)
        f.close()
