import os
import logging
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from ..common.utils import get_command_args, configure_logger, dir_create
from ..common.gen_samples import get_hard_samples
from ..common.data_plotter import DataPlotter


"""
pythonw -m ad_examples.aad.test_hard_data
"""


def plot_dataset(x, cls_cols, orig_labels, pl):
    plt.xlim([np.min(x[:, 0]), np.max(x[:, 0])])
    plt.ylim([np.min(x[:, 1]), np.max(x[:, 1])])
    for cls in np.unique(orig_labels):
        X = x[np.where(orig_labels == cls)[0], :]
        pl.scatter(X[:, 0], X[:, 1], c=cls_cols.get(cls, "grey"), marker='x',
                   linewidths=2.0, s=24, label="class %d (%s)" % (cls, "nominal" if cls == 0 else "anomaly"))
    pl.legend(loc='lower right', prop={'size': 4})


def np_mat_to_str(x):
    n = x.shape[0]
    s = ""
    for i in range(n):
        s += ",".join(x[i, :])
        s += os.linesep
    return s


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/test_hard_data.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    rnd.seed(42)

    x, y, ns = get_hard_samples()
    orig_labels = np.repeat(np.arange(len(ns))+1, ns)
    orig_labels[np.where(orig_labels <= 2)] = 0

    n = x.shape[0]
    idxs = np.arange(n, dtype=int)
    np.random.shuffle(idxs)

    x = x[idxs, :]
    y = y[idxs]
    orig_labels = orig_labels[idxs]

    dp = DataPlotter(pdfpath="./temp/test_hard_data.pdf", rows=2, cols=2)
    for i in range(3, len(ns)+1):
        pl = dp.get_next_plot()
        cls_cols = {0: "grey", 1: "blue", 2: "green", i: "red"}
        plot_dataset(x, cls_cols, orig_labels, pl)
    dp.close()

    dp = DataPlotter(pdfpath="./temp/test_hard_data_all.pdf", rows=1, cols=1)
    pl = dp.get_next_plot()
    cls_cols = {3: "blue", 4: "green", 5: "red", 6: "cyan", 7: "brown", 8: "orange", 9: "pink"}
    plot_dataset(x, cls_cols, orig_labels, pl)
    dp.close()

    dataset = "toy_hard"
    out_dir = os.path.join(".", "temp", dataset, "fullsamples")
    dir_create(out_dir)
    out_file_dat = os.path.join(out_dir, "%s_1.csv" % dataset)
    out_file_cls = os.path.join(out_dir, "%s_1_orig_labels.csv" % dataset)
    y = ["anomaly" if v == 1 else "nominal" for v in y]
    with open(out_file_dat, 'w') as f:
        f.write("label,x,y" + os.linesep)
        for i in range(n):
            f.write("%s,%f,%f%s" % (y[i], x[i, 0], x[i, 1], os.linesep))
    with open(out_file_cls, 'w') as f:
        f.write("ground.truth,label" + os.linesep)
        for cls in zip(y, orig_labels):
            f.write("%s,%d%s" % (cls[0], cls[1], os.linesep))
