import logging
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from ..common.utils import get_command_args, configure_logger
from ..common.gen_samples import get_demo_samples, plot_sample
from ..common.data_plotter import DataPlotter
from ..loda.loda import Loda

"""
python -m ad_examples.ad.ad_outlier --plot --debug --log_file=temp/ad_outlier.log --dataset=face --algo=ifor

Supported algorithms: ifor, loda, lof, ocsvm

Supported synthetic datasets:
    face
    face_diff
    donut
    donut_diff
    1
    4
"""


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=False, debug_args=["--algo=ifor",
                                                     "--dataset=face",
                                                     "--debug",
                                                     "--plot",
                                                     "--log_file=temp/ad_outlier.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    ad_type = args.algo  # ocsvm, ifor, lof, loda
    # ad_type = "ifor"
    # ad_type = "lof"

    sample_type = args.dataset + "_"
    # sample_type = "4_"
    # sample_type = "donut_"
    # sample_type = "donut_diff_"
    # sample_type = "face_"
    # sample_type = "face_diff_"

    rnd.seed(42)

    x, y = get_demo_samples(sample_type)

    n = x.shape[0]

    outliers_fraction = 0.1

    xx = yy = x_grid = Z = scores = None
    if args.plot:
        plot_sample(x, y, pdfpath="temp/ad_%ssamples.pdf" % (sample_type))

        # to plot probability contours
        xx, yy = np.meshgrid(np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 50),
                             np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 50))
        x_grid = np.c_[xx.ravel(), yy.ravel()]

    if ad_type == "ocsvm":
        ad = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)
        ad.fit(x)
        scores = -ad.decision_function(x).reshape((n,))
        Z = -ad.decision_function(x_grid)
    elif ad_type == "ifor":
        ad = IsolationForest(max_samples=256, contamination=outliers_fraction, random_state=None)
        ad.fit(x)
        scores = -ad.decision_function(x)
        Z = -ad.decision_function(x_grid)
    elif ad_type == "lof":
        ad = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
        ad.fit(x)
        scores = -ad._decision_function(x)
        Z = -ad._decision_function(x_grid)
    elif ad_type == "loda":
        ad = Loda(mink=100, maxk=200)
        ad.fit(x)
        scores = -ad.decision_function(x)
        Z = -ad.decision_function(x_grid)

    logger.debug("scores:\n%s" % str(list(scores)))
    top_anoms = np.argsort(-scores)[np.arange(10)]

    if args.plot:
        # plot_samples_and_lines(x, lines=None, line_colors=None, line_legends=None,
        #                        top_anoms=top_anoms, pdfpath="temp/%s_%soutlier.pdf" % (ad_type, sample_type))
        Z = Z.reshape(xx.shape)
        pdfpath = "temp/ad_%scontours_%s.pdf" % (sample_type, ad_type)
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()
        pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))
        dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        pl.scatter(x[top_anoms, 0], x[top_anoms, 1], marker='o', s=35,
                   edgecolors='red', facecolors='none')
        dp.close()

