import logging
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from ..common.utils import (
    get_command_args, configure_logger, LogisticRegressionClassifier,
    SVMClassifier, DTClassifier, RFClassifier
)
from ..common.gen_samples import get_demo_samples, plot_sample
from ..common.data_plotter import DataPlotter

"""
pythonw -m ad_examples.ad.pseudo_anom_outlier
"""


def transform_2D_data(x):
    """Transforms to higher polynomial features
    (x1, x2) -> (x1^2, x1, x1x2, x2, x2^2)
    (x1, x2) -> (x1^4, x1^3 x2, ..., x1, x2)
    """
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    if False:
        xx = np.hstack((x1 ** 2, x1, x1 * x2, x2, x2 ** 2))
    else:
        xx = np.hstack((x1**5, x1**4 * x2, x1**3 * x2**2, x1**2 * x2**3, x1 * x2**4, x2**5,
                        x1**4, x1**3 * x2, x1**2 * x2**2, x1 * x2**3, x2**4,
                        x1**3, x1**2 * x2, x1 * x2**2, x2**3,
                        x1**2, x1 * x2, x2**2,
                        x1, x2))
    return xx


def get_artificial_2D_data_uniform(x_range, y_range, n):
    x = np.hstack((np.transpose([rnd.uniform(x_range[0], x_range[1], n)]),
                   np.transpose([rnd.uniform(y_range[0], y_range[1], n)])))
    return x


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/pseudo_anom_outlier.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    # RF forest seems to work best on the donut dataset with uniform artificial anomalies
    # classifier_type = "DT"
    # classifier_type = "RF"
    # classifier_type = "LR"
    classifier_type = "SVM"
    transform_2D = not (classifier_type == "SVM" or
                        classifier_type == "DT" or
                        classifier_type == "RF")

    # sample_type = "4_"
    sample_type = "donut_"
    # sample_type = "face_"

    rnd.seed(42)

    x, y = get_demo_samples(sample_type)
    n = x.shape[0]

    x_artificial = get_artificial_2D_data_uniform((np.min(x[:, 0]), np.max(x[:, 0])),
                                                  (np.min(x[:, 1]), np.max(x[:, 1])), 1000)

    if transform_2D:
        x_tr = transform_2D_data(x)
        x_artificial_tr = transform_2D_data(x_artificial)
    else:
        x_tr = x.copy()
        x_artificial_tr = x_artificial.copy()
    logger.debug(x_tr.shape)

    if args.plot:
        x_cat = np.vstack((x_artificial, x))
        y_cat = np.append(np.ones(x_artificial.shape[0]), np.zeros(n))
        plot_sample(x_cat, y_cat, pdfpath="temp/pseudo_anom_outlier_cat.pdf")

    tx = np.vstack((x_tr, x_artificial_tr))
    ty = np.append(np.zeros(n), np.ones(x_artificial.shape[0]))
    logger.debug("augmented data size: %s" % str(tx.shape))

    if classifier_type == "LR":
        classifier = LogisticRegressionClassifier.fit(tx, ty, C=1000.)
    elif classifier_type == "SVM":
        classifier = SVMClassifier.fit(tx, ty, C=1000., kernel='rbf')
    elif classifier_type == "DT":
        classifier = DTClassifier.fit(tx, ty, max_depth=15)
    elif classifier_type == "RF":
        classifier = RFClassifier.fit(tx, ty, n_estimators=30, max_depth=10)
    else:
        raise ValueError("invalid classifier type %s" % classifier_type)

    probs = classifier.predict_prob_for_class(x_tr, 1)  # predict on only the actual data
    # logger.debug("classes: %s" % str(classifier.clf.classes_))
    logger.debug("predicted probs:\n%s" % str(list(probs)))
    outliers = np.where(probs > (0.65 if classifier_type == "RF" else 0.8))
    # logger.debug("#outliers: %d" % len(outliers[0]))
    # top_anoms = np.argsort(-probs)[np.arange(10)]
    top_anoms = outliers[0]

    if args.plot:

        # plot_samples_and_lines(x, lines=None, line_colors=None, line_legends=None,
        #                        top_anoms=top_anoms, pdfpath="temp/pseudo_anom_outlier_%s.pdf" % classifier_type)

        # plot probability contours
        xx, yy = np.meshgrid(np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 50),
                             np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 50))
        x_grid = np.c_[xx.ravel(), yy.ravel()]
        if transform_2D:
            x_grid_tr = transform_2D_data(x_grid)
        else:
            x_grid_tr = x_grid.copy()
        probs_grid = classifier.predict_prob_for_class(x_grid_tr, 1)
        logger.debug("predicted grid probs of size %s" % str(x_grid_tr.shape))
        Z = probs_grid.reshape(xx.shape)
        pdfpath = "temp/pseudo_anom_outlier_contours_%s.pdf" % classifier_type
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()
        pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))
        dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        pl.scatter(x[top_anoms, 0], x[top_anoms, 1], marker='o', s=35,
                   edgecolors='red', facecolors='none')
        dp.close()
