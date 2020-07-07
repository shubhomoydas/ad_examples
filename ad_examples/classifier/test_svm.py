import logging
import numpy as np
import numpy.random as rnd

from ..common.utils import get_command_args, configure_logger
from ..common.gen_samples import MVNParams, get_synthetic_samples
from .svm import BinaryLinearSVMClassifier, PairwiseLinearSVMClassifier


"""
pythonw -m ad_examples.classifier.test_svm
"""


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/svm.log"])
    configure_logger(args)

    logger.debug("SVM test...")

    sampledefs = list([
        MVNParams(
            mu=np.array([2., 4.]),
            mcorr=np.array([
                [1,  0.50],
                [0,  1.00]]),
            dvar=np.array([0.5, 0.5])
        ),
        MVNParams(
            mu=np.array([1., 1.]),
            mcorr=np.array([
                [1, -0.25],
                [0,  1.00]]),
            dvar=np.array([0.5, 0.5])
        ),
        MVNParams(
            mu=np.array([4., 1.]),
            mcorr=np.array([
                [1, 0.25],
                [0, 1.00]]),
            dvar=np.array([0.5, 0.5])
        )
    ])

    rnd.seed(42)

    # x, y = get_synthetic_samples([sampledefs[0], sampledefs[1]], [0, 1], [50, 70])
    x, y = get_synthetic_samples(sampledefs, [0, 1, 2], [50, 70, 60])

    M = len(np.unique(y))
    d = 2
    dataset_sig = ""
    if M == 2:
        logger.debug("Binary")
        dataset_sig = "binary"
        svm = BinaryLinearSVMClassifier(C=100)
    else:
        if False:
            logger.debug("Multi-Class")
            dataset_sig = "multiclass"
            svm = MultiClassLinearSVMClassifier(C=100, penalty_type='L1', penalize_bias=False)
        else:
            logger.debug("Pairwise Multi-Class")
            dataset_sig = "pairwise"
            svm = PairwiseLinearSVMClassifier(C=100)

    W, B = svm.fit(x, y)
    logger.debug("\nw:\n%s\nb:\n%s" % (str(W), str(B)))
    # exit(0)

    if True:
        pred_y = svm.predict(x)
        # logger.debug("pred_y:\n%s" % str(pred_y))
        errors = np.sum([1. if p[0] != p[1] else 0. for p in zip(pred_y, y)])
        logger.debug("errors (training set):%f" % errors)

    # x.w0 + y.w1 + b = 0
    # => y = -x.(w0/w1) -b/w1
    # z = interpolate_2D_line_by_slope_and_intercept(np.array([np.min(x[:, 0]), np.max(x[:, 0])]),
    #                                                -w0[0]/w0[1], -b0/w0[1])

    lines = []
    xmin = np.min(x[:, 0]); xmax = np.max(x[:, 0])
    # logger.debug("xmin: %f, xmax: %f" % (xmin, xmax))
    if M > 2:
        for i in range(M):
            w = W[:, i]
            b = B[i]
            zw = interpolate_2D_line_by_slope_and_intercept(np.array([xmin, xmax]),
                                                            -w[0] / w[1], -b / w[1])
            # logger.debug(zw)
            lines.append(zw)
    else:
        w = W
        b = B
        zw = interpolate_2D_line_by_slope_and_intercept(np.array([xmin, xmax]),
                                                        -w[0] / w[1], -b / w[1])
        # logger.debug(zw)
        lines.append(zw)

    plot_samples_and_lines(x, pdfpath="temp/svm_samples_%s.pdf" % dataset_sig,
                           labels=y, lbl_color_map={0: "blue", 1: "red", 2: "green"})

    plot_samples_and_lines(x, lines, line_colors=["blue", "red", "green"],
                           line_legends=svm.w_names,
                           top_anoms=None, pdfpath="temp/svm_boundaries_%s.pdf" % dataset_sig,
                           labels=y, lbl_color_map={0: "blue", 1: "red", 2:"green"})

