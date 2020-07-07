import logging
import numpy as np
import numpy.random as rnd

from ..common.utils import get_command_args, configure_logger, cbind
from ..common.gen_samples import (
    plot_sample, get_synthetic_samples, interpolate_2D_line_by_slope_and_intercept, plot_samples_and_lines
)
from ..common.sgd_optimization import sgdRMSProp

"""
python -m ad_examples.ad.outlier_effect
"""


def f_MAD(w, x, y):
    """Compute Minimum Absolute Deviation (MAD) loss
    loss = 1/n * \sum{ |y - x.w| }
    """
    loss = np.mean(np.abs(y - x.dot(w)))
    return loss


def g_MAD(w, x, y):
    """Compute gradient of Minimum Absolute Deviation (MAD) loss
    g = 1/n * d/dw \sum{ |y - x.w| }
      = 1/n * \sum{ sign(e) * (-x) }  where e = y - x.w
    """
    e = y - x.dot(w)
    grad = np.multiply(-x, np.transpose([np.sign(e)]))
    mean_grad = np.mean(grad, axis=0)
    # logger.debug(mean_grad.shape)
    return mean_grad


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/outlier_effect.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    rnd.seed(42)

    # samples with one outlier noise
    x1, y1 = get_synthetic_samples(stype=3)
    n = x1.shape[0]

    # remove outlier noise
    x2 = x1[np.arange(x1.shape[0]-1)]
    y2 = y1[np.arange(x1.shape[0]-1)]

    plot_sample(x1, y1, pdfpath="temp/outlier_effect_samples.pdf")

    X1 = cbind(np.ones(shape=(x1.shape[0], 1), dtype=float), x1[:, [0]])
    b1 = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(x1[:, [1]])

    X2 = cbind(np.ones(shape=(x2.shape[0], 1), dtype=float), x2[:, [0]])
    b2 = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(x2[:, [1]])

    # MAD estimation with outlier
    b3 = np.zeros(X1.shape[1], dtype=float)
    b3 = sgdRMSProp(b3, X1, x1[:, 1], f_MAD, g_MAD, learning_rate=0.001, ro=0.9, batch_size=n, max_epochs=2000)

    # MAD estimation without outlier
    b4 = np.zeros(X2.shape[1], dtype=float)
    b4 = sgdRMSProp(b4, X2, x2[:, 1], f_MAD, g_MAD, learning_rate=0.001, ro=0.9, batch_size=n, max_epochs=2000)

    logger.debug("b1:\n%s" % str(b1))
    logger.debug("b2:\n%s" % str(b2))
    logger.debug("b3:\n%s" % str(list(b3)))
    logger.debug("b4:\n%s" % str(list(b4)))

    # x = np.arange(-4, 8, 0.2)
    x = np.array([np.min(x1[:, 0]), np.max(x1[:, 0])])
    z1 = interpolate_2D_line_by_slope_and_intercept(x, b1[1], b1[0])
    z2 = interpolate_2D_line_by_slope_and_intercept(x, b2[1], b2[0])
    z3 = interpolate_2D_line_by_slope_and_intercept(x, b3[1], b3[0])
    z4 = interpolate_2D_line_by_slope_and_intercept(x, b4[1], b4[0])
    # z4 = None

    if args.plot:
        plot_samples_and_lines(x1, [z1, z2, z3, z4], ['red', 'blue', 'green', 'brown'],
                               line_legends=['L2 w outlier', 'L2 w/o outlier',
                                             'L1 w outlier', 'L1 w/o outlier'],
                               top_anoms=np.array([x1.shape[0]-1]),
                               pdfpath="temp/outlier_effect.pdf")
