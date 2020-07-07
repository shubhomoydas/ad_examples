import logging
import numpy as np
import numpy.random as rnd
from scipy import stats
import matplotlib.pyplot as plt

from ..common.utils import get_command_args, configure_logger
from ..common.gen_samples import get_demo_samples, plot_sample, plot_samples_and_lines
from ..common.data_plotter import DataPlotter

"""
python -m ad_examples.ad.kde_outlier
"""


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/kde_outlier.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    oneD = True

    if oneD:
        # 1D data
        rnd.seed(42)
        x = np.append(stats.uniform.rvs(-1, 0.5, 20), stats.uniform.rvs(0.3, 2.0, 15))
        kernel = stats.gaussian_kde(x)
        scores = kernel.evaluate(x)
        xx = np.arange(x.min(), x.max(), 0.1)
        # logger.debug("xx:\n%s" % str(xx))
        t_scores = kernel.evaluate(xx)
        # logger.debug("scores:\n%s" % str(t_scores))
        px = np.hstack((np.transpose([x]), np.zeros(shape=(x.shape[0], 1))))
        tx = np.hstack((np.transpose([xx]), np.transpose([t_scores])))
        top_anoms = np.argsort(scores)[np.arange(5)]

        sxs = []
        # xx_ = np.arange(x.min(), x.max(), 0.1)
        logger.debug("kernel.factor:\n%s" % str(kernel.factor))
        for i in range(len(x)):
            # x_ = x[i]
            # k_ = stats.gaussian_kde(x_, kernel.factor)
            # ts_ = kernel.evaluate(xx_)
            ts_ = (1./(kernel.factor * len(x))) * np.exp(-0.5 * ((xx - x[i]) / kernel.factor) ** 2)
            logger.debug("ts_:\n%s" % str(list(ts_)))
            tx_ = np.hstack((np.transpose([xx]), np.transpose([ts_])))
            sxs.append(tx_)
        scols = ['blue'] * len(sxs)
        lines = [tx]
        lines.extend(sxs)
        line_colors = ['red']
        line_colors.extend(scols)
        line_widths = [2]
        line_widths.extend([1] * len(sxs))
        logger.debug(line_colors)
        plot_samples_and_lines(px,
                               lines=lines, line_colors=line_colors, line_legends=None,
                               top_anoms=top_anoms,
                               pdfpath="temp/kde_1d_outlier.pdf",
                               line_widths=line_widths, samplescol="green", marker='x', s=35)
    else:
        # sample_type = "4_"
        # sample_type = "donut_"
        sample_type = "face_"

        rnd.seed(42)

        x, y = get_demo_samples(sample_type)

        n = x.shape[0]

        xx = yy = x_grid = Z = scores = None
        if args.plot:
            # plot_synthetic_samples(x, y, pdfpath="temp/kde_%ssamples.pdf" % sample_type)

            # to plot probability contours
            xx, yy = np.meshgrid(np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 50),
                                 np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 50))
            x_grid = np.c_[xx.ravel(), yy.ravel()]

        kernel = stats.gaussian_kde(x.T)
        scores = kernel.evaluate(x.T)
        logger.debug(scores)
        top_anoms = np.argsort(scores)[np.arange(10)]

        if args.plot:
            # plot_samples_and_lines(x, lines=None, line_colors=None, line_legends=None,
            #                        top_anoms=top_anoms,
            #                        pdfpath="temp/kde_%soutlier.pdf" % sample_type)

            test_scores = kernel.evaluate(x_grid.T)
            Z = -np.reshape(test_scores, xx.shape)
            pdfpath = "temp/kde_%scontours.pdf" % sample_type
            dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
            pl = dp.get_next_plot()
            pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))
            dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
            pl.scatter(x[top_anoms, 0], x[top_anoms, 1], marker='o', s=35,
                       edgecolors='green', facecolors='none')
            dp.close()

