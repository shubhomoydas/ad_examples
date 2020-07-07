import os
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import logging

from ..common.utils import normalize, get_command_args, configure_logger, dir_create
from ..common.gen_samples import (
    get_sphere_samples, interpolate_2D_line_by_point_and_vec, interpolate_2D_line_by_slope_and_intercept,
    MVNParams, generate_dependent_normal_samples
)
from ..common.data_plotter import DataPlotter


"""
pythonw -m ad_examples.aad.plot_anomalies_rectangle
"""


def get_x_tau(x, w, tau):
    v = x.dot(w)
    ranked = np.argsort(-v)
    tau_id = ranked[int(tau * len(v))]
    return tau_id, x[tau_id]


def plot_anomalies_ifor(outdir, plot=False, plot_legends=False):
    u_theta = np.pi * 4. / 4 + np.pi * 5 / 180
    x, y = get_sphere_samples([(50, 0, np.pi * 4. / 4, np.pi * 4. / 4 + np.pi * 2 / 4),
                               (15, 1, u_theta - np.pi * 5 / 180, u_theta + np.pi * 5 / 180),
                               (15, 1, np.pi * 6. / 4 - np.pi * 1.5 / 180, np.pi * 6. / 4)])
    n, d = x.shape

    id_nomls = np.where(y == 0)[0]
    id_anoms = np.where(y == 1)[0]
    n_anoms = len(id_anoms)

    x_nomls, y_nomls = x[id_nomls, :], y[id_nomls]
    x_anoms, y_anoms = x[id_anoms, :], y[id_anoms]

    if plot:
        axis_fontsize = 16

        line_colors = ["blue", "red", "red"]
        line_types = ["--", "--", "-"]
        line_widths = [2, 2, 2]
        lines = list()
        line_labels = list()

        tau = n_anoms * 1. / n  # multiplying by a factor to move the plane lower
        w = normalize(np.ones(2))
        r = np.array([np.min(x[:, 0]), np.max(x[:, 0])])

        tau_id, x_tau = get_x_tau(x, w, tau)
        q_tau = w.dot(x_tau)

        # plot the true weight vector
        u = interpolate_2D_line_by_point_and_vec(np.array([-1., 1.]), [0., 0.],
                                                 [np.cos(u_theta + np.pi * 1 / 4), np.sin(u_theta + np.pi * 1 / 4)])
        lines.append(u)
        line_labels.append(r"True weights ${\bf u}$")

        zd = interpolate_2D_line_by_point_and_vec(np.array([-1., 1.0]), [0., 0.], w)
        lines.append(zd)
        line_labels.append(r"Uniform weights ${\bf w}_{unif}$")

        zw = interpolate_2D_line_by_slope_and_intercept(np.array([-1., 1.]), -w[0] / w[1], q_tau / w[1])
        lines.append(zw)
        line_labels.append(r"hyperplane $\perp$ ${\bf w}_{unif}$")

        pdffile = os.path.join(outdir, "anomalies_in_ifor.pdf")
        dp = DataPlotter(pdfpath=pdffile, rows=1, cols=1)
        pl = dp.get_next_plot()
        pl.set_aspect('equal')
        # plt.xlabel('x', fontsize=axis_fontsize)
        # plt.ylabel('y', fontsize=axis_fontsize)
        plt.xticks([])
        plt.yticks([])
        plt.xlim([-1.05, 1.05])
        plt.ylim([-1.05, 1.05])
        pl.scatter(x_nomls[:, 0], x_nomls[:, 1], s=45, c="blue", marker="+", label="Nominal")
        pl.scatter(x_anoms[:, 0], x_anoms[:, 1], s=45, c="red", marker="+", label="Anomaly")
        for i, line in enumerate(lines):
            color = "blue" if line_colors is None else line_colors[i]
            pl.plot(line[:, 0], line[:, 1], line_types[i], color=color, linewidth=line_widths[i],
                    label=line_labels[i] if plot_legends else None)

        plt.axhline(0, linestyle="--", color="lightgrey")
        plt.axvline(0, linestyle="--", color="lightgrey")

        if plot_legends:
            pl.legend(loc='lower right', prop={'size': 12})
        dp.close()
    return x, y


def plot_anomalies_rect(outdir, plot=False, plot_legends=False):
    x_nomls = rnd.uniform(0., 1., 500)
    x_nomls = np.reshape(x_nomls, newshape=(250, -1))

    anom_mu = (0.83, 0.95)
    u_theta = np.arctan(0.9 / 0.8)

    anom_score_dist = MVNParams(
        mu=np.array([anom_mu[0], anom_mu[1]]),
        mcorr=np.array([
            [1, -0.5],
            [0, 1.0]]),
        dvar=np.array([0.002, 0.0005])
    )
    n_anoms = 30
    x_anoms = generate_dependent_normal_samples(n_anoms,
                                                anom_score_dist.mu,
                                                anom_score_dist.mcorr,
                                                anom_score_dist.dvar)

    x = np.vstack([x_nomls, x_anoms])
    y = np.array(np.zeros(x_nomls.shape[0], dtype=int))
    y = np.append(y, np.ones(x_anoms.shape[0], dtype=int))

    if plot:
        n, d = x.shape

        # tau is computed assuming that the anomalies occupy tau-proportion
        # of the circumference
        tau = n_anoms * 1.3 / n  # multiplying by a factor to move the plane lower
        w = normalize(np.ones(2))
        r = np.array([np.min(x[:, 0]), np.max(x[:, 0])])

        line_colors = ["blue", "red", "red"]
        line_types = ["--", "--", "-"]
        line_widths = [2, 2, 2]
        lines = list()
        line_labels = list()

        tau_id, x_tau = get_x_tau(x, w, tau)
        q_tau = w.dot(x_tau)

        # plot the true weight vector
        u = interpolate_2D_line_by_point_and_vec(np.array([0., 1.]), [0., 0.],
                                                 [np.cos(u_theta), np.sin(u_theta)])
        lines.append(u)
        line_labels.append(r"True weights ${\bf u}$")

        zd = interpolate_2D_line_by_point_and_vec(np.array([0., 1.0]), [0., 0.], w)
        lines.append(zd)
        line_labels.append(r"Uniform weights ${\bf w}_{unif}$")

        zw = interpolate_2D_line_by_slope_and_intercept(np.array([0., 1.05]), -w[0] / w[1], q_tau / w[1])
        lines.append(zw)
        line_labels.append(r"hyperplane $\perp$ ${\bf w}_{unif}$")

        axis_fontsize = 16
        pdffile = os.path.join(outdir, "anomalies_in_rect.pdf")
        dp = DataPlotter(pdfpath=pdffile, rows=1, cols=1)
        pl = dp.get_next_plot()
        pl.set_aspect('equal')
        # plt.xlabel('x', fontsize=axis_fontsize)
        # plt.ylabel('y', fontsize=axis_fontsize)
        plt.xticks([])
        plt.yticks([])
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        pl.scatter(x_nomls[:, 0], x_nomls[:, 1], s=45, c="blue", marker="+", label="Nominal")
        pl.scatter(x_anoms[:, 0], x_anoms[:, 1], s=45, c="red", marker="+", label="Anomaly")
        for i, line in enumerate(lines):
            color = "blue" if line_colors is None else line_colors[i]
            pl.plot(line[:, 0], line[:, 1], line_types[i], color=color, linewidth=line_widths[i],
                    label=line_labels[i] if plot_legends else None)

        if plot_legends:
            pl.legend(loc='lower right', prop={'size': 12})
        dp.close()

    return x, y


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/plot_anomalies_rectangle.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    rnd.seed(42)

    outdir = "./temp/illustration"
    dir_create(outdir)

    # plot isolation forest score distribution illustration
    # plot_anomalies_ifor(outdir, plot=True, plot_legends=False)

    plot_anomalies_rect(outdir, plot=True, plot_legends=False)