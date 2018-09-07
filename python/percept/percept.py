from common.utils import *
import numpy as np
import numpy.random as rnd
from aad.simple_aad import *
from common.gen_samples import *


"""
pythonw -m percept.percept
"""


class Oracle(object):
    def __init__(self, y):
        self.y = np.array([1 if v == 1 else -1 for v in y], dtype=int)

    def get_label(self, i):
        return self.y[i]


def plot_learning(x, y, q, queried, aad, u_theta, dp, title=None,
                  plot_true_w=True, plot_xtau=True, plot_hyperplane=True, plot_w=True,
                  plot_theta=False, plot_legends=False):
    lbl_color_map = {0: "blue", 1: "red", 2: "green"}
    line_colors = ["blue", "red", "red"]
    line_types = ["--", "--", "-"]
    line_widths = [1, 1, 1]
    xmx = 4.1 if plot_legends else 1.1
    xlim = [-1.1, xmx]
    ylim = [-1.1, 1.1]
    marker = '+'
    s = 15
    samplescol = "grey"
    linewidth = 2

    pl = dp.get_next_plot()
    pl.set_aspect('equal')
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, fontsize=8)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # print([int(v) for v in queried.keys()])
    qidxs = np.array([int(v) for v in queried.keys()], dtype=int)

    nomls = np.where(y==0)[0]
    anoms = np.where(y==1)[0]
    pl.scatter(x[nomls, 0], x[nomls, 1], marker=marker, s=s, facecolors='blue', edgecolors="blue", label="Nominal")
    pl.scatter(x[anoms, 0], x[anoms, 1], marker=marker, s=s, facecolors='red', edgecolors="red", label="Anomaly")
    # dp.plot_points(x, pl, labels=y, lbl_color_map=lbl_color_map,
    #                marker=marker, s=s, facecolors='none', defaultcol=samplescol)
    if len(qidxs) > 0:
        pl.scatter(x[qidxs, 0], x[qidxs, 1], marker='o', s=45,
                   edgecolors='brown', facecolors='none')

    if q is not None:
        x_q = x[q]
        pl.scatter(x_q[0], x_q[1], marker='o', s=45, edgecolors='black', facecolors='none')

    x_tau = None
    if plot_xtau and aad.tau_relative:
        _, x_tau = aad.get_x_tau(x)
        pl.scatter(x_tau[0], x_tau[1], marker='+', s=45, edgecolors='orange', facecolors='orange')

    w = aad.w
    logger.debug("w: %s" % str(w))

    r = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
    # logger.debug("r: %s" % str(r))

    lines = list()
    line_labels = list()
    if plot_true_w:
        # plot the true weight vector
        u = interpolate_2D_line_by_point_and_vec(np.array([0., 1.]), [0., 0.],
                                                 [np.cos(u_theta), np.sin(u_theta)])
        lines.append(u)
        line_labels.append(r"True weights ${\bf w}^*$")

    # w0*x + w1*y = (1-aad.tau)
    if aad.fixed_tau:
        q_tau = 1 - aad.tau
    elif aad.tau_relative:
        q_tau = w.dot(x_tau)
    else:
        raise ValueError("q_tau could not be determined")

    if plot_w:
        # draw the computed weighted vector passing through the center
        # Tis is perpendicular to the hyper-plane
        zd = interpolate_2D_line_by_point_and_vec(np.array([0., w[0]]), [0., 0.], w)
        lines.append(zd)
        line_labels.append(r"Uniform weights ${\bf w}_{unif}$")

    if plot_hyperplane:
        # draw the hyper-plane passing through tau-th score
        zw = interpolate_2D_line_by_slope_and_intercept(r, -w[0] / w[1], q_tau / w[1])
        lines.append(zw)
        line_labels.append(r"hyperplane $\perp$ ${\bf w}_{unif}$")

    if plot_theta:
        pl.text(0.22, 0.33, r"${\theta}$", fontsize=20)

    for i, line in enumerate(lines):
        color = "blue" if line_colors is None else line_colors[i]
        pl.plot(line[:, 0], line[:, 1], line_types[i], color=color, linewidth=line_widths[i],
                label=line_labels[i] if plot_legends else None)

    if plot_legends:
        pl.legend(loc='lower right', prop={'size': 14})


def get_param_sig(tau_relative, fixed_tau, use_prior, update_only_on_error):
    return "%s_%s_%s%s" % \
           ("taurel" if tau_relative else "norel",
            "fixedtau" if fixed_tau else "vartau",
            "prior" if use_prior else "noprior",
            "_upderr" if update_only_on_error else "")


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/percept.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    rnd.seed(42)

    u_theta = np.pi/4 + np.pi*20./180
    x, y = get_sphere_samples([(100, 0, 0, 2*np.pi),
                               ( 15, 1, u_theta - np.pi*10./180, u_theta + np.pi*10./180)])
    n = x.shape[0]
    d = x.shape[1]

    # tau is computed assuming that the anomalies occupy tau-proportion
    # of the circumference
    tau = 1 - np.cos(0.15 * np.pi)
    tau_relative = True
    fixed_tau = True

    use_prior = True
    update_only_on_error = False

    # randomize sequence for better performance of perceptron
    ids = np.arange(n)
    rnd.shuffle(ids)
    x = x[ids]
    y = y[ids]
    oracle = Oracle(y)

    queried = dict()
    if use_prior:
        prior = normalize(np.ones(d, dtype=float))
    else:
        prior = None
    aad = SimpleActive(Ca=1., Cn=1., Cx=1.,
                       tau=tau, tau_relative=tau_relative, fixed_tau=fixed_tau,
                       prior_sigma2=d * 10.)
    aad.fit(x, w0=prior, prior=prior)

    dir_create("./temp/percept")
    sig = get_param_sig(tau_relative, fixed_tau, use_prior, update_only_on_error)
    pdfpath = "./temp/percept/percept_%s.pdf" % sig

    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)

    plot_initial_only = False
    # logger.debug("Oracle: %s" % str(oracle.y))
    u = np.array([np.cos(u_theta), np.sin(u_theta)])
    if plot_initial_only:
        budget = 0
        title = None
    else:
        budget = 30
        title = r"initial (${\theta}$: %1.2f)" % (np.arccos(u.dot(aad.w)) * 180. / np.pi)
    plot_learning(x, y, None, queried, aad, u_theta, dp,
                  title=title,
                  plot_xtau=False, plot_theta=plot_initial_only, plot_legends=plot_initial_only
                  )
    for iter in range(budget):
        # active learning step
        q = aad.get_query(x, queried)
        queried[q] = oracle.get_label(q)
        # logger.debug(queried)
        # logger.debug("q: %d, label: %d" % (q, queried[q]))

        if (not update_only_on_error) or queried[q] != 1:
            if update_only_on_error:
                logger.debug("updating on error...")
            aad.update(x, queried)

        if iter % 1 == 0:
            idxs, y_ = aad.as_arrays(queried)
            anoms, noms = aad.separate_label_indexes(y_)
            plot_learning(x, y, q, queried, aad, u_theta, dp, title=r"iter %d (${\theta}$: %1.2f) [%d/%d]" %
                                                                    (iter + 1, np.arccos(u.dot(aad.w)) * 180. / np.pi,
                                                                     len(anoms), len(noms)))
            logger.debug("iter %d: anoms: %d, noms: %d" % (iter+1, len(anoms), len(noms)))

    dp.close()
