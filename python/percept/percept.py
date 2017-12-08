from common.utils import *
import numpy as np
import numpy.random as rnd
from common.gen_samples import *
from common.sgd_optimization import sgdRMSProp


"""
pythonw -m percept.percept
"""


def normalize(w):
    # normalize ||w|| = 1
    w_norm = np.sqrt(w.dot(w))
    if w_norm > 0:
        w = w / w_norm
    return w


class Oracle(object):
    def __init__(self, y):
        self.y = np.array([1 if v == 1 else -1 for v in y], dtype=int)

    def get_label(self, i):
        return self.y[i]


class ActiveAnomalyDetector(object):
    def __init__(self, tau=0.05, tau_relative=True, fixed_tau=False,
                 learning_rate=0.001,
                 Ca=1.0, Cn=1.0, Cx=1.0, prior_sigma2=1.0):
        self.tau = tau
        self.tau_relative = tau_relative
        self.fixed_tau = fixed_tau
        self.learning_rate = learning_rate
        self.w = None
        self.Ca = Ca
        self.Cn = Cn
        self.Cx = Cx
        self.prior_sigma2 = prior_sigma2

        self.w_prior = None

    def get_top_not_in(self, ranked, queried):
        for i in ranked:
            if i not in queried:
                return i
        return None

    def fit(self, x, w0=None, prior=None):
        d = x.shape[1]

        if w0 is None:
            w = rnd.uniform(0.0, 1.0, d)
        else:
            w = np.copy(w0)
        self.w = normalize(w)

        if prior is None:
            self.w_prior = None
        else:
            # w_prior = np.zeros(d, dtype=float)
            # w_prior = np.ones(d, dtype=float)
            self.w_prior = normalize(prior)

        return self.w

    def get_query(self, x, queried):
        scores = x.dot(self.w)
        ranked = np.argsort(-scores)
        return self.get_top_not_in(ranked, queried)

    def argtau(self, s, tau):
        ranked = np.argsort(s)
        return ranked[int(tau * len(s))]

    def separate_label_indexes(self, y):
        anoms = list()
        noms = list()
        for i, yi in enumerate(y):
            if yi == 1:
                anoms.append(i)
            else:
                noms.append(i)
        return np.array(anoms), np.array(noms)

    def as_arrays(self, keyvals):
        keys = []
        vals = []
        for key, val in keyvals.items():
            keys.append(key)
            vals.append(val)
        return np.array(keys), np.array(vals)

    def loss(self, w, x, y, x_tau, q_tau, w_prior):
        """
        loss = (1./(2*sigma2)) * (w-w_prior).(w-w_prior)
                    + 1/n_noms  * Cn * \sum_i{ 1[yi== 1]*hinge(x[ noms].w - q_tau) }
                    + 1/n_anoms * Ca * \sum_i{ 1[yi==-1]*hinge(q_tau - x[anoms].w) }
        :param w:
        :param x:
        :param queried:
        :param x_tau:
        :param q_tau:
        :return:
        """
        anoms, noms = self.separate_label_indexes(y)
        q_xtau = None
        if self.tau_relative:
            q_xtau = x_tau.dot(w)
        loss = 0.
        if len(anoms) > 0:
            margin_dist = x[anoms].dot(w)
            loss += self.Ca * np.mean(np.maximum(0., q_tau - margin_dist))
            if self.tau_relative:
                loss += self.Cx * np.mean(np.maximum(0., q_xtau - margin_dist))
        if len(noms) > 0:
            margin_dist = x[noms].dot(w)
            loss += self.Cn * np.mean(np.maximum(0., margin_dist - q_tau))
            if self.tau_relative:
                loss += self.Cx * np.mean(np.maximum(0., margin_dist - q_xtau))
        if self.w_prior is not None:
            wp = w - w_prior
            loss += (1./(2*self.prior_sigma2)) * wp.dot(wp)
        # logger.debug("loss: %f" % loss)
        return loss

    def loss_grad(self, w, x, y, x_tau, q_tau, w_prior):
        """
        loss_grad = (1/sigma2) * (w-w_prior)
                    + 1/n_noms  * Cn * \sum_i{ 1[yi== 1][hinge(x[ noms].w - q_tau) > 0] *  x }
                    + 1/n_anoms * Ca * \sum_i{ 1[yi==-1][hinge(q_tau - x[anoms].w) > 0] * -x }
        :param w:
        :param x:
        :param queried:
        :param x_tau:
        :param q_tau:
        :return:
        """
        anoms, noms = self.separate_label_indexes(y)
        # logger.debug("anoms: %d, noms: %d" % (len(anoms), len(noms)))
        q_xtau = None
        if self.tau_relative:
            q_xtau = x_tau.dot(w)
        grad_w = np.zeros(len(w), dtype=float)
        if len(anoms) > 0:
            ax = x[anoms]
            margin_dist = ax.dot(w)
            loss_idxs1 = np.where(margin_dist < q_tau)[0]
            grad_w -= self.Ca * (1./len(anoms)) * np.sum(ax[loss_idxs1], axis=0)

            if self.tau_relative:
                loss_idxs2 = np.where(margin_dist < q_xtau)[0]
                # grad_w += self.Cx * (1./len(anoms)) * \sum{ (x_tau - x[anoms]) }
                grad_w -= self.Cx * (1./len(anoms)) * np.sum(ax[loss_idxs2], axis=0)
                grad_w += self.Cx * x_tau
        if len(noms) > 0:
            nx = x[noms]
            margin_dist = nx.dot(w)
            loss_idxs1 = np.where(margin_dist > q_tau)[0]
            grad_w += self.Cn * (1./len(noms)) * np.sum(nx[loss_idxs1], axis=0)
            if self.tau_relative:
                loss_idxs2 = np.where(margin_dist > q_xtau)[0]
                # grad_w += self.Cx * (1./len(anoms)) * \sum{ (x[anoms] - x_tau) }
                grad_w += self.Cx * (1./len(noms)) * np.sum(nx[loss_idxs2], axis=0)
                grad_w -= self.Cx * x_tau
        # logger.debug("grad_w: %s" % str(grad_w))
        if self.w_prior is not None:
            wp = w - w_prior
            grad_w += (1./self.prior_sigma2) * wp
        return grad_w

    def get_x_tau(self, x):
        v = x.dot(self.w)
        tau_id = self.argtau(-v, self.tau)
        return tau_id, x[tau_id]

    def update(self, x, queried):

        x_tau = q_tau = None

        if self.tau_relative:
            v = x.dot(self.w)
            tau_id = self.argtau(-v, self.tau)
            x_tau = x[tau_id]
            q_tau = v[tau_id]
        else:
            q_tau = 1 - self.tau

        x_idxs, y_ = self.as_arrays(queried)
        x_ = x[x_idxs]

        def f(w, x, y):
            return self.loss(w, x, y, x_tau, q_tau, self.w_prior)

        def g(w, x, y):
            return self.loss_grad(w, x, y, x_tau, q_tau, self.w_prior)

        w0 = np.copy(self.w)
        w = sgdRMSProp(w0, x_, y_, f, g, learning_rate=self.learning_rate, max_epochs=15000)
        self.w = normalize(w)

        return self.w


def get_sphere_samples(sampledef):
    x = np.zeros(shape=(0, 2))
    y = np.zeros(0, dtype=int)
    for n, label, start_angle, end_angle in sampledef:
        thetas = rnd.uniform(start_angle, end_angle, n)
        samples = np.hstack([np.transpose([np.cos(thetas)]), np.transpose([np.sin(thetas)])])
        x = np.vstack([x, samples])
        y = np.append(y, np.ones(n, dtype=int)*label)
    return x, y


def plot_learning(x, y, q, queried, aad, u_theta, dp, title=None):
    lbl_color_map = {0: "blue", 1: "red", 2: "green"}
    line_colors = ["blue", "green", "red"]
    line_types = ["--", "-", "--"]
    line_widths = [1, 1, 1]
    xlim = [-1.1, 1.1]
    ylim = [-1.1, 1.1]
    marker = '+'
    s = 15
    samplescol = "grey"
    linewidth = 2

    pl = dp.get_next_plot()
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

    qidxs = np.array(queried.keys())
    dp.plot_points(x, pl, labels=y, lbl_color_map=lbl_color_map,
                   marker=marker, s=s, facecolors='none', defaultcol=samplescol)
    if len(qidxs) > 0:
        pl.scatter(x[qidxs, 0], x[qidxs, 1], marker='o', s=45,
                   edgecolors='brown', facecolors='none')

    if q is not None:
        x_q = x[q]
        pl.scatter(x_q[0], x_q[1], marker='o', s=45, edgecolors='black', facecolors='none')

    x_tau = None
    if aad.tau_relative:
        _, x_tau = aad.get_x_tau(x)
        pl.scatter(x_tau[0], x_tau[1], marker='+', s=45, edgecolors='orange', facecolors='orange')

    w = aad.w
    logger.debug("w: %s" % str(w))

    r = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
    # logger.debug("r: %s" % str(r))
    u = interpolate_2D_line_by_point_and_vec(np.array([0., 1.]), [0., 0.],
                                             [np.cos(u_theta), np.sin(u_theta)])
    # w0*x + w1*y = (1-aad.tau)
    if aad.fixed_tau:
        q_tau = 1 - aad.tau
    elif aad.tau_relative:
        q_tau = w.dot(x_tau)
    else:
        raise ValueError("q_tau could not be determined")
    zw = interpolate_2D_line_by_slope_and_intercept(r, -w[0] / w[1], q_tau / w[1])
    zd = interpolate_2D_line_by_point_and_vec(np.array([0., w[0]]), [0., 0.], w)

    lines = [u, zw, zd]
    for i, line in enumerate(lines):
        color = "blue" if line_colors is None else line_colors[i]
        pl.plot(line[:, 0], line[:, 1], line_types[i], color=color, linewidth=line_widths[i])


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

    tau = 0.15
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
    aad = ActiveAnomalyDetector(Ca=1., Cn=1., Cx=1.,
                                tau=tau, tau_relative=tau_relative, fixed_tau=fixed_tau,
                                prior_sigma2=d * 10.)
    aad.fit(x, w0=prior, prior=prior)

    dir_create("./temp/percept")
    sig = get_param_sig(tau_relative, fixed_tau, use_prior, update_only_on_error)
    pdfpath = "./temp/percept/percept_%s.pdf" % sig

    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)

    # logger.debug("Oracle: %s" % str(oracle.y))
    u = np.array([np.cos(u_theta), np.sin(u_theta)])
    plot_learning(x, y, None, queried, aad, u_theta, dp, title="initial (%1.2f)" %
                                                               (np.arccos(u.dot(aad.w)) * 180. / np.pi))
    budget = 40
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
            plot_learning(x, y, q, queried, aad, u_theta, dp, title="iter %d (%1.2f) [%d/%d]" %
                                                                    (iter + 1, np.arccos(u.dot(aad.w)) * 180. / np.pi,
                                                                     len(anoms), len(noms)))
            logger.debug("iter %d: anoms: %d, noms: %d" % (iter+1, len(anoms), len(noms)))

    dp.close()
