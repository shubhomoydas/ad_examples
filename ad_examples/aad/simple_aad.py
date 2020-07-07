import numpy as np
import numpy.random as rnd

from ..common.utils import normalize
from ..common.sgd_optimization import sgdRMSProp


class SimpleActive(object):
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


