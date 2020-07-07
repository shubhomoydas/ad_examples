import numpy as np
from ..common.utils import ncol, rep


def aad_loss_linear(w, xi, yi, qval, in_constr_set=None, x_tau=None,
                    Ca=1.0, Cn=1.0, Cx=1.0,
                    withprior=False, w_prior=None, sigma2=1.0, prior_influence=1.0):
    """
    Computes AAD loss:
        for square_slack:
            ( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )
        else:
            ( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )

    :param w: numpy.array
        parameter vector with both weights and slack variables
    :param xi: csr_matrix
    :param yi: numpy.array
    :param qval: float
        tau-th quantile value
    :param Ca: float
    :param Cn: float
    :param Cx: float
    :param withprior: boolean
    :param w_prior: numpy.array
    :param w_old: numpy.array
    :param sigma2: float
    :param square_slack: boolean
    :return:
    """
    s = xi.dot(w)

    loss_a = 0  # loss w.r.t w for anomalies
    loss_n = 0  # loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    tau_rel_loss = None
    if x_tau is not None:
        tau_rel_loss = x_tau.dot(w)
    for i in range(len(yi)):
        lbl = yi[i]

        if lbl == 1:
            n_anom += 1
        else:
            n_noml += 1

        if qval is not None:
            if lbl == 1 and s[i] < qval:
                loss_a += Ca * (qval - s[i])
            elif lbl == 0 and s[i] >= qval:
                loss_n += Cn * (s[i] - qval)
            else:
                # no loss
                pass

        if tau_rel_loss is not None and (in_constr_set is None or in_constr_set[i] == 1):
            # add loss relative to tau-th ranked instance
            # loss =
            #   Cx * (x_tau - xi).w  if yi = 1 and (x_tau - xi).w > 0
            #   Cx * (xi - x_tau).w  if y1 = 0 and (xi - x_tau).w > 0
            tau_val = tau_rel_loss[0]
            if lbl == 1 and s[i] < tau_val:
                loss_a += Cx * (tau_val - s[i])
            elif lbl == 0 and s[i] >= tau_val:
                loss_n += Cx * (s[i] - tau_val)
            else:
                # no loss
                pass

    loss = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml))

    if withprior and w_prior is not None:
        w_diff = w - w_prior
        loss += (1. * prior_influence / (2. * sigma2)) * (w_diff.dot(w_diff))

    return loss


def aad_loss_gradient_linear(w, xi, yi, qval, in_constr_set=None, x_tau=None,
                             Ca=1.0, Cn=1.0, Cx=1.0,
                             withprior=False, w_prior=None, sigma2=1.0, prior_influence=1.0):
    """
    Computes jacobian of AAD loss:
        for square_slack:
            jacobian( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )
        else:
            jacobian( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )
    """

    m = ncol(xi)

    grad = np.zeros(m, dtype=float)

    s = xi.dot(w)

    loss_a = rep(0, m)  # the derivative of loss w.r.t w for anomalies
    loss_n = rep(0, m)  # the derivative of loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    anom_idxs = list()
    noml_idxs = list()
    anom_tau_idxs = list()
    noml_tau_idxs = list()

    tau_score = None
    if x_tau is not None:
        tau_score = x_tau.dot(w)

    for i in range(len(yi)):
        lbl = yi[i]

        if lbl == 1:
            n_anom += 1
        else:
            n_noml += 1

        if qval is not None:
            if lbl == 1 and s[i] < qval:
                # loss_a[:] = loss_a - Ca * xi[i, :]
                anom_idxs.append(i)
            elif lbl == 0 and s[i] >= qval:
                # loss_n[:] = loss_n + Cn * xi[i, :]
                noml_idxs.append(i)
            else:
                # no loss
                pass

        # add loss-gradient relative to tau-th ranked instance
        if x_tau is not None and (in_constr_set is None or in_constr_set[i] == 1):
            # add loss-gradient relative to tau-th ranked instance
            # loss =
            #   Cx * (x_tau - xi).w  if yi = 1 and (x_tau - xi).w > 0
            #   Cx * (xi - x_tau).w  if y1 = 0 and (xi - x_tau).w > 0
            # loss_gradient =
            #   Cx * (x_tau - xi)  if yi = 1 and (x_tau - xi).w > 0
            #   Cx * (xi - x_tau)  if y1 = 0 and (xi - x_tau).w > 0
            tau_val = tau_score[0]
            if lbl == 1 and s[i] < tau_val:
                # loss_a[:] = loss_a + Cx * (x_tau - xi[i, :])
                anom_tau_idxs.append(i)
            elif lbl == 0 and s[i] >= tau_val:
                # loss_n[:] = loss_n + Cx * (xi[i, :] - x_tau)
                noml_tau_idxs.append(i)
            else:
                # no loss
                pass

    anom_idxs = np.array(anom_idxs, dtype=int)
    noml_idxs = np.array(noml_idxs, dtype=int)
    anom_tau_idxs = np.array(anom_tau_idxs, dtype=int)
    noml_tau_idxs = np.array(noml_tau_idxs, dtype=int)

    if len(anom_idxs) > 0:
        loss_a[:] = -Ca * np.sum(xi[anom_idxs], axis=0)
    if len(anom_tau_idxs) > 0:
        loss_a[:] = loss_a + Cx * (len(anom_tau_idxs) * x_tau - np.sum(xi[anom_tau_idxs], axis=0))

    if len(noml_idxs) > 0:
        loss_n[:] = Cn * np.sum(xi[noml_idxs], axis=0)
    if len(noml_tau_idxs) > 0:
        loss_n[:] = loss_n + Cx * (np.sum(xi[noml_tau_idxs], axis=0) - len(noml_tau_idxs) * x_tau)

    grad[0:m] = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml))

    if withprior and w_prior is not None:
        prior_mul = max(1., n_anom + n_noml)
        w_diff = w - w_prior
        grad[0:m] += (1. * prior_influence / sigma2) * w_diff

    return grad

