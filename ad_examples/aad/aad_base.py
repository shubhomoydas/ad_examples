import numpy as np

from ..common.utils import (
    order, quantile, normalize, matrix, nrow, sample, append, timer, difftime, get_random_item, logger
)
from ..common.metrics import fn_auc
from ..common.sgd_optimization import sgd, sgdMomentum, sgdRMSProp, sgdAdam, sgdRMSPropNestorov
from .aad_globals import (
    ENSEMBLE_SCORE_LINEAR, INIT_ZERO, INIT_UNIF, TAU_SCORE_FIXED, TAU_SCORE_NONE,
    PRIOR_INFLUENCE_ADAPTIVE, PRIOR_INFLUENCE_FIXED,
    AAD_CONSTRAINT_TAU_INSTANCE,
    AAD_IFOREST, AAD_HSTREES, AAD_RSFOREST, AAD_MULTIVIEW_FOREST, LODA, PRECOMPUTED_SCORES,
    initialization_types
)

from .query_model import Query
from .aad_loss import aad_loss_linear, aad_loss_gradient_linear


class Ensemble(object):
    """Stores all ensemble scores"""

    def __init__(self, samples, labels=None, scores=None, weights=None,
                 agg_scores=None, ordered_anom_idxs=None, original_indexes=None,
                 auc=0.0, model=None):
        self.samples = samples
        self.labels = labels
        self.scores = scores
        self.weights = weights
        self.agg_scores = agg_scores
        self.ordered_anom_idxs = ordered_anom_idxs
        self.original_indexes = original_indexes
        self.auc = auc
        self.model = model

        if original_indexes is None:
            self.original_indexes = np.arange(samples.shape[0])

        if agg_scores is not None and ordered_anom_idxs is None:
            self.ordered_anom_idxs = order(agg_scores, decreasing=True)


class Budget(object):
    def __init__(self, topK, budget):
        self.topK = topK
        self.budget = budget


def get_budget_topK(n, opts):
    # set topK as per tau or input topK
    topK = opts.topK
    if topK <= 0:
        topK = int(np.round(opts.tau * n))  # function of total number of instances
    budget = opts.budget
    if budget <= 0:
        budget = int(np.round(opts.tau * n))
    budget = min(opts.maxbudget, budget)
    return Budget(topK=topK, budget=budget)


def estimate_qtau(samples, model, opts, lo=-1.0, hi=1.0):
    n = samples.shape[0]
    bt = get_budget_topK(n, opts)
    scores = np.zeros(0, dtype=float)
    for i in range(50):
        w = model.get_random_weights(lo=lo, hi=hi)
        s = samples.dot(w)
        scores = np.append(scores, s)
    qval = quantile(scores, (1.0 - (bt.topK * 1.0 / float(n))) * 100.0)
    qmin = np.min(scores)
    qmax = np.max(scores)
    return qval, qmin, qmax


class MetricsStructure(object):
    def __init__(self, train_aucs=None, test_aucs=None, train_precs=None, test_precs=None,
                 train_aprs=None, test_aprs=None, train_n_at_top=None, test_n_at_top=None,
                 all_weights=None, queried=None):
        self.train_aucs = train_aucs
        self.test_aucs = test_aucs
        self.train_precs = train_precs
        self.test_precs = test_precs
        self.train_aprs = train_aprs
        self.test_aprs = test_aprs
        self.train_n_at_top = train_n_at_top
        self.test_n_at_top = test_n_at_top
        self.all_weights = all_weights
        self.queried = queried
        self.test_indexes = []


def get_aad_metrics_structure(budget, opts):
    metrics = MetricsStructure(
        train_aucs=np.zeros(shape=(1, budget)),
        # for precision@k first two columns are fid,k
        train_precs=[],
        train_aprs=np.zeros(shape=(1, budget)),
        train_n_at_top=[],
        all_weights=[],
        queried=[]
    )
    for k in range(len(opts.precision_k)):
        metrics.train_precs.append(np.zeros(shape=(1, budget)))
        metrics.train_n_at_top.append(np.zeros(shape=(1, budget)))
    return metrics


EVT_BEFORE_FEEDBACK = 0
EVT_AFTER_FEEDBACK = 1


class AadEventListener(object):
    def __init__(self):
        pass

    def __call__(self, event_type, x, y, iter, queried, model, opts):
        pass


class Aad(object):
    def __init__(self, detector_type,
                 ensemble_score=ENSEMBLE_SCORE_LINEAR,
                 random_state=None, event_listener=None):
        self.detector_type = detector_type
        self.ensemble_score = ensemble_score
        self.event_listener = event_listener
        if random_state is None:
            self.random_state = np.random.RandomState(42)
        else:
            self.random_state = random_state

        # ensemble weights learned through weak-supervision
        self.w = None
        self.qval = None

        # quick lookup of the uniform weight vector.
        # IMPORTANT: Treat this as readonly once set in fit()
        self.w_unif_prior = None

    def get_num_members(self):
        """Returns the number of ensemble members"""
        if self.w is not None:
            return len(self.w)
        return None

    def get_uniform_weights(self):
        m = self.get_num_members()
        if m is None:
            raise ValueError("weights not initialized")
        w = np.ones(m, dtype=float)
        return normalize(w)

    def get_zero_weights(self, m=None):
        if m is None:
            m = self.get_num_members()
            if m is None:
                raise ValueError("weights not initialized")
        return np.zeros(m, dtype=float)

    def get_random_weights(self, m=None, samples=None, lo=-1.0, hi=1.0):
        if samples is not None:
            w_rnd = np.ravel(get_random_item(samples, self.random_state).todense())
        else:
            if m is None:
                m = self.get_num_members()
                if m is None:
                    raise ValueError("weights not initialized")
            w_rnd = self.random_state.uniform(lo, hi, m)
        w_rnd = normalize(w_rnd)
        return w_rnd

    def init_weights(self, init_type=INIT_UNIF, samples=None):
        logger.debug("Initializing weights to %s" % initialization_types[init_type])
        if init_type == INIT_UNIF:
            self.w = self.get_uniform_weights()
        elif init_type == INIT_ZERO:
            self.w = self.get_zero_weights()
        else:
            self.w = self.get_random_weights(samples=samples)

    def get_score(self, x, w=None):
        if w is None:
            w = self.w
        if w is None:
            raise ValueError("weights not initialized")
        score = x.dot(w)
        return score

    def get_auc(self, scores, labels):
        n = len(scores)
        tmp = np.empty(shape=(n, 2), dtype=float)
        tmp[:, 0] = labels
        tmp[:, 1] = -scores
        auc = fn_auc(tmp)
        return auc

    def supports_streaming(self):
        return False

    def get_tau_ranked_instance(self, x, w, tau_rank):
        s = self.get_score(x, w)
        ps = order(s, decreasing=True)[tau_rank]
        return matrix(x[ps, :], nrow=1)

    def get_top_quantile(self, x, w, topK):
        # IMPORTANT: qval will be computed using the linear dot product
        # s = self.get_score(x, w)
        s = x.dot(w)
        return quantile(s, (1.0 - (topK * 1.0 / float(nrow(x)))) * 100.0)

    def order_by_score(self, x, w=None):
        anom_score = self.get_score(x, w)
        return order(anom_score, decreasing=True), anom_score

    def transform_to_ensemble_features(self, x, dense=False, norm_unit=False):
        """Should compute the scores from each ensemble member for each instance in x"""
        raise NotImplementedError("Need to implement this method in subclass")

    def get_truncated_constraint_set(self, w, x, y, hf,
                                     max_anomalies_in_constraint_set=1000,
                                     max_nominals_in_constraint_set=1000):
        hf_tmp = np.array(hf)
        yf = y[hf_tmp]
        ha_pos = np.where(yf == 1)[0]
        hn_pos = np.where(yf == 0)[0]

        if len(ha_pos) > 0:
            ha = hf_tmp[ha_pos]
        else:
            ha = np.array([], dtype=int)

        if len(hn_pos) > 0:
            hn = hf_tmp[hn_pos]
        else:
            hn = np.array([], dtype=int)

        if len(ha) > max_anomalies_in_constraint_set or \
                len(hn) > max_nominals_in_constraint_set:
            # logger.debug("len(ha) %d, len(hn) %d; random selection subset" % (len(ha), len(hn)))
            in_set_ha = np.zeros(len(ha), dtype=int)
            in_set_hn = np.zeros(len(hn), dtype=int)
            if len(ha) > max_anomalies_in_constraint_set:
                tmp = sample(range(len(ha)), max_anomalies_in_constraint_set)
                in_set_ha[tmp] = 1
            else:
                in_set_ha[:] = 1
            if len(hn) > max_nominals_in_constraint_set:
                tmp = sample(range(len(hn)), max_nominals_in_constraint_set)
                in_set_hn[tmp] = 1
            else:
                in_set_hn[:] = 1
            hf = append(ha, hn)
            in_set = append(in_set_ha, in_set_hn)
            # logger.debug(in_set)
        else:
            in_set = np.ones(len(hf), dtype=int)

        return hf, in_set

    def aad_weight_update(self, w, x, y, hf, w_prior, opts,
                          tau_score=None, tau_rel=True, linear=True):
        n = x.shape[0]
        bt = get_budget_topK(n, opts)

        if opts.tau_score_type == TAU_SCORE_FIXED:
            self.qval = tau_score
        elif opts.tau_score_type == TAU_SCORE_NONE:
            self.qval = None
        else:
            self.qval = self.get_top_quantile(x, w, bt.topK)

        hf, in_constr_set = self.get_truncated_constraint_set(w, x, y, hf,
                                                              max_anomalies_in_constraint_set=opts.max_anomalies_in_constraint_set,
                                                              max_nominals_in_constraint_set=opts.max_nominals_in_constraint_set)

        # logger.debug("Linear: %s, sigma2: %f, with_prior: %s" %
        #              (str(linear), opts.priorsigma2, str(opts.withprior)))

        x_tau = None
        if tau_rel:
            x_tau = self.get_tau_ranked_instance(x, w, bt.topK)
            # logger.debug("x_tau:")
            # logger.debug(to_dense_mat(x_tau))

        if opts.prior_influence == PRIOR_INFLUENCE_ADAPTIVE:
            prior_influence = 1. / max(1., 0. if hf is None else len(hf))
        elif opts.prior_influence == PRIOR_INFLUENCE_FIXED:
            prior_influence = 1.
        else:
            raise ValueError("Invalid prior_influence specified: %d" % opts.prior_influence)

        def if_f(w, x, y):
            if linear:
                return aad_loss_linear(w, x, y, self.qval, in_constr_set=in_constr_set, x_tau=x_tau,
                                       Ca=opts.Ca, Cn=opts.Cn, Cx=opts.Cx,
                                       withprior=opts.withprior, w_prior=w_prior,
                                       sigma2=opts.priorsigma2, prior_influence=prior_influence)
            else:
                raise ValueError("Only linear loss supported")

        def if_g(w, x, y):
            if linear:
                return aad_loss_gradient_linear(w, x, y, self.qval, in_constr_set=in_constr_set, x_tau=x_tau,
                                                Ca=opts.Ca, Cn=opts.Cn, Cx=opts.Cx,
                                                withprior=opts.withprior, w_prior=w_prior,
                                                sigma2=opts.priorsigma2, prior_influence=prior_influence)
            else:
                raise ValueError("Only linear loss supported")
        if False:
            w_new = sgd(w, x[hf, :], y[hf], if_f, if_g,
                        learning_rate=0.001, max_epochs=1000, eps=1e-5,
                        shuffle=True, rng=self.random_state)
        elif False:
            w_new = sgdMomentum(w, x[hf, :], y[hf], if_f, if_g,
                                learning_rate=0.001, max_epochs=1000,
                                shuffle=True, rng=self.random_state)
        elif True:
            # sgdRMSProp seems to run fastest and achieve performance close to best
            # NOTE: this was an observation on ANNThyroid_1v3 and toy2 datasets
            w_new = sgdRMSProp(w, x[hf, :], y[hf], if_f, if_g,
                               learning_rate=0.001, max_epochs=1000,
                               shuffle=True, rng=self.random_state)
        elif False:
            # sgdAdam seems to get best performance while a little slower than sgdRMSProp
            # NOTE: this was an observation on ANNThyroid_1v3 and toy2 datasets
            w_new = sgdAdam(w, x[hf, :], y[hf], if_f, if_g,
                            learning_rate=0.001, max_epochs=1000,
                            shuffle=True, rng=self.random_state)
        else:
            w_new = sgdRMSPropNestorov(w, x[hf, :], y[hf], if_f, if_g,
                                       learning_rate=0.001, max_epochs=1000,
                                       shuffle=True, rng=self.random_state)
        w_len = w_new.dot(w_new)
        # logger.debug("w_len: %f" % w_len)
        if np.isnan(w_len):
            # logger.debug("w_new:\n%s" % str(list(w_new)))
            raise ArithmeticError("weight vector contains nan")
        w_new = w_new / np.sqrt(w_len)
        return w_new

    def update_weights(self, x, y, ha, hn, opts, w=None, tau_score=None):
        """Learns new weights for one feedback iteration

        Args:
            x: np.ndarray
                input data
            y: np.array(dtype=int)
                labels. Only the values at indexes in ha and hn are relevant. Rest may be np.nan.
            ha: np.array(dtype=int)
                indexes of labeled anomalies in x
            hn: indexes of labeled nominals in x
            opts: Opts
            w: np.array(dtype=float)
                current parameter values
        """

        if w is None:
            w = self.w

        w_prior = None
        if opts.withprior:
            if opts.unifprior:
                w_prior = self.w_unif_prior
            else:
                w_prior = w

        tau_rel = opts.constrainttype == AAD_CONSTRAINT_TAU_INSTANCE
        if (opts.detector_type == AAD_IFOREST or
                opts.detector_type == AAD_HSTREES or
                opts.detector_type == AAD_RSFOREST or
                opts.detector_type == AAD_MULTIVIEW_FOREST or
                opts.detector_type == LODA or
                opts.detector_type == PRECOMPUTED_SCORES):
            w_new = self.aad_weight_update(w, x, y, hf=append(ha, hn),
                                                  w_prior=w_prior, opts=opts, tau_score=tau_score, tau_rel=tau_rel,
                                                  linear=(self.ensemble_score == ENSEMBLE_SCORE_LINEAR))
        else:
            raise ValueError("Invalid weight update for ensemble detectors: %d" % opts.detector_type)
            # logger.debug("w_new:")
            # logger.debug(w_new)

        self.w = w_new

    def aad_learn_ensemble_weights_with_budget(self, ensemble, opts):

        if opts.budget == 0:
            return None

        x = ensemble.scores
        y = ensemble.labels

        n, m = x.shape
        bt = get_budget_topK(n, opts)

        metrics = get_aad_metrics_structure(opts.budget, opts)
        ha = []
        hn = []
        xis = []

        qstate = Query.get_initial_query_state(opts.qtype, opts=opts, qrank=bt.topK,
                                               a=1., b=1., budget=bt.budget)

        save_weights = (ensemble.samples is not None and ensemble.samples.shape[1] == 2) and bt.budget < 100

        if save_weights:
            metrics.all_weights = np.zeros(shape=(opts.budget, m))
        else:
            metrics.all_weights = None

        if self.w is None:
            self.init_weights(init_type=opts.init, samples=None)

        est_tau_val = None
        if opts.tau_score_type == TAU_SCORE_FIXED:
            est_tau_val, _, _ = estimate_qtau(x, self, opts, lo=0.0, hi=1.0)
            logger.debug("Using fixed estimated tau val: %f" % est_tau_val)

        i = 0
        feedback_iter = 0
        while len(xis) < bt.budget:

            starttime_iter = timer()

            metrics.queried = xis  # xis keeps growing with each feedback iteration

            order_anom_idxs, anom_score = self.order_by_score(x, self.w)

            xi_ = qstate.get_next_query(maxpos=n, ordered_indexes=order_anom_idxs,
                                        queried_items=xis,
                                        x=x, lbls=y, y=anom_score,
                                        w=self.w, hf=append(ha, hn),
                                        ensemble=ensemble,
                                        model=self,  # some custom query models might need this access
                                        remaining_budget=bt.budget - len(xis))

            if False and len(xi_) > 1:
                logger.debug("#feedback: %d" % len(xi_))

            xis.extend(xi_)

            if opts.single_inst_feedback:
                # Forget the previous feedback instances and
                # use only the current feedback for weight updates
                ha = []
                hn = []

            for xi in xi_:
                if y[xi] == 1:
                    ha.append(xi)
                else:
                    hn.append(xi)
                if save_weights:
                    # save the weights in each iteration for later analysis
                    metrics.all_weights[i, :] = self.w
                i += 1

            qstate.update_query_state()

            if not opts.do_not_update_weights:
                self.update_weights(x, y, ha=ha, hn=hn, opts=opts, tau_score=est_tau_val)

            if self.event_listener is not None:
                self.event_listener(event_type=EVT_AFTER_FEEDBACK, x=x, y=y,
                                    iter=feedback_iter, queried=xis, model=self, opts=opts)

            feedback_iter += 1

            if np.mod(i, 1) == 0:
                endtime_iter = timer()
                tdiff = difftime(endtime_iter, starttime_iter, units="secs")
                logger.debug("Completed [%s] fid %d rerun %d feedback %d in %f sec(s)" %
                             (opts.dataset, opts.fid, opts.runidx, i, tdiff))

        return metrics


