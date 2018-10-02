import numpy.random as rnd
import tensorflow as tf
from aad.aad_globals import *
from loda.loda import Loda
from .afss import AFSS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def set_random_seeds(py_seed=42, np_seed=42, tf_seed=42):
    random.seed(py_seed)
    rnd.seed(np_seed)
    tf.set_random_seed(tf_seed)


def to_2D_matrix(x):
    if x is not None and len(x.shape) == 1:
        return x.reshape((1, -1))
    return None


def set_results_dir(opts):
    if opts.results_dir is None or opts.results_dir == "":
        if opts.dataset is None or opts.dataset == "":
            opts.results_dir = "./temp/glad"
        else:
            opts.results_dir = "./temp/glad/%s/%s" % (opts.dataset, opts.get_alad_metrics_name_prefix())


class AnomalyEnsemble(object):
    def __init__(self, ensemble_type="ensemble"):
        self.ensemble_type = ensemble_type
        self.model = None
        self.m = 0

    def get_ensemble_type(self):
        return self.ensemble_type

    def is_projection_based(self):
        if isinstance(self.model, Loda):
            return True
        return False

    def get_num_members(self):
        return self.m

    def get_scores(self, x):
        """ Return individual ensemble scores - m scores per instance

        Note: Higher means more anomalous
        """
        pass

    def decision_function(self, x):
        """ Return combined ensemble scores - one score per instance

        Note: higher means more anomalous
        """
        pass

    def get_projections(self):
        """ Returns projection vectors when the ensemble is made of random projections """
        if isinstance(self.model, Loda):
            return self.model.get_projections()
        # raise RuntimeError("get_projections() supported by LODA only")
        return None  # check with is_projection_based() before calling this API


class AnomalyEnsembleLoda(AnomalyEnsemble):
    def __init__(self, loda_model):
        AnomalyEnsemble.__init__(self, ensemble_type="loda")
        self.model = loda_model
        self.m = loda_model.m

    def get_scores(self, x):
        # return higher for more anomalous
        return -self.model.get_projection_scores(x)

    def decision_function(self, x):
        # return higher for more anomalous
        return -self.model.decision_function(x)


def get_afss_model(opts, n_output=1):

    n_hidden = opts.afss_nodes
    if n_hidden < 1:
        n_hidden = max(50, n_output * 3)
        logger.debug("Setting n_hidden nodes to %d" % n_hidden)

    afss = AFSS(n_neurons=[n_hidden, n_output], names=["hidden1", "output"],
                activations=[tf.nn.sigmoid, None], bias_prob=opts.afss_bias_prob,
                prime=not opts.afss_no_prime, c_q_tau=opts.afss_c_tau, c_x_tau=opts.afss_c_tau,
                lambda_prior=opts.afss_lambda_prior, l2_penalty=True, l2_lambda=1e-3,
                train_batch_size=opts.train_batch_size,
                max_init_epochs=opts.n_epochs, max_afss_epochs=1,
                max_labeled_reps=opts.afss_max_labeled_reps)

    return afss


class SequentialResults(object):
    def __init__(self, num_seen=None, num_seen_baseline=None,
                 queried_indexes=None, queried_indexes_baseline=None):
        self.num_seen = to_2D_matrix(num_seen)
        self.num_seen_baseline = to_2D_matrix(num_seen_baseline)
        self.queried_indexes = to_2D_matrix(queried_indexes)
        self.queried_indexes_baseline = to_2D_matrix(queried_indexes_baseline)

    def merge(self, sr):
        """ Merges the results in sr into the current results

        :param sr: SequentialResults
        :return: None
        """
        if self.num_seen is None:
            self.num_seen = sr.num_seen
        else:
            self.num_seen = np.vstack([self.num_seen, sr.num_seen])

        if self.num_seen_baseline is None:
            self.num_seen_baseline = sr.num_seen_baseline
        else:
            self.num_seen_baseline = np.vstack([self.num_seen_baseline, sr.num_seen_baseline])

        if self.queried_indexes is None:
            self.queried_indexes = sr.queried_indexes
        else:
            self.queried_indexes = np.vstack([self.queried_indexes, sr.queried_indexes])

        if self.queried_indexes_baseline is None:
            self.queried_indexes_baseline = sr.queried_indexes_baseline
        else:
            self.queried_indexes_baseline = np.vstack([self.queried_indexes_baseline, sr.queried_indexes_baseline])

    def write_to_csv(self, opts):
        prefix = opts.get_alad_metrics_name_prefix()
        num_seen_file = os.path.join(opts.results_dir, "%s-num_seen.csv" % (prefix,))
        baseline_file = os.path.join(opts.results_dir, "%s-baseline.csv" % (prefix,))
        queried_idxs_file = os.path.join(opts.results_dir, "%s-queried.csv" % (prefix,))
        queried_idxs_baseline_file = os.path.join(opts.results_dir, "%s-queried-baseline.csv" % (prefix,))

        runidx_cols = np.repeat(np.array([opts.fid, 0], dtype=int).reshape((1, 2)),
                                opts.reruns, axis=0)
        runidx_cols[0:opts.reruns, 1] = np.arange(opts.reruns, dtype=int) + 1
        if self.num_seen is not None:
            np.savetxt(num_seen_file,
                       np.hstack([runidx_cols, self.num_seen]), fmt='%d', delimiter=',')

        if self.num_seen_baseline is not None:
            np.savetxt(baseline_file,
                       np.hstack([runidx_cols, self.num_seen_baseline]), fmt='%d', delimiter=',')

        if self.queried_indexes is not None:
            np.savetxt(queried_idxs_file,
                       np.hstack([runidx_cols, self.queried_indexes]), fmt='%d', delimiter=',')

        if self.queried_indexes_baseline is not None:
            np.savetxt(queried_idxs_baseline_file,
                       np.hstack([runidx_cols, self.queried_indexes_baseline]),
                       fmt='%d', delimiter=',')

