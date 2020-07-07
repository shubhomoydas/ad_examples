import os
import random
import numpy as np
import numpy.random as rnd
import tensorflow as tf
from ..common.utils import logger
from ..common.expressions import get_feature_meta_default
from ..aad.classifier_trees import DecisionTreeAadWrapper
from ..loda.loda import Loda
from .afss import AFSS
from ..aad.forest_description import InstancesDescriber
from ..loda.loda import ProjectionVectorsHistograms, LodaModel, LodaResult
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

    def get_members(self):
        """ Return a list of members which implement the decision_function(x) API """
        raise NotImplementedError


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

    def get_members(self):
        members = []
        loda_model = self.model.loda_model  # LodaResult object
        pvh = loda_model.pvh.pvh  # ProjectionVectorsHistograms object
        w = pvh.w
        hists = pvh.hists
        for i in range(self.m):
            pvh_ = ProjectionVectorsHistograms(w=w[:, [i]], hists=[hists[i]])
            loda_ = LodaModel(k=1, pvh=pvh_, sigs=None)
            loda_model = LodaResult(anomranks=None, nll=None, pvh=loda_)
            loda = Loda()
            loda.m = 1
            loda.loda_model = loda_model
            members.append(AnomalyEnsembleLoda(loda))
        return members


def get_afss_model(opts, n_output=1):

    layer_sizes = opts.afss_nodes
    if len(layer_sizes) == 0 or any(n < 1 for n in layer_sizes):
        layer_sizes = [max(50, n_output * 3)]
        logger.debug("Setting layer_sizes to [%d]" % layer_sizes[0])

    logger.debug("l2_lambda: %f" % opts.afss_l2_lambda)
    logger.debug("max_afss_epochs: %d" % opts.max_afss_epochs)

    n_neurons = layer_sizes + [n_output]

    names = ["hidden%d" % (i+1) for i in range(len(n_neurons)-1)]
    names.append("output")
    activations = []
    if len(n_neurons) > 2:
        activations = [tf.nn.leaky_relu] * (len(n_neurons) - 2)
    activations.extend([tf.nn.sigmoid, None])

    logger.debug("n_neurons (%d): %s" % (len(n_neurons), str(n_neurons)))
    logger.debug("names: %s" % str(names))

    afss = AFSS(n_neurons=n_neurons, names=names,
                activations=activations, bias_prob=opts.afss_bias_prob,
                prime=not opts.afss_no_prime, c_q_tau=opts.afss_c_tau, c_x_tau=opts.afss_c_tau,
                lambda_prior=opts.afss_lambda_prior, l2_penalty=True, l2_lambda=opts.afss_l2_lambda,
                train_batch_size=opts.train_batch_size,
                max_init_epochs=opts.n_epochs, max_afss_epochs=opts.max_afss_epochs,
                max_labeled_reps=opts.afss_max_labeled_reps)

    return afss


class GLADRelevanceDescriber(InstancesDescriber):
    """ Generates descriptions for relevance of detectors

    Employs a decision tree to generate rule-based descriptors.
    """
    def __init__(self, x, y, model, opts, max_rank=0):
        InstancesDescriber.__init__(self, x, y, model, opts, sample_negative=False)
        self.meta = get_feature_meta_default(x, y)
        self.max_rank = max_rank

    def get_member_relevance_scores_ranks(self, x):
        scores_all = self.model.decision_function(x)

        ranks_all = np.argsort(-scores_all, axis=1)
        best_member = ranks_all[:, 0]

        ranks_all = np.argsort(ranks_all, axis=1)

        return scores_all, ranks_all, best_member

    def describe(self, instance_indexes):
        """ Generates descriptions for the input instances

        :param instance_indexes: indexes of positive instances
        :return: list of (list of int, list of dict, Rules)
        """

        if instance_indexes is None:
            x = self.x
        else:
            x = self.x[instance_indexes]

        scores_all, ranks_all, best_member = self.get_member_relevance_scores_ranks(x)
        max_rank = self.max_rank
        if max_rank <= 0:
            max_rank = scores_all.shape[1]
        # logger.debug("max_rank: %d" % max_rank)

        descriptions = []
        for i in range(scores_all.shape[1]):
            scores = scores_all[:, i]
            ranks = ranks_all[:, i]
            # logger.debug("scores [%d]:\n%s" % (i, str(list(scores))))
            # logger.debug("ranks [%d]:\n%s" % (i, str(list(ranks))))

            # Instances having AFSS probabilities > bias_prob are in 'relevant'
            # regions for an ensemble member.
            labels = np.zeros(len(scores), dtype=np.int32)
            rel_indexes = np.where(scores > self.model.bias_prob)[0]
            rel_indexes = rel_indexes[ranks[rel_indexes] < max_rank]

            if len(rel_indexes) > 0:
                labels[rel_indexes] = 1
                # train a decision tree classifier that will separate 0s from 1s
                dt = DecisionTreeAadWrapper(x, labels, max_depth=None)
                # get all regions from the decision tree where probability of
                # predicting anomaly class is > 0.5
                region_idxs = np.where(dt.d > 0.5)[0]
                feature_ranges = [dt.all_regions[i].region for i in region_idxs]
                logger.debug(region_idxs)
                logger.debug(feature_ranges)
                rules, str_rules = self.convert_regions_to_rules(feature_ranges, region_indexes=region_idxs)
            else:
                logger.debug("No relevant regions found for member %d..." % i)
                region_idxs = []
                feature_ranges = []
                rules = []
            descriptions.append((region_idxs, feature_ranges, rules))

        return descriptions, best_member


class GLADEnsembleLimeExplainer(object):
    """ Generates explanations with LIME

    Explainer and Describer are two different concepts:
        - Explainer tells us why the detector considers an instance an anomaly.
        - Describer generates a compact description for one (or more) instances.

    Must install LIME. Use the following command:
        pip install lime

    References:
        "Why Should I Trust You?" Explaining the Predictions of Any Classifier
            by Marco Tulio Ribeiro, Sameer Singh and Carlos Guestrin, KDD, 2016.
            https://marcotcr.github.io/lime
    """
    def __init__(self, x, y, ensemble, afss, feature_names=None):
        self.members = ensemble.get_members()
        self.afss = afss
        self.describer = GLADRelevanceDescriber(x, y, model=afss, opts=None)
        logger.debug("#ensemble members: %d" % len(self.members))
        try:
            import lime
            from lime.lime_tabular import LimeTabularExplainer
            logger.debug("loaded LIME")
            self.explainer = LimeTabularExplainer(x, mode="regression", feature_names=feature_names,
                                                  random_state=42)
        except:
            self.explainer = None
            logger.warning("Failed to load LIME. Install LIME with command: 'pip install lime' or "
                           "see: https://marcotcr.github.io/lime")
            print("WARNING: Failed to load LIME. Install LIME with command: 'pip install lime' or "
                  "see: https://marcotcr.github.io/lime")

    def explain(self, inst, member_index=-1):
        """ Generates explanation with a single ensemble member

        First, finds the best member for the instance using AFSS relevance scores.
        Then employs LIME to generate the explanation.

        :param inst: 1d array of instance features
        :param member_index: int
            The index of the anomaly detector in the ensemble.
            If -1, the most relevant member as per AFSS will be used
        :return: LIME Explanation, best ensemble, member relevance
        """
        if self.explainer is None:
            print("WARNING: Explainer is not initialized. No explanations generated.")
            logger.warning("Explainer is not initialized. No explanations generated.")
            return None, None, None
        member_relevance = None
        if member_index < 0:
            member_relevance, ranks_all, best_member = self.describer.get_member_relevance_scores_ranks(np.reshape(inst, (1, -1)))
            member_index = best_member[0]
            # logger.debug("best member index: %d" % member_index)
        explanation = self.explainer.explain_instance(inst,
                                                      predict_fn=self.members[member_index].decision_function)
        return explanation, member_index, member_relevance


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

