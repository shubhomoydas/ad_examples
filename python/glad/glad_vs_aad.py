from .glad_batch import *
from aad.aad_base import *


def get_precomputed_aad_args(budget=30, detector_type=PRECOMPUTED_SCORES):
    # return the AAD parameters which will be parsed later
    return ["--resultsdir=./temp", "--randseed=42",
            "--reruns=1",
            "--detector_type=%d" % detector_type,
            "--forest_score_type=%d" %
            (IFOR_SCORE_TYPE_NEG_PATH_LEN if detector_type == AAD_IFOREST
             else HST_LOG_SCORE_TYPE if detector_type == AAD_HSTREES
             else RSF_SCORE_TYPE if detector_type == AAD_RSFOREST else 0),
            "--init=%d" % INIT_UNIF,  # initial weights
            "--withprior", "--unifprior",  # use an (adaptive) uniform prior
            # ensure that scores of labeled anomalies are higher than tau-ranked instance,
            # while scores of nominals are lower
            "--constrainttype=%d" % AAD_CONSTRAINT_TAU_INSTANCE,
            "--querytype=%d" % QUERY_DETERMINISIC,  # query strategy
            "--num_query_batch=1",  # number of queries per iteration
            "--budget=%d" % budget,  # total number of queries
            "--tau=0.03",
            # normalize is NOT required in general.
            # Especially, NEVER normalize if detector_type is anything other than AAD_IFOREST
            # "--norm_unit",
            "--forest_n_trees=100", "--forest_n_samples=256",
            "--forest_max_depth=%d" % (100 if detector_type == AAD_IFOREST else 7),
            # leaf-only is preferable, else computationally and memory expensive
            "--forest_add_leaf_nodes_only",
            "--ensemble_score=%d" % ENSEMBLE_SCORE_LINEAR,
            "--resultsdir=./temp",
            "--log_file=./temp/afss_vs_aad.log",
            "--debug"]


def populate_aad_opts(afss_opts, aad_opts):
    aad_opts.budget = afss_opts.budget
    aad_opts.tau = afss_opts.afss_tau
    aad_opts.reruns = afss_opts.reruns


class AadWithExisingEnsemble(Aad):
    def __init__(self, ensemble, opts=None, random_state=None):
        Aad.__init__(self, PRECOMPUTED_SCORES, random_state=random_state)
        self.ensemble = ensemble
        self.opts = opts
        self.m = ensemble.get_num_members()
        w = np.ones(self.m, dtype=float)
        self.w = normalize(w)

    def fit(self, x):
        # not required
        pass

    def get_num_members(self):
        """Returns the number of ensemble members"""
        return self.m

    def transform_to_ensemble_features(self, x, dense=True, norm_unit=False):
        return self.ensemble.get_scores(x)


def aad_active_learn_ensemble(x, y, ensemble, opts):
    model = AadWithExisingEnsemble(ensemble, opts,
                                   random_state=np.random.RandomState(opts.randseed))
    model.init_weights(init_type=opts.init)

    # get the transformed data which will be used for actual score computations
    x_transformed = model.transform_to_ensemble_features(x, dense=True, norm_unit=False)

    # populate labels as some dummy value (-1) initially
    y_labeled = np.ones(x.shape[0], dtype=int) * -1

    baseline_scores = model.get_score(x_transformed, model.w)
    baseline_queried = np.argsort(-baseline_scores)
    baseline_found = np.cumsum(y[baseline_queried[np.arange(opts.budget)]])
    logger.debug("AAD baseline found:\n%s" % (str(list(baseline_found))))

    qstate = Query.get_initial_query_state(opts.qtype, opts=opts, budget=opts.budget)
    queried = []  # labeled instances
    ha = []  # labeled anomaly instances
    hn = []  # labeled nominal instances
    while len(queried) < opts.budget:
        ordered_idxs, anom_score = model.order_by_score(x_transformed)
        qx = qstate.get_next_query(ordered_indexes=ordered_idxs,
                                   queried_items=queried)
        queried.extend(qx)
        for xi in qx:
            y_labeled[xi] = y[xi]  # populate the known labels
            if y[xi] == 1:
                ha.append(xi)
            else:
                hn.append(xi)

        # incorporate feedback and adjust ensemble weights
        model.update_weights(x_transformed, y_labeled, ha=ha, hn=hn, opts=opts, tau_score=opts.tau)

        qstate.update_query_state()

    # the number of anomalies discovered within the budget while incorporating feedback
    found = np.cumsum(y[queried])
    logger.debug("AAD found:\n%s" % (str(list(found))))

    # make queried indexes 1-indexed
    queried = np.array(queried, dtype=int) + 1
    baseline_queried = np.array(baseline_queried[0:opts.budget], dtype=int) + 1

    results = SequentialResults(num_seen=found, num_seen_baseline=baseline_found,
                                queried_indexes=queried,
                                queried_indexes_baseline=baseline_queried)
    return results


def aad_active_learn(x, y, ensembles, aad_opts, glad_opts):

    logger.debug("dataset: %s, shape: %s" % (aad_opts.dataset, str(x.shape)))

    all_aad_results = SequentialResults()
    orig_randseed = aad_opts.randseed
    for i, ensemble in enumerate(ensembles):
        tm = Timer()

        aad_opts.randseed = orig_randseed + i
        aad_opts.runidx = i + 1
        set_random_seeds(aad_opts.randseed, aad_opts.randseed + 1, aad_opts.randseed + 2)

        logger.debug("# ensemble members: %d" % ensemble.m)

        aad_results = aad_active_learn_ensemble(x, y, ensemble, aad_opts)
        all_aad_results.merge(aad_results)

        logger.debug(tm.message("completed AAD run %d/%d:" % (i + 1, aad_opts.reruns)))

    tmp = glad_opts.ensemble_type
    # a hack to set the results filename prefix appropriately for AAD
    glad_opts.ensemble_type = "aad"
    all_aad_results.write_to_csv(glad_opts)
    glad_opts.ensemble_type = tmp

    return x, y, all_aad_results, ensembles


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    dir_create("./temp/glad")

    glad_args = get_glad_command_args(debug=False, debug_args=None)
    # print "log file: %s" % args.log_file
    configure_logger(glad_args)

    glad_opts = GladOpts(glad_args)
    logger.debug("GLAD: %s" % glad_opts.str_opts())

    # Prepare the aad arguments. It is easier to first create the parsed args and
    # then create the actual AadOpts from the args
    aad_args = get_aad_command_args(debug=True, debug_args=get_precomputed_aad_args())

    aad_opts = AadOpts(aad_args)
    populate_aad_opts(glad_opts, aad_opts)
    logger.debug("AAD: %s" % aad_opts.str_opts())

    x, y, results, ensembles = glad_active_learn(glad_opts)

    if glad_opts.compare_aad:
        _, _, all_aad_results, _ = aad_active_learn(x, y, ensembles, aad_opts, glad_opts)
