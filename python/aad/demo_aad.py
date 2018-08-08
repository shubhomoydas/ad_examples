from common.data_plotter import *
from common.gen_samples import *

from aad.aad_support import *
from aad.forest_description import *

"""
A simple no-frills demo of how to use AAD in an interactive loop.

To execute:
pythonw -m aad.demo_aad
"""

logger = logging.getLogger(__name__)


def get_debug_args(budget=30, detector_type=AAD_IFOREST):
    # return the AAD parameters what will be parsed later
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
            "--log_file=./temp/demo_aad.log",
            "--debug"]


def describe_instances(x, instance_indexes, model, opts):
    """ Generates compact descriptions for the input instances

    :param x: np.ndarray
        The instance matrix with ALL instances
    :param instance_indexes: np.array(dtype=int)
        Indexes for the instances which need to be described
    :param model: Aad
        Trained Aad model
    :param opts: AadOpts
    :return: np.array(dtype=int)
        Indexes of regions that describe the instances requested
    """
    if not is_forest_detector(opts.detector_type):
        raise ValueError("Descriptions only supported by forest-based detectors")

    # get feature ranges which will be used to compute volumes
    feature_ranges = get_sample_feature_ranges(x)

    # get top region indexes which will be candidates for description
    reg_idxs = get_regions_for_description(x, instance_indexes=instance_indexes,
                                           model=model, n_top=opts.describe_n_top)

    # get volume of each candidate region
    volumes = get_region_volumes(model, reg_idxs, feature_ranges)

    # get the smallest set of smallest regions that together cover all instances
    selected_region_idxs = get_compact_regions(x, model=model,
                                               instance_indexes=instance_indexes,
                                               region_indexes=reg_idxs,
                                               volumes=volumes, p=opts.describe_volume_p)
    desc_regions = [model.all_regions[ridx].region for ridx in selected_region_idxs]
    return selected_region_idxs, desc_regions


def detect_anomalies_and_describe(x, y, opts):
    rng = np.random.RandomState(opts.randseed)

    # prepare the AAD model
    model = get_aad_model(x, opts, rng)
    model.fit(x)
    model.init_weights(init_type=opts.init)

    # get the transformed data which will be used for actual score computations
    x_transformed = model.transform_to_ensemble_features(x, dense=False, norm_unit=opts.norm_unit)

    # populate labels as some dummy value (-1) initially
    y_labeled = np.ones(x.shape[0], dtype=int) * -1

    # at this point, w is uniform weight. Compute the number of anomalies
    # discovered within the budget without incorporating any feedback
    baseline_scores = model.get_score(x_transformed, model.w)
    baseline_queried = np.argsort(-baseline_scores)
    baseline_found = np.cumsum(y[baseline_queried[np.arange(opts.budget)]])
    print("baseline found:\n%s" % (str(list(baseline_found))))

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

        # most query strategies (including QUERY_DETERMINISIC) do not have anything
        # in update_query_state(), but it might be good to call this just in case...
        qstate.update_query_state()

    # the number of anomalies discovered within the budget while incorporating feedback
    found = np.cumsum(y[queried])
    print("AAD found:\n%s" % (str(list(found))))

    # generate compact descriptions for the detected anomalies
    if len(ha) > 0:
        ridxs, region_extents = describe_instances(x, np.array(ha), model=model, opts=opts)
        logger.debug("selected_region_idxs:\n%s" % (str(list(ridxs))))
        logger.debug("region_extents: these are of the form [{feature_index: (feature range), ...}, ...]\n%s" %
                     (str(region_extents)))


if __name__ == "__main__":

    # Prepare the aad arguments. It is easier to first create the parsed args and
    # then create the actual AadOpts from the args
    args = get_aad_command_args(debug=True, debug_args=get_debug_args())
    configure_logger(args)

    opts = AadOpts(args)
    logger.debug(opts.str_opts())

    np.random.seed(opts.randseed)

    # load synthetic (toy 2) dataset
    x, y = get_synthetic_samples(stype=2)

    # run interactive anomaly detection loop
    detect_anomalies_and_describe(x, y, opts)

