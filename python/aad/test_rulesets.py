from common.gen_samples import *
from common.expressions import *
from .aad_globals import *
from aad.forest_description import get_regions_for_description, is_forest_detector
from .demo_aad import detect_anomalies_and_describe
from bayesian_ruleset.bayesian_ruleset import BayesianRuleset, get_max_len_in_rules


"""
pythonw -m aad.test_rulesets
"""


def get_debug_args(budget=30, detector_type=AAD_IFOREST):
    # return the AAD parameters what will be parsed later
    return ["--resultsdir=./temp", "--dataset=toy2", "--randseed=42",
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
            "--log_file=./temp/test_rulesets.log",
            "--debug"]


def get_top_regions(x, instance_indexes=None, model=None, opts=None):
    """ Gets the regions having highest anomaly scores

    :param x: np.ndarray
        The instance matrix with ALL instances
    :param instance_indexes: np.array(dtype=int)
        Indexes for the instances which need to be described
    :param model: Aad
        Trained Aad model
    :param opts: AadOpts
    :return: tuple, list(map)
        tuple: (region indexes, #instances among instance_indexes that fall in the region)
        list(map): list of region extents where each region extent is a
            map {feature index: feature range}
    """
    if not is_forest_detector(opts.detector_type):
        raise ValueError("Descriptions only supported by forest-based detectors")

    # get top region indexes which will be candidates for rule sets
    reg_idxs = get_regions_for_description(x, instance_indexes=instance_indexes,
                                           model=model, n_top=opts.describe_n_top)
    desc_regions = [model.all_regions[ridx].region for ridx in reg_idxs]
    return reg_idxs, desc_regions


def test_rulesets(x, y, meta, region_extents=None, str_rules=None, opts=None):
    if region_extents is not None:
        rules, str_rules = convert_feature_ranges_to_rules(region_extents, meta)
    else:
        rules = convert_strings_to_conjunctive_rules(str_rules, meta)
    for i, rule in enumerate(rules):
        idxs = rule.where_satisfied(x, y)
        rule.set_confusion_matrix(idxs, y)
        n_anom = np.sum(y[idxs])
        logger.debug("Rule %d: %d/%d; %s" % (i, n_anom, len(idxs), str(rule)))
    return rules, str_rules


def plot_selected_regions(x, y, regions, query_instances=None,
                          title=None, dp=None):
    pl = dp.get_next_plot()
    # pl.set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')

    if title is not None: plt.title(title, fontsize=8)

    pl.scatter(x[y == 0, 0], x[y == 0, 1], marker='x', s=12,
               facecolors='grey', edgecolors="grey", label="Nominal", linewidths=0.8)
    pl.scatter(x[y == 1, 0], x[y == 1, 1], marker='x', s=26,
               facecolors='red', edgecolors="red", label="Anomaly", linewidths=1.5)

    if query_instances is not None:
        pl.scatter(x[query_instances, 0], x[query_instances, 1], marker='o', s=60,
                   facecolors='none', edgecolors="green", label="Queried", linewidths=1.0)

    axis_lims = (plt.xlim(), plt.ylim())
    for region in regions:
        plot_rect_region(pl, region, "red", axis_lims)

    pl.legend(loc='lower right', prop={'size': 6})


def plot_rule_annotations(str_compact_rules, str_bayesian_rules, dp):
    pl = dp.get_next_plot()
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, 1])
    plt.ylim([-2, 2])
    pl.text(0.01, 1, r"${\bf Compact Descriptions:}$" + "\n  " + "\n  ".join(str_compact_rules),
            fontsize=5, color="black")
    pl.text(0.01, -1, r"${\bf Bayesian Rulesets:}$" + "\n  " + "\n  ".join(str_bayesian_rules),
            fontsize=5, color="black")


def get_subset_for_rule_mining(x, y, must_include=None, frac_more=1.0, negative_label=0):
    """ Returns a subset of instances

    The output will set the labels of must_include instances to
    the original labels. Labels on other sampled instances will
    be assumed to be the negative class.

    The returned instances will be input to a fully supervised;
    therefore, we are assigning the negative class to unlabeled
    instances. We assume that labeling instances as negative
    (i.e., nominal) is reasonable for anomaly detection.

    :param x:
    :param y:
    :param must_include:
    :param frac_more:
    :param negative_label:
    :return:
    """
    idxs = np.zeros(len(y), dtype=np.int32)
    idxs[must_include] = 1
    idxs_to_select = np.where(idxs == 0)[0]
    np.random.shuffle(idxs_to_select)
    n_more = min(len(idxs_to_select), int(len(must_include) * frac_more))
    more_select = idxs_to_select[:n_more]
    selected = list(must_include)
    selected.extend(more_select)

    # assume that labels of must_select instances are known,
    # while others are negative
    y_o = np.ones(len(y), dtype=y.dtype) * negative_label
    y_o[must_include] = y[must_include]

    return x[selected], y_o[selected]


def test_aad_rules(opts):

    plot = True

    x, y = read_anomaly_dataset(opts.dataset)
    y = np.asarray(y, dtype=np.int32)
    logger.debug(x.dtype.names)
    meta = get_feature_meta_default(x, y, feature_names=["x", "y"], label_name="label")
    logger.debug(meta)

    file_path_compact = os.path.join(opts.resultsdir, "%s_compact_rules.txt" % opts.dataset)
    file_path_top = os.path.join(opts.resultsdir, "%s_top_rules.txt" % opts.dataset)
    file_path_bayesian = os.path.join(opts.resultsdir, "%s_bayesian_rules.txt" % opts.dataset)
    file_path_queried = os.path.join(opts.resultsdir, "%s_queried.txt" % opts.dataset)
    load_rules_from_file = (os.path.isfile(file_path_compact)
                            and os.path.isfile(file_path_top)
                            and os.path.isfile(file_path_queried))

    if not load_rules_from_file:
        logger.debug("Rules file(s) not found ... regenerating rules")

        model, x_transformed, queried, ridxs_counts, region_extents_compact = \
            detect_anomalies_and_describe(x, y, opts)
        queried = np.array(queried, dtype=np.int32)

        ha = queried[np.where(y[queried] == 1)[0]]
        _, region_extents_top = get_top_regions(x, ha, model, opts)

        rules_compact, str_compact_rules = test_rulesets(x, y, meta=meta, region_extents=region_extents_compact, opts=opts)
        save_strings_to_file(str_compact_rules, file_path_compact)

        rules_top, str_top_rules = test_rulesets(x, y, meta=meta, region_extents=region_extents_top, opts=opts)
        save_strings_to_file(str_top_rules, file_path_top)

        save_strings_to_file([",".join([str(v) for v in queried])], file_path_queried)
    else:
        logger.debug("Rules file(s) found ... reading rules from file")

        str_compact_rules = load_strings_from_file(file_path_compact)
        str_top_rules = load_strings_from_file(file_path_top)
        str_queried = load_strings_from_file(file_path_queried)[0]

        queried = [int(v) for v in str_queried.split(",")]
        logger.debug("queried:\n%s" % str(queried))
        rules_compact, _ = test_rulesets(x, y, meta=meta, str_rules=str_compact_rules, opts=opts)
        rules_top, _ = test_rulesets(x, y, meta=meta, str_rules=str_top_rules, opts=opts)

    compact_ranges = convert_conjunctive_rules_to_feature_ranges(rules_compact, meta)
    top_ranges = convert_conjunctive_rules_to_feature_ranges(rules_top, meta)

    logger.debug("Compact ranges:\n%s" % str(compact_ranges))
    logger.debug("Top ranges:\n%s" % str(top_ranges))

    # rules = rules_compact
    rules = rules_top

    x_br, y_br = get_subset_for_rule_mining(x, y, queried, frac_more=1.0)

    br = BayesianRuleset(meta=meta, opts=None,  max_iter=200,
                         maxlen=get_max_len_in_rules(rules),
                         n_min_support_stop=int(0.1 * len(y_br)))
    br.fit(x_br, y_br, rules)

    logger.debug("predicted_rules:")
    for idx in br.predicted_rules:
        logger.debug("rule %d: %s" % (idx, str(br.rules[idx])))

    str_bayesian_rules = convert_conjunctive_rules_to_strings([br.rules[idx] for idx in br.predicted_rules])
    save_strings_to_file(str_bayesian_rules, file_path_bayesian)

    bayesian_ruleset_ranges = convert_conjunctive_rules_to_feature_ranges([br.rules[idx] for idx in br.predicted_rules], meta)
    logger.debug("Predicted ranges:\n%s" % str(bayesian_ruleset_ranges))

    if plot:
        path = os.path.join(opts.resultsdir, "%s_rulesets.pdf" % opts.dataset)
        dp = DataPlotter(pdfpath=path, rows=2, cols=2, save_tight=True)
        plot_selected_regions(x, y, regions=top_ranges, query_instances=queried,
                              title="Candidate Regions (Most Anomalous)\nAfter AAD (budget: %d)" % opts.budget, dp=dp)
        plot_selected_regions(x, y, regions=compact_ranges, query_instances=queried,
                              title="Compact Descriptions\nMinimum volume subspaces", dp=dp)
        plot_selected_regions(x, y, regions=bayesian_ruleset_ranges, query_instances=queried,
                              title="Bayesian Rulesets\nWang, Rudin, et al. (2016)", dp=dp)
        # plot the inferred rules
        plot_rule_annotations(str_compact_rules, str_bayesian_rules, dp)
        dp.close()


if __name__ == "__main__":

    # Prepare the aad arguments. It is easier to first create the parsed args and
    # then create the actual AadOpts from the args
    args = get_aad_command_args(debug=True, debug_args=get_debug_args())
    configure_logger(args)

    opts = AadOpts(args)
    logger.debug(opts.str_opts())

    random.seed(opts.randseed)
    np.random.seed(opts.randseed+1)

    test_aad_rules(opts)
