import os
import random
import numpy as np
import matplotlib.pyplot as plt

from ..common.utils import logger, configure_logger
from ..common.data_plotter import plot_rect_region, DataPlotter
from ..common.expressions import save_strings_to_file, evaluate_ruleset, load_strings_from_file, \
    convert_conjunctive_rules_to_feature_ranges, get_feature_meta_default, convert_conjunctive_rules_to_strings
from ..common.gen_samples import read_anomaly_dataset
from .aad_globals import (
    AAD_IFOREST, AAD_HSTREES, AAD_RSFOREST,
    IFOR_SCORE_TYPE_NEG_PATH_LEN, HST_LOG_SCORE_TYPE, RSF_SCORE_TYPE,
    INIT_UNIF, AAD_CONSTRAINT_TAU_INSTANCE, QUERY_DETERMINISIC, ENSEMBLE_SCORE_LINEAR,
    get_aad_command_args, AadOpts
)
from .demo_aad import detect_anomalies_and_describe
from .aad_ruleset_support import get_rulesets, prepare_conjunctive_rulesets
from .forest_description import CompactDescriber, BayesianRulesetsDescriber
from .anomaly_dataset_support import dataset_feature_names

"""
pythonw -m ad_examples.aad.test_rulesets
"""


def get_debug_args(dataset="toy2", budget=30, detector_type=AAD_IFOREST):
    # return the AAD parameters what will be parsed later
    return ["--resultsdir=./temp", "--dataset=%s" % dataset, "--randseed=42",
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
            "--debug", "--plot2D"]


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
    if str_compact_rules is not None:
        pl.text(0.01, 1, r"${\bf Compact\ Descriptions:}$ Predict 'anomaly' if:" + "\n " + "\n or ".join(str_compact_rules),
                fontsize=4, color="black")
    if str_bayesian_rules is not None:
        pl.text(0.01, -1, r"${\bf Bayesian\ Rulesets:}$ Predict 'anomaly' if:" + "\n " + "\n or ".join(str_bayesian_rules),
                fontsize=4, color="black")


def test_aad_rules(opts):

    x, y = read_anomaly_dataset(opts.dataset)
    y = np.asarray(y, dtype=np.int32)
    logger.debug(x.dtype.names)
    feature_names = dataset_feature_names.get(opts.dataset)
    meta = get_feature_meta_default(x, y, feature_names=feature_names, label_name="label")
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

        # reuse code from demo_aad.py
        model, _, queried, _, _ = detect_anomalies_and_describe(x, y, opts)

        describer = CompactDescriber(x, y, model, opts, sample_negative=True)
        _, _, rules = describer.describe(np.array(queried, dtype=np.int32))
        precision, recall, f1 = evaluate_ruleset(x, y, rules, average="binary")
        logger.debug("precision: %f, recall: %f, f1: %f" % (precision, recall, f1))

        # we will recompute the Bayesian ruleset every time just for DEBUG
        r_top, r_compact, _ = get_rulesets(x, y, queried=queried, model=model,
                                           meta=meta, opts=opts, bayesian=False)

        rules_top, regions_top, str_rules_top, _ = r_top
        rules_compact, regions_compact, str_rules_compact, _ = r_compact

        save_strings_to_file(str_rules_compact, file_path_compact)
        save_strings_to_file(str_rules_top, file_path_top)

        save_strings_to_file([",".join([str(v) for v in queried])], file_path_queried)
    else:
        logger.debug("Rules file(s) found ... reading rules from file")

        str_rules_top = load_strings_from_file(file_path_top)
        str_rules_compact = load_strings_from_file(file_path_compact)
        str_queried = load_strings_from_file(file_path_queried)[0]

        queried = [int(v) for v in str_queried.split(",")]
        logger.debug("queried:\n%s" % str(queried))
        rules_compact, str_rules_compact = prepare_conjunctive_rulesets(x, y, meta=meta, str_rules=str_rules_compact, opts=opts)
        rules_top, str_rules_top = prepare_conjunctive_rulesets(x, y, meta=meta, str_rules=str_rules_top, opts=opts)

        regions_top = convert_conjunctive_rules_to_feature_ranges(rules_top, meta)
        regions_compact = convert_conjunctive_rules_to_feature_ranges(rules_compact, meta)

    logger.debug("Top regions:\n%s" % str(regions_top))
    logger.debug("Compact regions:\n%s" % str(regions_compact))
    logger.debug("Compact ruleset:\n  %s" % "\n  ".join(str_rules_compact))

    plot_annotations = True
    plot_bayesian_rulesets = True

    bayesian_describer = BayesianRulesetsDescriber(x, y, model=None, opts=opts, meta=meta, candidate_rules=rules_top)
    regids_bayesian, regions_bayesian, rules_bayesian = bayesian_describer.describe(instance_indexes=queried)
    str_rules_bayesian = convert_conjunctive_rules_to_strings(rules_bayesian)

    logger.debug("Bayesian regions:\n%s" % str(regions_bayesian))
    logger.debug("Bayesian ruleset:\n  %s" % "\n  ".join(str_rules_bayesian))

    save_strings_to_file(str_rules_bayesian, file_path_bayesian)

    _, _, f1_compact = evaluate_ruleset(x, y, rules_compact, average="weighted")
    _, _, f1_bayesian = evaluate_ruleset(x, y, rules_bayesian, average="weighted")
    print("F1 scores: compact descriptions: %f; bayesian: %f" % (f1_compact, f1_bayesian))

    if x.shape[1] == 2 and opts.plot2D:
        path = os.path.join(opts.resultsdir, "%s_rulesets.pdf" % opts.dataset)
        dp = DataPlotter(pdfpath=path, rows=2, cols=2, save_tight=True)
        plot_selected_regions(x, y, regions=regions_top, query_instances=queried,
                              title="Candidate Regions (Most Anomalous)\nAfter BAL (budget: %d, #regions: %d)" %
                                    (opts.budget, len(regions_top)), dp=dp)
        plot_selected_regions(x, y, regions=regions_compact, query_instances=queried,
                              title="Compact Descriptions\nWith Interpretability", dp=dp)
        if plot_bayesian_rulesets:
            plot_selected_regions(x, y, regions=regions_bayesian, query_instances=queried,
                                  title="Bayesian Rulesets\nWang, Rudin, et al. (2016)", dp=dp)
        if plot_annotations:
            # plot the inferred rules
            plot_rule_annotations(str_rules_compact,
                                  str_bayesian_rules=str_rules_bayesian if plot_bayesian_rulesets else None,
                                  dp=dp)
        dp.close()


if __name__ == "__main__":

    # Prepare the aad arguments. It is easier to first create the parsed args and
    # then create the actual AadOpts from the args
    args = get_aad_command_args(debug=True, debug_args=get_debug_args(dataset="toy2"))
    configure_logger(args)

    opts = AadOpts(args)
    logger.debug(opts.str_opts())

    random.seed(opts.randseed)
    np.random.seed(opts.randseed+1)

    test_aad_rules(opts)
