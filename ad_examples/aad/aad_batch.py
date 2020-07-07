import os
import logging
import numpy as np

from ..common.utils import configure_logger, read_data_as_matrix, Timer, rbind
from ..common.expressions import save_strings_to_file
from .aad_globals import get_aad_command_args, ORIG_TREE_SCORE_TYPE, detector_types, AadOpts
from .aad_base import AadEventListener, EVT_AFTER_FEEDBACK, Ensemble
from .forest_aad_detector import is_forest_detector
from .aad_support import (
    get_queried_indexes, write_baseline_query_indexes, get_aad_model, summarize_ensemble_num_seen,
    SequentialResults, write_sequential_results_to_csv
)
from .aad_test_support import aad_unit_tests_battery, \
    evaluate_forest_original, debug_qvals, \
    plot_tsne_queries
from .aad_ruleset_support import get_rulesets


class AadListenerForRules(AadEventListener):
    def __init__(self, x, y):
        AadEventListener.__init__(self)
        self.x = x
        self.y = y
        self.r_top = []
        self.r_compact = []
        self.r_bayesian = []
        self.all_queried = []

    def __call__(self, event_type, x, y, iter, queried, model, opts):
        if event_type != EVT_AFTER_FEEDBACK or not is_forest_detector(opts.detector_type):
            return

        if (iter+1) % opts.rule_output_interval == 0:
            if opts.compact_rules or opts.bayesian_rules:
                # in the below, we do *NOT* use the input x, y because that x is transformed.
                # instead, we will use self.x, self.y where self.x has the original features.
                r_top, r_compact, r_bayesian = get_rulesets(self.x, self.y, queried=queried,
                                                            model=model, meta=None,
                                                            opts=opts,
                                                            bayesian=opts.bayesian_rules)

                self.all_queried.append((iter, np.array(queried, dtype=np.int32)))
                self.r_top.append((iter, r_top))
                self.r_compact.append((iter, r_compact))

                if r_compact is not None:
                    _, _, str_rules_compact, _ = r_compact
                    # logger.debug("Compact rules:\n  %s" % "\n  ".join(str_rules_compact))

                if r_bayesian is not None:
                    self.r_bayesian.append((iter, r_bayesian))

                    _, _, str_rules_bayesian, _ = r_bayesian
                    # logger.debug("Bayesian ruleset:\n  %s" % "\n  ".join(str_rules_bayesian))

    def write_rules_to_file(self, rules_data, fileprefix, out_dir):
        if len(rules_data) == 0:
            return
        for i, r_info in enumerate(rules_data):
            iter, r_rules = r_info
            if r_rules is not None:
                _, _, str_rules, _ = r_rules
            else:
                str_rules = []
            filepath = os.path.join(out_dir, "%s_%d.txt" % (fileprefix, iter+1))
            save_strings_to_file(str_rules, file_path=filepath)

    def write_all_queries_to_file(self, fileprefix, out_dir):
        if len(self.all_queried) == 0:
            return

        for iter, queried in self.all_queried:
            filepath = os.path.join(out_dir, "%s_%d.txt" % (fileprefix, iter+1))
            save_strings_to_file([",".join([str(v) for v in queried])], file_path=filepath)

    def output_all_data(self, opts):
        fileprefix_top = "%s_top_rules" % opts.get_alad_metrics_name_prefix()
        fileprefix_compact = "%s_compact_rules" % opts.get_alad_metrics_name_prefix()
        fileprefix_bayesian = "%s_bayesian_rules" % opts.get_alad_metrics_name_prefix()
        fileprefix_queries = "%s_queried" % opts.get_alad_metrics_name_prefix()

        self.write_rules_to_file(self.r_top, fileprefix_top, out_dir=opts.resultsdir)
        self.write_rules_to_file(self.r_compact, fileprefix_compact, out_dir=opts.resultsdir)
        self.write_rules_to_file(self.r_bayesian, fileprefix_bayesian, out_dir=opts.resultsdir)
        self.write_all_queries_to_file(fileprefix_queries, out_dir=opts.resultsdir)


def aad_batch():

    logger = logging.getLogger(__name__)

    dense = False  # DO NOT Change this!

    args = get_aad_command_args(debug=False)
    # print "log file: %s" % args.log_file
    configure_logger(args)

    opts = AadOpts(args)
    # print opts.str_opts()
    logger.debug(opts.str_opts())

    if opts.streaming:
        raise ValueError("Streaming not supported. Use aad_stream.py for streaming algorithm.")

    run_aad = True
    run_tests = opts.plot2D and opts.reruns == 1 and \
                (not is_forest_detector(opts.detector_type) or opts.forest_score_type != ORIG_TREE_SCORE_TYPE)

    baseline_query_indexes_only = False

    X_train, labels = read_data_as_matrix(opts)

    # X_train = X_train[0:10, :]
    # labels = labels[0:10]

    logger.debug("loaded file: %s" % opts.datafile)
    logger.debug("results dir: %s" % opts.resultsdir)
    logger.debug("detector_type: %s" % detector_types[opts.detector_type])

    model = None
    X_train_new = None
    metrics = None
    if run_aad:
        # use this to run AAD

        opts.fid = 1

        all_num_seen = None
        all_num_seen_baseline = None
        all_queried_indexes = None
        all_queried_indexes_baseline = None

        all_baseline = ""
        all_orig_iforest = ""

        all_orig_num_seen = None

        baseline_query_info = []

        for runidx in opts.get_runidxs():
            tm_run = Timer()
            opts.set_multi_run_options(opts.fid, runidx)

            random_state = np.random.RandomState(args.randseed + opts.fid * opts.reruns + runidx)

            # fit the model
            event_listener = AadListenerForRules(X_train, labels)
            model = get_aad_model(X_train, opts, random_state, event_listener=event_listener)
            model.fit(X_train)

            if is_forest_detector(opts.detector_type) and \
                    opts.forest_score_type == ORIG_TREE_SCORE_TYPE:
                orig_num_seen = evaluate_forest_original(X_train, labels, opts.budget, model, x_new=None)
                tmp = np.zeros((1, 2+orig_num_seen.shape[1]), dtype=orig_num_seen.dtype)
                tmp[0, 0:2] = [opts.fid, runidx]
                tmp[0, 2:tmp.shape[1]] = orig_num_seen[0, :]
                all_orig_num_seen = rbind(all_orig_num_seen, tmp)
                logger.debug(tm_run.message("Original detector runidx: %d" % runidx))
                continue

            if is_forest_detector(opts.detector_type):
                logger.debug("total #nodes: %d" % (len(model.all_regions)))

            X_train_new = model.transform_to_ensemble_features(X_train, dense=dense, norm_unit=opts.norm_unit)

            if False and opts.norm_unit:
                norms = X_train_new.power(2).sum(axis=1)
                logger.debug("norms:\n%s" % str(list(norms.T)))

            baseline_w = model.get_uniform_weights()

            agg_scores = model.get_score(X_train_new, baseline_w)
            if False and is_forest_detector(opts.detector_type):
                original_scores = 0.5 - model.decision_function(X_train)
                queried = np.argsort(-original_scores)
                n_found = np.cumsum(labels[queried[np.arange(opts.budget)]])
                logger.debug("#anomalies found by original detector:\n%s" % str(list(n_found)))

            if baseline_query_indexes_only:
                baseline_query_info.append(get_queried_indexes(agg_scores, labels, opts))
                continue

            ensemble = Ensemble(X_train, labels, X_train_new, baseline_w,
                                agg_scores=agg_scores, original_indexes=np.arange(X_train.shape[0]),
                                auc=0.0, model=None)

            # model.init_weights(init_type=opts.init, samples=X_train_new)
            model.init_weights(init_type=opts.init, samples=None)

            metrics = model.aad_learn_ensemble_weights_with_budget(ensemble, opts)

            if metrics is not None:
                num_seen, num_seen_baseline, queried_indexes, queried_indexes_baseline = \
                    summarize_ensemble_num_seen(ensemble, metrics, fid=opts.fid)
                all_num_seen = rbind(all_num_seen, num_seen)
                all_num_seen_baseline = rbind(all_num_seen_baseline, num_seen_baseline)
                all_queried_indexes = rbind(all_queried_indexes, queried_indexes)
                all_queried_indexes_baseline = rbind(all_queried_indexes_baseline, queried_indexes_baseline)
                logger.debug("baseline: \n%s" % str([v for v in num_seen_baseline[0, :]]))
                logger.debug("num_seen: \n%s" % str([v for v in num_seen[0, :]]))

                if False:
                    debug_qvals(X_train_new, model, metrics, args.resultsdir, opts)
            else:
                queried = np.argsort(-agg_scores)
                n_found = np.cumsum(labels[queried[np.arange(60)]])
                all_baseline = all_baseline + ",".join([str(v) for v in n_found]) + os.linesep

                orig_iforest_scores = model.decision_function(X_train)  # smaller is more anomalous
                queried = np.argsort(orig_iforest_scores)
                n_found = np.cumsum(labels[queried[np.arange(60)]])
                all_orig_iforest = all_orig_iforest + ",".join([str(v) for v in n_found]) + os.linesep

            event_listener.output_all_data(opts)
            logger.debug(tm_run.message("Completed runidx: %d" % runidx))

            if runidx == 1 and False:
                plot_tsne_queries(X_train, labels, ensemble, metrics, opts)

            if not run_tests:
                metrics = None  # release memory
                model = None
                X_train_new = None
                ensemble = None

        if all_num_seen is not None:
            results = SequentialResults(num_seen=all_num_seen, num_seen_baseline=all_num_seen_baseline,
                                        true_queried_indexes=all_queried_indexes,
                                        true_queried_indexes_baseline=all_queried_indexes_baseline)
            write_sequential_results_to_csv(results, opts)
        else:
            logger.debug("baseline:\n%s\norig iforest:\n%s" % (all_baseline, all_orig_iforest))

        if all_orig_num_seen is not None:
            prefix = opts.get_alad_metrics_name_prefix()
            orig_num_seen_file = os.path.join(opts.resultsdir, "%s-orig_num_seen.csv" % (prefix,))
            np.savetxt(orig_num_seen_file, all_orig_num_seen, fmt='%d', delimiter=',')

        if len(baseline_query_info) > 0:
            write_baseline_query_indexes(baseline_query_info, opts)

    if run_tests:
        aad_unit_tests_battery(X_train, labels, model, metrics, opts,
                               args.resultsdir, dataset_name=args.dataset)


if __name__ == "__main__":
    aad_batch()
