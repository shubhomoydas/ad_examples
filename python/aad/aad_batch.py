import os
import numpy as np

import logging

from common.utils import *

from aad.aad_globals import *
from aad.aad_support import *
from aad.aad_test_support import aad_unit_tests_battery, \
    get_queried_indexes, write_baseline_query_indexes, \
    evaluate_forest_original, debug_qvals, check_random_vector_angle, \
    plot_tsne_queries
from aad.forest_description import *


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
            model = get_aad_model(X_train, opts, random_state)
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

            if metrics is not None and opts.describe_anomalies and is_forest_detector(opts.detector_type):
                descriptions = get_anomaly_descriptions(X_train, labels, model, metrics, opts)

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
