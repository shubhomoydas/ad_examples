import os
import numpy as np

import logging

from aad.aad_ruleset_support import *
import matplotlib.pyplot as plt
from common.data_plotter import DataPlotter


def load_rules(x, y, meta, fileprefix, out_dir, opts):
    rule_scores = dict()
    rule_lengths = dict()
    rules_data = []
    for iter in range(opts.rule_output_interval, opts.budget+1, opts.rule_output_interval):
        filepath = os.path.join(out_dir, "%s_%d.txt" % (fileprefix, iter))
        if os.path.isfile(filepath):
            str_rules = load_strings_from_file(filepath)
            rules = convert_strings_to_conjunctive_rules(str_rules, meta)
            if len(rules) == 0:
                logger.debug("No rules found in iter %d of %s" % (iter, fileprefix))
            else:
                f1 = evaluate_ruleset(x, y, rules, average="binary")
                logger.debug("Iter %d, F1 score: %f" % (iter, f1))
                # logger.debug("\n  " + "\n  ".join(str_rules))
                rules_data.append((iter, rules, str_rules, f1))
                rule_scores[iter] = f1
                rule_lengths[iter] = len(rules)
        else:
            logger.debug("file not found:\n%s" % filepath)
            rules_data.append((iter, None, None, 0))
    return {"f1s": rule_scores, "lengths": rule_lengths, "data": rules_data}


def accumulate_values(all_values, acc_values):
    # input is a list of dicts
    for values in all_values:
        for key, val in values.items():
            if key not in acc_values:
                acc_values[key] = []
            acc_values[key].append(val)
    return acc_values


def summarize_values(values):
    summary = np.ndarray(shape=(0, 3), dtype=np.float32)
    for key in sorted(values.keys()):
        s = np.array([key, np.mean(values[key]), np.std(values[key])], dtype=np.float32).reshape((1, -1))
        summary = np.vstack((summary, s))
    return summary


def aggregate_rules_data(values):
    f1s = summarize_values(accumulate_values([v["f1s"] for v in values], dict()))
    lengths = summarize_values(accumulate_values([v["lengths"] for v in values], dict()))
    logger.debug("f1s:\n%s" % str(f1s))
    logger.debug("lengths:\n%s" % str(lengths))

    return {"f1s": f1s, "lengths": lengths}


def string_agg_scores(agg_scores):
    str_f1s = []
    str_lengths = []
    for key in sorted(agg_scores.keys()):
        vals = agg_scores[key]
        str_f1s.append("%f (%f)" % (vals[0][0], vals[1][0]))
        str_lengths.append("%f (%f)" % (vals[0][1], vals[1][1]))
    return str_f1s, str_lengths


def load_all_rule_data(x, y, meta, opts):

    acc_top = []
    acc_compact = []
    acc_bayesian = []

    for runidx in opts.get_runidxs():
        opts.set_multi_run_options(1, runidx)

        fileprefix_top = "%s_top_rules" % opts.get_alad_metrics_name_prefix()
        fileprefix_compact = "%s_compact_rules" % opts.get_alad_metrics_name_prefix()
        fileprefix_bayesian = "%s_bayesian_rules" % opts.get_alad_metrics_name_prefix()
        fileprefix_queries = "%s_queried" % opts.get_alad_metrics_name_prefix()

        logger.debug("Loading top rules\n")
        top_data = load_rules(x, y, meta, fileprefix_top, out_dir=opts.resultsdir, opts=opts)
        acc_top.append(top_data)

        logger.debug("Loading compact rules\n")
        compact_data = load_rules(x, y, meta, fileprefix_compact, out_dir=opts.resultsdir, opts=opts)
        acc_compact.append(compact_data)

        logger.debug("Loading bayesian rules\n")
        bayesian_data = load_rules(x, y, meta, fileprefix_bayesian, out_dir=opts.resultsdir, opts=opts)
        acc_bayesian.append(bayesian_data)

    logger.debug("Aggregated Top:")
    agg_top = aggregate_rules_data(acc_top)

    logger.debug("Aggregated Compact:")
    agg_compact = aggregate_rules_data(acc_compact)

    logger.debug("Aggregated Bayesian:")
    agg_bayesian = aggregate_rules_data(acc_bayesian)

    return agg_top, agg_compact, agg_bayesian


def analyze_rules():

    logger = logging.getLogger(__name__)

    args = get_aad_command_args(debug=False)
    # print "log file: %s" % args.log_file
    configure_logger(args)

    opts = AadOpts(args)
    logger.debug(opts.str_opts())

    X_train, labels = read_data_as_matrix(opts)
    meta = get_feature_meta_default(X_train, labels)

    agg_top, agg_compact, agg_bayesian = load_all_rule_data(X_train, labels, meta, opts)

    if opts.plot2D:
        pdfpath = "%s/%s.pdf" % (opts.resultsdir, "f1_scores")
        legend_handles = []
        dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=2, save_tight=True)
        pl = dp.get_next_plot()
        plt.xlabel('Feedback iterations', fontsize=8)
        plt.ylabel('F1 Score', fontsize=8)
        # plt.ylim([0.5, 1])
        plt.title("Comparison of F1 scores", fontsize=8)

        ln, = pl.plot(agg_compact["f1s"][:, 0], agg_compact["f1s"][:, 1],
                      "-", color="red", linewidth=1, label="Compact Descriptions")
        legend_handles.append(ln)

        ln, = pl.plot(agg_bayesian["f1s"][:, 0], agg_bayesian["f1s"][:, 1],
                      "-", color="blue", linewidth=1, label="Bayesian Rulesets")
        legend_handles.append(ln)

        pl.legend(handles=legend_handles, loc='lower right', prop={'size': 8})

        legend_handles = []
        pl = dp.get_next_plot()
        plt.xlabel('Feedback iterations', fontsize=8)
        plt.ylabel('Rule Length', fontsize=8)
        plt.title("Comparison of Rule lengths", fontsize=8)

        ln, = pl.plot(agg_top["lengths"][:, 0], agg_top["lengths"][:, 1],
                      "-", color="grey", linewidth=1, label="Candidate Rules")
        legend_handles.append(ln)

        ln, = pl.plot(agg_compact["lengths"][:, 0], agg_compact["lengths"][:, 1],
                      "-", color="red", linewidth=1, label="Compact Descriptions")
        legend_handles.append(ln)

        ln, = pl.plot(agg_bayesian["lengths"][:, 0], agg_bayesian["lengths"][:, 1],
                      "-", color="blue", linewidth=1, label="Bayesian Rulesets")
        legend_handles.append(ln)

        pl.legend(handles=legend_handles, loc='upper right', prop={'size': 8})

        dp.close()


if __name__ == "__main__":
    analyze_rules()
