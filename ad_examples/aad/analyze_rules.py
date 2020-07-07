import os
import numpy as np

import logging

import matplotlib.pyplot as plt

from ..common.utils import read_csv, read_data_as_matrix, configure_logger
from ..common.expressions import (
    load_strings_from_file, evaluate_ruleset, get_feature_meta_default,
    convert_strings_to_conjunctive_rules
)
from ..common.data_plotter import DataPlotter
from .aad_globals import get_aad_command_args, AadOpts
from .anomaly_dataset_support import dataset_configs, dataset_feature_names


def load_rules(x, y, meta, fileprefix, out_dir, opts, evaluate_f1=True):
    f1s = dict()
    precisions = dict()
    recalls = dict()
    rule_lengths = dict()
    num_rules = dict()
    rules_data = []
    for iter in range(opts.rule_output_interval, opts.budget+1, opts.rule_output_interval):
        filepath = os.path.join(out_dir, "%s_%d.txt" % (fileprefix, iter))
        if os.path.isfile(filepath):
            str_rules = load_strings_from_file(filepath)
            rules = convert_strings_to_conjunctive_rules(str_rules, meta)
            if len(rules) == 0:
                logger.debug("No rules found in iter %d of %s" % (iter, fileprefix))
            else:
                if evaluate_f1:
                    precision, recall, f1 = evaluate_ruleset(x, y, rules, average="binary")
                    logger.debug("Iter %d, F1 score: %f" % (iter, f1))
                else:
                    precision = recall = f1 = 0.
                # logger.debug("\n  " + "\n  ".join(str_rules))
                rules_data.append((opts.runidx, iter, rules, str_rules, f1))
                f1s[iter] = f1
                precisions[iter] = precision
                recalls[iter] = recall
                num_rules[iter] = len(rules)
                rule_lengths[iter] = np.mean([len(rule) for rule in rules])
                logger.debug("iter: %d, rule_lengths: %f" % (iter, rule_lengths[iter]))
        else:
            logger.debug("file not found:\n%s" % filepath)
            rules_data.append((opts.runidx, iter, None, None, 0))
    return {"f1s": f1s, "recalls": recalls, "precisions": precisions,
            "lengths": rule_lengths, "num_rules": num_rules, "data": rules_data}


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
    precisions = summarize_values(accumulate_values([v["precisions"] for v in values], dict()))
    recalls = summarize_values(accumulate_values([v["recalls"] for v in values], dict()))
    lengths = summarize_values(accumulate_values([v["lengths"] for v in values], dict()))
    num_rules = summarize_values(accumulate_values([v["num_rules"] for v in values], dict()))
    rules_data = []
    for v in values:
        rules_data.extend(v["data"])
    logger.debug("f1s:\n%s" % str(f1s))
    logger.debug("precisions:\n%s" % str(precisions))
    logger.debug("recalls:\n%s" % str(recalls))
    logger.debug("lengths:\n%s" % str(lengths))
    logger.debug("num_rules:\n%s" % str(num_rules))

    return {"f1s": f1s, "recalls": recalls, "precisions": precisions,
            "lengths": lengths, "num_rules": num_rules, "data": rules_data}


def string_agg_scores(agg_scores):
    str_f1s = []
    str_lengths = []
    for key in sorted(agg_scores.keys()):
        vals = agg_scores[key]
        str_f1s.append("%f (%f)" % (vals[0][0], vals[1][0]))
        str_lengths.append("%f (%f)" % (vals[0][1], vals[1][1]))
    return str_f1s, str_lengths


def load_all_rule_data(x, y, meta, opts, evaluate_f1=True):

    acc_top = []
    acc_compact = []
    acc_bayesian = []

    for runidx in opts.get_runidxs():
        opts.set_multi_run_options(1, runidx)

        fileprefix_top = "%s_top_rules" % opts.get_alad_metrics_name_prefix()
        fileprefix_compact = "%s_compact_rules" % opts.get_alad_metrics_name_prefix()
        fileprefix_bayesian = "%s_bayesian_rules" % opts.get_alad_metrics_name_prefix()
        fileprefix_queries = "%s_queried" % opts.get_alad_metrics_name_prefix()

        logger.debug("Loading top rules")
        # do not evaluate F1 scores for Top because it is expensive and we do not plot these
        top_data = load_rules(x, y, meta, fileprefix_top, out_dir=opts.resultsdir,
                              opts=opts, evaluate_f1=False)
        acc_top.append(top_data)

        logger.debug("\nLoading compact rules")
        compact_data = load_rules(x, y, meta, fileprefix_compact, out_dir=opts.resultsdir,
                                  opts=opts, evaluate_f1=evaluate_f1)
        acc_compact.append(compact_data)

        logger.debug("\nLoading bayesian rules")
        bayesian_data = load_rules(x, y, meta, fileprefix_bayesian, out_dir=opts.resultsdir,
                                   opts=opts, evaluate_f1=evaluate_f1)
        acc_bayesian.append(bayesian_data)

    logger.debug("Aggregated Top:")
    agg_top = aggregate_rules_data(acc_top)

    logger.debug("Aggregated Compact:")
    agg_compact = aggregate_rules_data(acc_compact)

    logger.debug("Aggregated Bayesian:")
    agg_bayesian = aggregate_rules_data(acc_bayesian)

    return agg_top, agg_compact, agg_bayesian


def write_all_summaries(summary, name, opts):
    for rname in ["f1s", "precisions", "recalls", "lengths", "num_rules"]:
        filename = "%s-all_%s_%s.csv" % (opts.dataset, name, rname)
        filepath = os.path.join(opts.resultsdir, filename)
        np.savetxt(filepath, summary[rname], fmt='%0.4f', delimiter=',')


def load_summary(name, opts):
    summary = dict()
    for rname in ["f1s", "precisions", "recalls", "lengths", "num_rules"]:
        filename = "%s-all_%s_%s.csv" % (opts.dataset, name, rname)
        filepath = os.path.join(opts.resultsdir, filename)
        logger.debug("loading %s" % filepath)
        summ = np.asmatrix(read_csv(file=filepath, header=None, sep=","))
        summary[rname] = summ
    return summary


def found_precomputed_summaries(opts):
    for name in ["top", "compact", "bayesian"]:
        for rname in ["f1s", "precisions", "recalls", "lengths", "num_rules"]:
            filename = "%s-all_%s_%s.csv" % (opts.dataset, name, rname)
            filepath = os.path.join(opts.resultsdir, filename)
            if not os.path.exists(filepath):
                logger.debug("summary file not found:\n%s" % filepath)
                return False
    return True


def plot_scores(agg_top, agg_compact, agg_bayesian, score_type, dp, opts, is_title=True):
    score_name = {"f1s": "F1 Score", "precisions": "Precision", "recalls": "Recall"}
    legend_handles = []
    pl = dp.get_next_plot()
    plt.xlabel('Feedback iterations', fontsize=8)
    plt.ylabel(score_name[score_type], fontsize=8)
    plt.ylim([0, 1])
    if is_title:
        plt.title("%s" % dataset_configs[opts.dataset][4], fontsize=8)

    ln, = pl.plot(agg_compact[score_type][:, 0], agg_compact[score_type][:, 1],
                  "-", color="red", linewidth=1, label="Compact Descriptions")
    legend_handles.append(ln)

    ln, = pl.plot(agg_bayesian[score_type][:, 0], agg_bayesian[score_type][:, 1],
                  "-", color="blue", linewidth=1, label="Bayesian Rulesets")
    legend_handles.append(ln)

    if opts.dataset == "toy2":
        # just for adding a legend for candidate rules, add a dummy curve
        ln, = pl.plot([0, 0], [0, 0], "-", color="grey", linewidth=1, label="Candidate Rules")
        legend_handles.append(ln)

    if opts.dataset in ["abalone", "ann_thyroid_1v3"] or (opts.dataset == "toy2" and score_type == "precisions"):
        pl.legend(handles=legend_handles, loc='lower right', prop={'size': 6})


def plot_rule_lengths(agg_top, agg_compact, agg_bayesian, dp, opts, is_title=True):
    legend_handles = []
    pl = dp.get_next_plot()
    plt.xlabel('Feedback iterations', fontsize=8)
    plt.ylabel('Rule length', fontsize=8)
    max_rule_length = max(np.max(agg_compact["lengths"][:, 1]), np.max(agg_bayesian["lengths"][:, 1]))
    plt.ylim([0, max_rule_length+1])
    if is_title:
        plt.title("%s" % dataset_configs[opts.dataset][4], fontsize=8)

    ln, = pl.plot(agg_compact["lengths"][:, 0], agg_compact["lengths"][:, 1],
                  "-", color="red", linewidth=1, label="Compact Descriptions")
    legend_handles.append(ln)

    ln, = pl.plot(agg_bayesian["lengths"][:, 0], agg_bayesian["lengths"][:, 1],
                  "-", color="blue", linewidth=1, label="Bayesian Rulesets")
    legend_handles.append(ln)

    if opts.dataset in ["abalone", "ann_thyroid_1v3"]:
        pl.legend(handles=legend_handles, loc='upper right', prop={'size': 6})


def plot_num_rules(agg_top, agg_compact, agg_bayesian, dp, opts, is_title=True):
    legend_handles = []
    pl = dp.get_next_plot()
    plt.xlabel('Feedback iterations', fontsize=8)
    plt.ylabel('# Rules', fontsize=8)
    if is_title:
        plt.title("%s" % dataset_configs[opts.dataset][4], fontsize=8)

    ln, = pl.plot(agg_top["num_rules"][:, 0], agg_top["num_rules"][:, 1],
                  "-", color="grey", linewidth=1, label="Candidate Rules")
    legend_handles.append(ln)

    ln, = pl.plot(agg_compact["num_rules"][:, 0], agg_compact["num_rules"][:, 1],
                  "-", color="red", linewidth=1, label="Compact Descriptions")
    legend_handles.append(ln)

    ln, = pl.plot(agg_bayesian["num_rules"][:, 0], agg_bayesian["num_rules"][:, 1],
                  "-", color="blue", linewidth=1, label="Bayesian Rulesets")
    legend_handles.append(ln)

    if opts.dataset in ["abalone", "ann_thyroid_1v3"]:
        pl.legend(handles=legend_handles, loc='upper right', prop={'size': 6})


def plot_blank_image(dp, message="Intentionally Blank"):
    pl = dp.get_next_plot()
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    pl.text(-0.3, 0, message, fontsize=6, color="black")


def swap_metadata(rules_data, meta):
    for rl in rules_data:
        if rl[2] is None:
            # possibly the rule file was missing at time of load (see load_rules() above)
            continue
        for rule in rl[2]:
            rule.meta = meta
        # logger.debug("runidx: %d, iter: %d\n  %s" % (rl[0], rl[1], "\n  ".join([str(v) for v in rl[2]])))


def print_readable_rules(opts):
    feature_names = dataset_feature_names.get(opts.dataset, None)
    if feature_names is None:
        logger.debug("Readable names not found...")
        return
    X_train, labels = read_data_as_matrix(opts)
    new_meta = get_feature_meta_default(X_train, labels, feature_names=feature_names)
    default_meta = get_feature_meta_default(X_train, labels, feature_names=None)
    agg_top, agg_compact, agg_bayesian = load_all_rule_data(X_train, labels, default_meta,
                                                            opts, evaluate_f1=False)
    swap_metadata(agg_top["data"], new_meta)
    swap_metadata(agg_compact["data"], new_meta)
    swap_metadata(agg_bayesian["data"], new_meta)

    logger.debug("Printing Compact Descriptions:")
    for rules_data in agg_compact["data"]:
        if rules_data[1] == opts.budget:
            str_rules = "\n  or ".join(["(" + str(r) + ")" for r in rules_data[2]])
            logger.debug("Iter %d:\n  %s" % (rules_data[1], str_rules))

    logger.debug("Printing Bayesian Rulesets:")
    for rules_data in agg_bayesian["data"]:
        if rules_data[1] == opts.budget:
            str_rules = "\n  or ".join(["(" + str(r) + ")" for r in rules_data[2]])
            logger.debug("Iter %d:\n  %s" % (rules_data[1], str_rules))


def analyze_rules_dataset(opts, dp, is_num_rules=True, is_blank=True, is_title=True):

    if not found_precomputed_summaries(opts):
        logger.debug("Precomputed summaries not found. Regenerating...")
        X_train, labels = read_data_as_matrix(opts)
        meta = get_feature_meta_default(X_train, labels, feature_names=None)

        agg_top, agg_compact, agg_bayesian = load_all_rule_data(X_train, labels, meta, opts)

        feature_names = dataset_feature_names.get(opts.dataset, None)
        if feature_names is not None:
            # put in user-friendly column names
            new_meta = get_feature_meta_default(x=X_train, y=labels, feature_names=feature_names)

            logger.debug("Swapping metadata of candidate rules")
            swap_metadata(agg_top["data"], new_meta)

            logger.debug("Swapping metadata of compact rules")
            swap_metadata(agg_compact["data"], new_meta)

            logger.debug("Swapping metadata of bayesian rules")
            swap_metadata(agg_bayesian["data"], new_meta)

        write_all_summaries(agg_top, "top", opts)
        write_all_summaries(agg_compact, "compact", opts)
        write_all_summaries(agg_bayesian, "bayesian", opts)
    else:
        logger.debug("Precomputed summaries found.")
        agg_top = load_summary("top", opts)
        agg_compact = load_summary("compact", opts)
        agg_bayesian = load_summary("bayesian", opts)

    if opts.plot2D:
        plot_scores(agg_top, agg_compact, agg_bayesian, "f1s", dp, opts, is_title=is_title)
        plot_scores(agg_top, agg_compact, agg_bayesian, "precisions", dp, opts, is_title=is_title)
        plot_scores(agg_top, agg_compact, agg_bayesian, "recalls", dp, opts, is_title=is_title)
        plot_rule_lengths(agg_top, agg_compact, agg_bayesian, dp, opts, is_title=is_title)
        if is_num_rules:
            plot_num_rules(agg_top, agg_compact, agg_bayesian, dp, opts, is_title=is_title)
        if is_blank:
            plot_blank_image(dp, message="Intentionally\nBlank")  # placeholder for alignment


def analyze_rules(opts):
    plot_short_summary = False
    if opts.dataset != "Xall":
        pdfpath = "%s/%s-f1_scores.pdf" % (opts.resultsdir, opts.dataset)
        if plot_short_summary:
            dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=2, save_tight=True)
            print_readable_rules(opts)
            analyze_rules_dataset(opts, dp=dp, is_num_rules=False, is_blank=False, is_title=False)
        else:
            dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=3, save_tight=True)
            analyze_rules_dataset(opts, dp=dp, is_num_rules=True, is_blank=True, is_title=True)
        dp.close()
        return

    datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1', 'covtype',
                'mammography', 'kddcup', 'shuttle_1v23567', 'weather', 'electricity'
                ]
    pdfpath = "%s/all-rule_analysis.pdf" % (opts.resultsdir)
    dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=3, save_tight=True)
    datafile = opts.datafile
    resultsdir = opts.resultsdir
    for dataset in datasets:
        opts.dataset = dataset
        if opts.dataset in ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1']:
            opts.budget = 300
        else:
            opts.budget = 500
        opts.datafile = datafile.replace("Xall", dataset)
        opts.resultsdir = resultsdir.replace("Xall", dataset)
        opts.resultsdir = opts.resultsdir.replace("bd500", "bd%d" % opts.budget)
        print("Analyzing %s" % opts.dataset)
        analyze_rules_dataset(opts, dp=dp)
    dp.close()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    args = get_aad_command_args(debug=False)
    # print "log file: %s" % args.log_file
    configure_logger(args)

    opts = AadOpts(args)
    logger.debug(opts.str_opts())

    analyze_rules(opts)
