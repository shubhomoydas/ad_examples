import os
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import logging
from pandas import DataFrame

from common.data_plotter import *

from aad.aad_globals import *
from aad.aad_support import *
from aad.forest_description import *
from aad.anomaly_dataset_support import *


"""
pythonw -m aad.plot_aad_results
"""


def get_n_intermediate(x, n=10):
    m = len(x)
    p = np.round(np.arange(0, n) * (m * 1. / n)).astype(dtype=int)
    return x[p]


def plot_results(results, cols, pdffile, num_seen=0, num_anoms=0,
                 plot_sd=False, ylabel=None,
                 legend_loc='lower right', legend_datasets=None, axis_fontsize=20, legend_size=14):
    dataset = results[0][0]
    dp = DataPlotter(pdfpath=pdffile, rows=1, cols=1)
    pl = dp.get_next_plot()

    plt.xlim([0, num_seen])
    if num_anoms < 0:
        # plot actual number of anomalies discovered instead of the fraction
        ylabel = '# of anomalies seen' if ylabel is None else ylabel
        ylim = 0.0
        for result in results:
            ylim = max(ylim, np.max(result[2]))
        plt.ylim([0., ylim+2])
    else:
        ylabel = '% of total anomalies seen' if ylabel is None else ylabel
        plt.ylim([0., 100.])

    plt.xlabel('# instances labeled', fontsize=axis_fontsize)
    plt.ylabel(ylabel, fontsize=axis_fontsize)

    for i, result in enumerate(results):
        num_found_avg = result[2]
        num_found_sd = result[3]
        if num_anoms > 0:
            num_found_avg = num_found_avg * 100. / num_anoms
            num_found_sd = num_found_sd * 100. / num_anoms
        logger.debug("label: %s" % result[1])
        pl.plot(np.arange(len(num_found_avg)), num_found_avg, '-',
                color=cols[i], linewidth=1, label=result[1])
        if plot_sd:
            pts = get_n_intermediate(np.arange(len(num_found_avg) + i*5, dtype=int))
            pts = np.minimum(len(num_found_avg)-1, pts)
            pl.errorbar(pts, num_found_avg[pts], yerr=1.96*num_found_sd[pts], fmt='.', color=cols[i])
    if legend_datasets is None or dataset in legend_datasets:
        pl.legend(loc=legend_loc, prop={'size': legend_size})
    dp.close()


def get_result_names(result_type):
    if result_type == "batch":
        return ['ifor', 'ifor_baseline', 'loda', 'ifor_noprior_unif', 'ifor_noprior_rand']
    elif result_type == "stream":
        # return ['ifor', 'ifor_baseline', 'ifor_stream_02', 'ifor_stream_q8b3', 'hstrees', 'hstrees_orig', 'rsforest_orig']
        return ['ifor', 'ifor_baseline', 'ifor_stream_KL', 'ifor_stream_q8b3', 'hstrees', 'hstrees_orig', 'rsforest_orig']
    elif result_type == "diversity":
        return ['ifor', 'ifor_q8b3']
    elif result_type == "stream_diff":
        # return ['ifor_stream_02', 'ifor_stream_no_weight_upd']  # , 'ifor_stream_no_tree_replace'
        return ['ifor_stream_KL', 'ifor_stream_02', 'ifor_stream_no_tree_replace', 'ifor']  # , 'ifor_stream_no_tree_replace'
    elif result_type == "stream_diff08":
        return ['ifor', 'ifor_stream_02', 'ifor_stream_08', 'ifor_stream_no_weight_upd', 'ifor_stream_no_weight_upd08', 'ifor_stream_KL']  # , 'ifor_stream_no_tree_replace'
    elif result_type == "compare_prior":
        return ['ifor', 'ifor_baseline', 'ifor_noprior_unif', 'ifor_noprior_rand']
    elif result_type == "ifor_top_vs_random":
        return ['ifor', 'ifor_q1b3', 'ifor_q8b3', 'ifor_q9b3', 'ifor_baseline', 'ifor_top_random']
        # return ['ifor', 'ifor_q1b3', 'ifor_q8b3', 'ifor_q9b3', 'ifor_q10b3', 'ifor_baseline']
    elif result_type == "ifor_vs_others":
        return [
            # 'ifor', 'ifor_stream_02', 'ifor_q8b3', 'ifor_stream_q8b3', 'ifor_stream_q1b3', 'ifor_baseline', # 'ifor_stream_no_upd', 'loda', 'loda_orig'
            'ifor', 'ifor_stream_q8b3', 'ifor_baseline',
            "loda", "loda_baseline",
            'hstrees_orig', 'hstrees_baseline', 'hstrees_q1b3',
        ]
    elif result_type == "unsupervised_only":
        return ["ifor_baseline", "loda_baseline", "hstrees_orig", "hstrees_baseline", "rsforest_orig"]
    elif result_type == "hstrees_only":
        return ["hstrees_orig", "hstrees_baseline", "hstrees_q1b3"]
    elif result_type == "hstrees_rsforest":
        return ["hstrees_orig", "hstrees", "rsforest_orig"]
    elif result_type == "ifor_loda":
        return ["ifor", "loda", "loda_baseline"]
    elif result_type == "fbonline":
        return ["ifor", "fbonline", "ifor_baseline"]
    else:
        raise ValueError("Invalid result_type: %s" % result_type)


def process_results(args, result_type="batch", plot=True, plot_sd=False, num_anoms=0,
                    legend_loc='lower right', legend_datasets=None, legend_size=14):
    result_names = get_result_names(result_type)

    cols = ["red", "green", "blue", "orange", "brown", "pink", "magenta", "black"]
    result_lists, result_map = get_result_defs(args)
    num_seen = 0
    # num_anoms = 0
    all_results = list()
    for i, r_name in enumerate(result_names):
        parent_folder = "./temp/aad/%s" % args.dataset
        if r_name == "ifor_stream_KL" or r_name == "ifor_stream_02":
            parent_folder = "./temp/aad/%s-new" % args.dataset
        rs = result_map[r_name]
        r_avg, r_sd, r_n = rs.get_results(parent_folder)
        logger.debug("[%s]\navg:\n%s\nsd:\n%s" % (rs.name, str(list(r_avg)), str(list(r_sd))))
        all_results.append((args.dataset, rs.display_name, r_avg, r_sd, r_n))
        num_seen = max(num_seen, len(r_avg))
        if num_anoms >= 0: num_anoms = max(num_anoms, rs.num_anoms)
    if plot:
        dir_create("./temp/aad_plots/%s" % result_type)
        plot_results(all_results, cols, "./temp/aad_plots/%s/num_seen-%s.pdf" % (result_type, args.dataset),
                     num_seen=num_seen, num_anoms=num_anoms, plot_sd=plot_sd,
                     legend_loc=legend_loc, legend_datasets=legend_datasets, legend_size=legend_size)
    return all_results, num_anoms


def plot_diversity_all(all_results, result_type, legend_loc='lower right', axis_fontsize=20):
    line_styles = ["-", "--"]
    dir_create("./temp/aad_plots/%s" % result_type)
    dp = DataPlotter(pdfpath="./temp/aad_plots/%s/diversity_num_seen.pdf" % result_type,
                     rows=1, cols=1)
    pl = dp.get_next_plot()
    plt.xlabel('# instances labeled', fontsize=axis_fontsize)
    plt.ylabel('% of total anomalies seen', fontsize=axis_fontsize)
    plt.xlim([0, 1500])
    plt.ylim([0, 100])
    for i, results_tmp in enumerate(all_results):
        results, num_anoms = results_tmp
        for j, rs in enumerate(results):
            dataset, display_name, r_avg, r_sd, r_n = rs
            dataset_name = dataset_configs[dataset][4]
            ln, = pl.plot(np.arange(len(r_avg)), r_avg * 100. / num_anoms,
                          line_styles[j],
                          color=dataset_colors[dataset], linewidth=1,
                          # label="%s (%s)" % (dataset, result_type)
                          label="%s (%s)" % (display_name, dataset_name)
                          )
    pl.legend(loc=legend_loc, prop={'size': 14})
    dp.close()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_aad_command_args(debug=True,
                                debug_args=["--dataset=abalone",
                                            "--debug",
                                            "--log_file=temp/plot_aad_results.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    dir_create("./temp/aad_plots")  # for logging and plots

    random.seed(42)
    rnd.seed(42)

    # result_type = "ifor_vs_others"
    # result_type = "stream"
    # result_type = "batch"
    # result_type = "compare_prior"
    # result_type = "unsupervised_only"
    # result_type = "hstrees_only"
    # result_type = "hstrees_rsforest"
    # result_type = "ifor_loda"
    result_type = "ifor_top_vs_random"
    # result_type = "diversity"
    # result_type = "stream_diff"
    # result_type = "stream_diff08"
    # result_type = "fbonline"

    plot_sd = False
    legend_loc = 'lower right'
    legend_datasets = None
    legend_size = 14
    num_anoms = 0
    if result_type == "batch":
        plot_sd = True
        legend_datasets = ["abalone"]
    elif result_type == "stream":
        legend_datasets = ["covtype"]
        legend_loc = 'upper left'
    elif result_type == "stream_diff":
        legend_loc = 'upper left'

    if result_type == "diversity":
        datasets = ['ann_thyroid_1v3', 'mammography', 'shuttle_1v23567']
    elif result_type == "stream_diff":
        legend_size = 20
        legend_datasets = ["electricity"]
        datasets = ['covtype', 'electricity', 'weather']
        # datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1', 'covtype', 'kddcup', 'mammography', 'shuttle_1v23567', 'electricity', 'weather']
    elif result_type == "stream_diff08":
        datasets = ['electricity', 'weather']
        legend_datasets = ["electricity"]
        legend_loc = 'upper left'
    elif result_type == "fbonline":
        datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1', # 'kddcup',
                    'shuttle_1v23567', 'mammography', # 'covtype',
                    'weather', "electricity"
                    ]
        plot_sd = True
        num_anoms = -1
    else:
        # datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1']  # , 'mammography']
        # datasets = ['abalone', 'yeast', 'ann_thyroid_1v3']
        datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1', 'kddcup',
                    'shuttle_1v23567', 'mammography', 'covtype',
                    'weather', "electricity"
                    ]
        # datasets = ['kddcup', 'shuttle_1v23567', 'covtype', 'mammography']
        # datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1', 'mammography']
        # datasets = ['weather']
    datasets = ['toy2']
    all_results = list()
    for dataset in datasets:
        args.dataset = dataset
        plot = (result_type != "diversity")
        all_results.append(process_results(args, result_type=result_type, plot=plot, plot_sd=plot_sd, num_anoms=num_anoms,
                                           legend_loc=legend_loc, legend_datasets=legend_datasets, legend_size=legend_size))

    if result_type == "diversity":
        plot_diversity_all(all_results, result_type)