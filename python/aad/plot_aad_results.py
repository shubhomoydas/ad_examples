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


def plot_results(results, cols, pdffile, num_seen=0, num_anoms=0):
    dp = DataPlotter(pdfpath=pdffile, rows=1, cols=1)
    pl = dp.get_next_plot()
    plt.xlabel('# instances labeled')
    plt.ylabel('fraction of total anomalies seen')
    plt.xlim([0, num_seen])
    plt.ylim([0., 1.])
    for i, result in enumerate(results):
        num_found = result[1]
        logger.debug("label: %s" % result[0])
        pl.plot(np.arange(len(num_found)), num_found * 1./num_anoms, '-',
                color=cols[i], linewidth=1, label=result[0])
    pl.legend(loc='lower right', prop={'size': 8})
    dp.close()


def get_result_names(result_type):
    if result_type == "ifor_top_vs_random":
        return ['ifor', 'ifor_q1b3', 'ifor_q8b3', 'ifor_baseline', 'ifor_top_random']
    elif result_type == "ifor_vs_others":
        return [
            # 'ifor', 'ifor_stream', 'ifor_q8b3', 'ifor_stream_q8b3', 'ifor_stream_q1b3', 'ifor_baseline', # 'ifor_stream_no_upd', 'loda', 'loda_orig'
            'ifor', 'ifor_stream_q8b3', 'ifor_baseline',
            "loda", "loda_baseline",
            'hstrees_orig', 'hstrees_baseline', 'hstrees_q1b3',
        ]
    else:
        raise ValueError("Invalid result_type: %s" % result_type)


def process_results(args):
    # result_type = "ifor_vs_others"
    result_type = "ifor_top_vs_random"
    result_names = get_result_names(result_type)

    cols = ["red", "green", "blue", "orange", "brown", "pink", "magenta", "black"]
    result_lists, result_map = get_result_defs(args)
    num_seen = 0
    num_anoms = 0
    all_results = list()
    for i, r_name in enumerate(result_names):
        parent_folder = "./temp/aad/%s" % args.dataset
        rs = result_map[r_name]
        r_avg, r_sd, r_n = rs.get_results(parent_folder)
        logger.debug("[%s]\navg:\n%s\nsd:\n%s" % (rs.name, str(list(r_avg)), str(list(r_sd))))
        all_results.append([rs.name, r_avg, r_sd, r_n])
        num_seen = max(num_seen, len(r_avg))
        num_anoms = max(num_anoms, rs.num_anoms)
    dir_create("./temp/aad_plots/%s" % result_type)
    plot_results(all_results, cols, "./temp/aad_plots/%s/results_anoms_found_%s.pdf" % (result_type, args.dataset),
                 num_seen=num_seen, num_anoms=num_anoms)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_aad_command_args(debug=True,
                                debug_args=["--dataset=abalone",
                                            "--debug",
                                            "--log_file=temp/aad_plots/plot_aad_results.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    dir_create("./temp/aad_plots")  # for logging and plots

    random.seed(42)
    rnd.seed(42)

    # datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1']  # , 'mammography']
    # datasets = ['abalone', 'yeast', 'ann_thyroid_1v3']
    datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1', 'kddcup', 'shuttle_1v23567', 'mammography', 'covtype']
    # datasets = ['kddcup', 'shuttle_1v23567', 'covtype', 'mammography']
    # datasets = ['mammography']
    datasets = ['toy2']
    for dataset in datasets:
        args.dataset = dataset
        process_results(args)