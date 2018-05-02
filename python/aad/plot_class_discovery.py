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
pythonw -m aad.plot_class_discovery
"""


def plot_results(results, pdffile, num_seen=0, num_anoms=0):
    cols = ["red", "green", "blue", "orange", "brown", "pink", "black"]
    dp = DataPlotter(pdfpath=pdffile, rows=1, cols=1)
    pl = dp.get_next_plot()
    plt.xlabel('iter')
    plt.ylabel('num_seen')
    plt.xlim([0, num_seen])
    plt.ylim([0., 1.])
    for i, result in enumerate(results):
        num_found = result[1]
        logger.debug("label: %s" % result[0])
        pl.plot(np.arange(len(num_found)), num_found * 1./num_anoms, '--',
                color=cols[i], linewidth=2, label=result[0])
    pl.legend(loc='lower right', prop={'size': 14})
    dp.close()


def plot_class_discovery(results, pdffile, batch_size, n_batches):
    cols = ["red", "green", "blue", "orange", "brown", "pink", "black"]
    dp = DataPlotter(pdfpath=pdffile, rows=1, cols=1)
    pl = dp.get_next_plot()
    plt.xlabel('iter')
    plt.ylabel('num_seen')
    plt.xlim([0, n_batches])
    plt.ylim([0., batch_size])
    for i, result in enumerate(results):
        num_found = result[4]
        logger.debug("label: %s" % result[0])
        pl.plot(np.arange(len(num_found)), num_found, '--',
                color=cols[i], linewidth=2, label=result[0])
    pl.legend(loc='lower right', prop={'size': 14})
    dp.close()


def get_num_discovered_classes(queried, labels):
    discovered = np.zeros(shape=queried.shape, dtype=int)
    for i in range(queried.shape[0]):
        all_l = set()
        lbls = labels[queried[i, :]]
        for j, y in enumerate(lbls):
            all_l.add(y)
            discovered[i, j] = len(all_l)
    return discovered


def iter_by_window(x, window_indexes=None):
    if window_indexes is None:
        yield x
        return
    if len(x) > len(window_indexes):
        raise ValueError("len(x) (%d) must be smaller or equal to length of window indexes (%d)" % (len(x), len(window_indexes)))
    start = 0
    n = len(x)
    while start < n:
        id = window_indexes[start]
        end = start + 1
        while end < n and window_indexes[end] == id:
            end += 1
        yield x[start:end]
        start = end


def get_num_discovered_classes_per_batch(queried, labels, batch_size, window_indexes=None):
    discovered = list()
    for i in range(0, queried.shape[0]):
        n_classes_found = list()
        # lbls = labels[queried[i, :]]
        # We need to take the window indexes into account for the streaming setting...
        for lbls in iter_by_window(labels[queried[i, :]],
                                   window_indexes=None if window_indexes is None else window_indexes[i, :]):
            m = len(lbls)
            # logger.debug("m: %d" % m)
            for j in range(0, m, batch_size):
                q = lbls[j:min(m, j+batch_size)]
                s = set(q)
                n_classes_found.append(len(s))
        discovered.append(np.array(n_classes_found, dtype=int))
    discovered = np.vstack(discovered)
    return discovered


def process_results(args):
    stream_sig = ""
    stream_sig = "_stream"
    # result_names = ['hstrees_orig', 'hstrees_30', 'hstrees_50', 'hstrees_50_baseline']
    # result_names = ['hstrees_orig', 'hstrees_50', 'hstrees_50_baseline', 'ifor', 'ifor_baseline']
    result_names = ['ifor%s_q1b3' % stream_sig, 'ifor%s_q8b3' % stream_sig]
    result_lists, result_map = get_result_defs(args)
    num_seen = 0
    num_anoms = 0
    n_batches = 0
    all_results = list()
    for i, r_name in enumerate(result_names):
        if r_name == "ifor" or r_name == "ifor_baseline":
            parent_folder = "./temp/aad-IJCAI18/%s" % args.dataset
        else:
            parent_folder = "./temp/aad/%s" % args.dataset
        rs = result_map[r_name]
        r_avg, r_sd, r_n = rs.get_results(parent_folder)
        # logger.debug("[%s]\navg:\n%s\nsd:\n%s" % (rs.name, str(list(r_avg)), str(list(r_sd))))
        num_seen = max(num_seen, len(r_avg))
        num_anoms = max(num_anoms, rs.num_anoms)
        orig_labels = rs.get_original_labels()
        # logger.debug("original labels:\n%s" % str(list(orig_labels)))
        queried = rs.get_queried(parent_folder)
        # logger.debug("queried:\n%s" % str(list(queried)))
        window_indexes = rs.get_window_indexes(parent_folder)
        # logger.debug("window_indexes:\n%s" % str(list(window_indexes)))
        discovered = get_num_discovered_classes_per_batch(queried, orig_labels, batch_size=3, window_indexes=window_indexes)
        avg_classes = np.mean(discovered, axis=0)
        n_batches = max(n_batches, len(avg_classes))
        # logger.debug("[%s] discovered:\n%s" % (r_name, str(list(avg_classes))))
        all_results.append([rs.name, r_avg, r_sd, r_n, avg_classes])
    plot_results(all_results, "./temp/aad_plots/results_class_%s%s.pdf" % (args.dataset, stream_sig),
                 num_seen=num_seen, num_anoms=num_anoms)
    plot_class_discovery(all_results, "./temp/aad_plots/results_classes_per_batch_%s%s.pdf" % (args.dataset, stream_sig),
                         batch_size=3, n_batches=n_batches)

    test_mean = np.mean(all_results[1][4] - all_results[0][4])
    test_sd = np.std(all_results[1][4] - all_results[0][4]) / np.sqrt(len(all_results[1][4]))
    logger.debug("[%s] mean diff: %f (%f)" % (args.dataset, test_mean, test_sd))

    c_mean = np.cumsum(all_results[1][4] - all_results[0][4]) / np.arange(1, len(all_results[1][4])+1, dtype=np.float32)
    logger.debug("c_mean:\n%s" % str(list(c_mean)))
    dp = DataPlotter(pdfpath="./temp/aad_plots/results_diff_classes_%s%s.pdf" % (args.dataset, stream_sig), rows=1, cols=1)
    pl = dp.get_next_plot()
    plt.xlabel('number of batches from start (batch size=3)')
    plt.ylabel('avg. classes per batch')
    plt.xlim([0, len(c_mean)])
    plt.ylim([np.min(c_mean), np.max(c_mean)])
    pl.plot(np.arange(len(c_mean)), c_mean, '--',
            color="red", linewidth=1, label="diff in num classes")
    pl.axhline(0., color="black", linewidth=1)
    pl.legend(loc='lower right', prop={'size': 14})
    dp.close()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_aad_command_args(debug=True,
                                debug_args=["--dataset=abalone",
                                            "--debug",
                                            "--log_file=temp/aad_plots/plot_class_discovery.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    dir_create("./temp/aad_plots")  # for logging and plots

    random.seed(42)
    rnd.seed(42)

    datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1']  # , 'mammography']
    # datasets = ['abalone']
    # datasets = ['mammography']
    datasets = ['mammography', 'kddcup', 'shuttle_1v23567', 'covtype']
    for dataset in datasets:
        args.dataset = dataset
        process_results(args)