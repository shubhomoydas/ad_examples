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


class ResultDefs(object):
    def __init__(self, name=None, display_name=None, dataset=None, num_anoms=None, budget=None,
                 filename=None, subdir=None, queried=None, window_indexes=None):
        self.name = name
        self.display_name = name if display_name is None else display_name
        self.dataset = dataset
        self.num_anoms = num_anoms
        self.budget = budget
        self.filename = filename
        self.subdir = subdir
        self.queried = queried
        self.window_indexes = window_indexes

    def get_complete_filepath(self, filename, parentdir=None):
        if self.subdir is not None:
            file = os.path.join(self.subdir, filename)
        else:
            file = filename
        if parentdir is not None:
            file = os.path.join(parentdir, file)
        return file

    def get_per_run_results(self, parentdir=None):
        file = self.get_complete_filepath(self.filename, parentdir)
        header = None
        start_col = 1
        if self.name == "fbonline":
            header = 0
            start_col = 1  # there is no fileidx
        resultsdf = pd.read_csv(file, header=header, sep=",")
        results = np.array(resultsdf.values, dtype=np.float32)
        return results[:, start_col:]

    def get_results(self, parentdir=None):
        results = self.get_per_run_results(parentdir=parentdir)
        r_avg = np.mean(results, axis=0)
        r_sd = np.std(results, axis=0)
        return r_avg, r_sd, results.shape[0]

    def get_queried(self, parentdir=None):
        file = self.get_complete_filepath(self.queried, parentdir)
        querieddf = pd.read_csv(file, header=None, sep=",")
        # the queried indexes are 1-indexed (for compatibility with R)
        # and hence we will subtract 1 from them.
        queried = np.array(querieddf.values, dtype=int)
        queried = queried[:, 2:] - 1
        return queried

    def get_window_indexes(self, parentdir=None):
        if self.window_indexes is None:
            return None
        file = self.get_complete_filepath(self.window_indexes, parentdir)
        windowdf = pd.read_csv(file, header=None, sep=",")
        # the window indexes are 1-indexed (for compatibility with R)
        # and hence we will subtract 1 from them.
        window_indexes = np.array(windowdf.values, dtype=int)
        window_indexes = window_indexes[:, 2:] - 1
        return window_indexes

    def get_original_labels(self, datasetdir="../datasets/anomaly"):
        original_label_file = os.path.join(datasetdir, self.dataset, "fullsamples",
                                           "%s_1_orig_labels.csv" % self.dataset)
        labelsdf = pd.read_csv(original_label_file, header=0, sep=",")
        return labelsdf.iloc[:, 1]


dataset_configs = {
    # 'dataset': [budget, num_anoms, window_size, max_number_of_windows, n_trees, display_name]
    'abalone': [300, 29, 512, 30, "Abalone"],
    'ann_thyroid_1v3': [300, 73, 512, 30, "ANN-Thyroid-1v3"],
    'cardiotocography_1': [300, 45, 512, 30, "Cardiotocography"],
    'yeast': [300, 55, 512, 30, "Yeast"],
    'mammography': [1500, 260, 4096, 30, "Mammography"],
    'covtype': [3000, 2747, 4096, 1000, "Covtype"],
    'kddcup': [3000, 2416, 4096, 30, "KDD-Cup-99"],
    'shuttle_1v23567': [1500, 867, 4096, 30, "Shuttle"],
    'weather': [1000, 656, 1024, 30, "Weather"],
    'electricity': [1500, 1372, 1024, 30, "Electricity"],
    'toy2': [45, 35, 512, 30, "Toy-2"]
}

dataset_colors = {"abalone": "red", "yeast": "green", "ann_thyroid_1v3": "blue",
                      "cardiotocography_1": "orange", "covtype": "magenta",
                      "mammography": "pink", "kddcup": "grey", "shuttle_1v23567": "brown",
                      "weather": "black", "electricity": "cyan",
                      "toy2": "black"}

def get_result_defs(args):
    budget = dataset_configs[args.dataset][0]
    num_anoms = dataset_configs[args.dataset][1]
    window_size = dataset_configs[args.dataset][2]
    max_windows = dataset_configs[args.dataset][3]

    hst_orig_f = "{dataset}-hstrees_tau_instance-trees25_samples256_nscore9_leaf-top-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-orig_num_seen.csv"
    hst_orig_d = "hstrees_trees25_samples256_i11_q1_bd{budget}_nscore9_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma50_mn50_d15"

    hst_no_upd_f = "{dataset}-hstrees_tau_instance-trees{trees}_samples256_nscore5_leaf-topb3-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0_no_upd-{type}.csv"
    hst_no_upd_d = "hstrees_trees{trees}_samples256_i11_q1b3_bd{budget}_nscore5_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma50_mn50_d8_no_upd"

    hst_f = "{dataset}-hstrees_tau_instance-trees{trees}_samples256_nscore5_leaf-top-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-{type}.csv"
    hst_d = "hstrees_trees{trees}_samples256_i11_q1_bd{budget}_nscore5_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma50_mn50_d8"

    hst_q1b3_f = "{dataset}-hstrees_tau_instance-trees{trees}_samples256_nscore5_leaf-topb3-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-{type}.csv"
    hst_q1b3_d = "hstrees_trees{trees}_samples256_i11_q1b3_bd{budget}_nscore5_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma50_mn50_d8"

    hst_stream_f = "{dataset}-hstrees_incr_n100_r0_2_tau_instance-trees{trees}_samples256_nscore5_leaf-top-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-sw{window_size}_asuTrue_mw{max_windows}f2_20_anomalous_tillbudget-{type}.csv"
    hst_stream_d = "hstrees_incr_trees{trees}_samples256_i11_q1_bd{budget}_nscore5_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma50_mn50_d8_stream{window_size}asu_mw{max_windows}f2_20_ret1_tillbudget"

    hst_stream_no_upd_f = "{dataset}-hstrees_n100_r0_2_tau_instance-trees{trees}_samples256_nscore5_leaf-top-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-sw{window_size}_asuTrue_no_upd_mw{max_windows}f2_20_anomalous_tillbudget-{type}.csv"
    hst_stream_no_upd_d = "hstrees_trees{trees}_samples256_i11_q1_bd{budget}_nscore5_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma50_mn50_d8_stream{window_size}asu_no_upd_mw{max_windows}f2_20_ret1_tillbudget"

    hst_stream_incr_f = "{dataset}-hstrees_incr_tau_instance-trees{trees}_samples256_nscore5_leaf-top-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-sw{window_size}_asuTrue_mw{max_windows}f2_20_anomalous_tillbudget-{type}.csv"
    hst_stream_incr_d = "hstrees_incr_trees50_samples256_i11_q1_bd{budget}_nscore5_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma50_mn50_d8_stream{window_size}asu_mw{max_windows}f2_20_ret1_tillbudget"

    hst_stream_incr_no_upd_f = "{dataset}-hstrees_incr_tau_instance-trees{trees}_samples256_nscore5_leaf-top-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-sw{window_size}_asuTrue_no_upd_mw{max_windows}f2_20_anomalous_tillbudget-{type}.csv"
    hst_stream_incr_no_upd_d = "hstrees_incr_trees50_samples256_i11_q1_bd{budget}_nscore5_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma50_mn50_d8_stream{window_size}asu_no_upd_mw{max_windows}f2_20_ret1_tillbudget"

    hst_q8_f = "{dataset}-hstrees_tau_instance-trees{trees}_samples256_nscore5_leaf-custom-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-{type}.csv"
    hst_q8_d = "hstrees_trees{trees}_samples256_i11_q8n10b3_bd{budget}_nscore5_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma50_mn50_d8"

    ifor_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-top-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-{type}.csv"
    ifor_d = "if_aad_trees{trees}_samples256_i7_q1_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_d100_norm"

    ifor_q8b3_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-custom-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-{type}.csv"
    ifor_q8b3_d = "if_aad_trees{trees}_samples256_i7_q8n10b3_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_d100_norm"

    ifor_q9b3_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-euclidean-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-{type}.csv"
    ifor_q9b3_d = "if_aad_trees{trees}_samples256_i7_q9n10b3_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_d100_norm"

    ifor_q10b3_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-diverse_euc-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-{type}.csv"
    ifor_q10b3_d = "if_aad_trees{trees}_samples256_i7_q10n10b3_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_d100_norm"

    ifor_q2b3_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-toprandomb3-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-{type}.csv"
    ifor_q2b3_d = "if_aad_trees{trees}_samples256_i7_q2n10b3_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_d100_norm"

    ifor_stream_02_f = "{dataset}-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-sw{window_size}_asuTrue_mw{max_windows}f2_20_anomalous_u10_tillbudget-{type}.csv"
    ifor_stream_02_d = "if_aad_trees{trees}_samples256_i7_q1_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream{window_size}asu_mw{max_windows}f2_20_ret1_tillbudget_u10_norm"

    ifor_stream_08_f = "{dataset}-iforest_tau_instance-trees100_samples256_nscore4_leaf-topb3-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-sw{window_size}_asuTrue_mw{max_windows}f2_20_anomalous_f0_80_tillbudget-{type}.csv"
    ifor_stream_08_d = "if_aad_trees{trees}_samples256_i7_q1b3_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream{window_size}asu_mw{max_windows}f2_20_ret1_tillbudget_f0.8_norm"

    ifor_stream_KL_f = "{dataset}-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-sw{window_size}_asuTrue_KL0_05_mw{max_windows}f2_20_anomalous_u10_tillbudget-{type}.csv"
    ifor_stream_KL_d = "if_aad_trees{trees}_samples256_i7_q1_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream{window_size}asu_KL0.05_mw{max_windows}f2_20_ret1_tillbudget_u10_norm"

    ifor_stream_no_weight_upd_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-topb3-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0_no_upd-norm-sw{window_size}_asuTrue_no_upd_mw{max_windows}f2_20_anomalous_tillbudget-{type}.csv"
    ifor_stream_no_weight_upd_d = "if_aad_trees{trees}_samples256_i7_q1b3_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream{window_size}asu_mw{max_windows}f2_20_ret1_tillbudget_no_upd_norm"

    ifor_stream_no_weight_upd_08_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-topb3-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0_no_upd-norm-sw{window_size}_asuTrue_no_upd_mw{max_windows}f2_20_anomalous_f0_80_tillbudget-{type}.csv"
    ifor_stream_no_weight_upd_08_d = "if_aad_trees{trees}_samples256_i7_q1b3_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream{window_size}asu_mw{max_windows}f2_20_ret1_tillbudget_f0.8_no_upd_norm"

    ifor_stream_q8b3_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-custom-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-sw{window_size}_asuTrue_mw{max_windows}f2_20_anomalous_tillbudget-{type}.csv"
    ifor_stream_q8b3_d = "if_aad_trees{trees}_samples256_i7_q8n10b3_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream{window_size}asu_mw{max_windows}f2_20_ret1_tillbudget_norm"

    ifor_q1b3_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-topb3-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-{type}.csv"
    ifor_q1b3_d = "if_aad_trees{trees}_samples256_i7_q1b3_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_d100_norm"

    ifor_stream_q1b3_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-topb3-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-sw{window_size}_asuTrue_mw{max_windows}f2_20_anomalous_tillbudget-{type}.csv"
    ifor_stream_q1b3_d = "if_aad_trees{trees}_samples256_i7_q1b3_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream{window_size}asu_mw{max_windows}f2_20_ret1_tillbudget_norm"

    loda_f = "{dataset}-loda_k300t500-top-unifprior-init_uniform-Ca100-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-{type}.csv"
    loda_d = "loda_k300t500_i13_q1_bd{budget}_tau0.03_xtau_s0.5_init1_ca100_cx0.001_ma100_mn100"

    loda_orig_f = "{dataset}-loda-top-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-{type}.csv"
    loda_orig_d = "loda_i13_q1_bd{budget}_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_orig"

    rsf_orig_f = "{dataset}-rsfor_tau_instance-trees30_samples256_nscore9_leaf-top-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-orig_num_seen.csv"
    rsf_orig_d = "rsforest_trees30_samples256_i12_q1_bd{budget}_nscore9_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma50_mn50_d15"

    ifor_noprior_unif_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-top-noprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-{type}.csv"
    ifor_noprior_unif_d = "if_aad_trees{trees}_samples256_i7_q1_bd{budget}_nscore4_leaf_tau0.03_xtau_noprior_init1_ca1_cx1_ma1000_mn1000_d100_norm"

    ifor_noprior_rand_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-top-noprior-init_random-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-{type}.csv"
    ifor_noprior_rand_d = "if_aad_trees{trees}_samples256_i7_q1_bd{budget}_nscore4_leaf_tau0.03_xtau_noprior_init2_ca1_cx1_ma1000_mn1000_d100_norm"

    ifor_stream_no_tree_replace_f = "{dataset}-iforest_tau_instance-trees{trees}_samples256_nscore4_leaf-top-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd{budget}-tau0_030-topK0-norm-sw{window_size}_asuFalse_KL0_05_mw{max_windows}f2_20_anomalous_u10_tillbudget-{type}.csv"
    ifor_stream_no_tree_replace_d = "if_aad_trees{trees}_samples256_i7_q1_bd{budget}_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream{window_size}_KL0.05_mw{max_windows}f2_20_ret1_tillbudget_u10_norm"

    fbonline_f = "{dataset}_1_summary_feed_{budget}_losstype_linear_updatetype_online_ngrad_1_reg_0_lrate_1_pwgt_0_inwgt_0_rtype_L2.csv"
    fbonline_d = "fbonline"

    result_lists = [
        ResultDefs(name="rsforest_orig", display_name="RSF-Batch", dataset=args.dataset, num_anoms=num_anoms,
                   filename=rsf_orig_f.format(dataset=args.dataset, budget=budget),
                   subdir=rsf_orig_d.format(budget=budget)),
        ResultDefs(name="hstrees_orig", display_name="HST-Batch", dataset=args.dataset, num_anoms=num_anoms,
                   filename=hst_orig_f.format(dataset=args.dataset, budget=budget),
                   subdir=hst_orig_d.format(budget=budget)),
        ResultDefs(name="hstrees_baseline", dataset=args.dataset, num_anoms=num_anoms,
                   filename=hst_q1b3_f.format(dataset=args.dataset, budget=budget, trees=50, type="baseline"),
                   subdir=hst_q1b3_d.format(dataset=args.dataset, budget=budget, trees=50)),
        ResultDefs(name="hstrees", display_name="HST-Batch (Feedback)", dataset=args.dataset, num_anoms=num_anoms,
                   filename=hst_q1b3_f.format(dataset=args.dataset, budget=budget, trees=50, type="num_seen"),
                   subdir=hst_q1b3_d.format(dataset=args.dataset, budget=budget, trees=50)),
        ResultDefs(name="hstrees_q1b3", dataset=args.dataset, num_anoms=num_anoms,
                   filename=hst_q1b3_f.format(dataset=args.dataset, budget=budget, trees=50, type="num_seen"),
                   subdir=hst_q1b3_d.format(dataset=args.dataset, budget=budget, trees=50)),
        ResultDefs(name="hstrees_q8_50", dataset=args.dataset, num_anoms=num_anoms,
                   filename=hst_q8_f.format(dataset=args.dataset, budget=budget, trees=50, type="num_seen"),
                   subdir=hst_q8_d.format(dataset=args.dataset, budget=budget, trees=50)),
        ResultDefs(name="loda", display_name="AAD-Batch", dataset=args.dataset, num_anoms=num_anoms,
                   filename=loda_f.format(dataset=args.dataset, budget=budget, type="num_seen"),
                   subdir=loda_d.format(dataset=args.dataset, budget=budget),
                   queried=loda_f.format(dataset=args.dataset, budget=budget, type="queried")),
        ResultDefs(name="loda_baseline", dataset=args.dataset, num_anoms=num_anoms,
                   filename=loda_f.format(dataset=args.dataset, budget=budget, type="baseline"),
                   subdir=loda_d.format(dataset=args.dataset, budget=budget),
                   queried=loda_f.format(dataset=args.dataset, budget=budget, type="queried")),
        ResultDefs(name="ifor", display_name="BAL", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen"),
                   subdir=ifor_d.format(dataset=args.dataset, budget=budget, trees=100),
                   queried=ifor_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried")),
        ResultDefs(name="ifor_baseline", display_name="Unsupervised Baseline", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_f.format(dataset=args.dataset, budget=budget, trees=100, type="baseline"),
                   subdir=ifor_d.format(dataset=args.dataset, budget=budget, trees=100),
                   queried=ifor_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried-baseline")),
        ResultDefs(name="ifor_noprior_unif", display_name="BAL (No Prior - Unif)", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_noprior_unif_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen"),
                   subdir=ifor_noprior_unif_d.format(dataset=args.dataset, budget=budget, trees=100),
                   queried=ifor_noprior_unif_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried")),
        ResultDefs(name="ifor_noprior_rand", display_name="BAL (No Prior - Rand)", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_noprior_rand_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen"),
                   subdir=ifor_noprior_rand_d.format(dataset=args.dataset, budget=budget, trees=100),
                   queried=ifor_noprior_rand_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried")),
        ResultDefs(name="ifor_top_random", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_q2b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen"),
                   subdir=ifor_q2b3_d.format(dataset=args.dataset, budget=budget, trees=100),
                   queried=ifor_q2b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried")),
        ResultDefs(name="ifor_stream_02", display_name="SAL (Replace 20% Trees)", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_stream_02_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_stream_02_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_stream_02_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="ifor_stream_08", display_name="SAL (Replace 80% Trees)", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_stream_08_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_stream_08_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_stream_08_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="ifor_stream_KL", display_name="SAL (KL Adaptive)", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_stream_KL_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_stream_KL_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_stream_KL_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="ifor_stream_no_weight_upd", display_name="SAL (No Feedback, Replace 20% Trees)", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_stream_no_weight_upd_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_stream_no_weight_upd_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_stream_no_weight_upd_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="ifor_stream_no_weight_upd08", display_name="SAL (No Feedback, Replace 80% Trees)", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_stream_no_weight_upd_08_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_stream_no_weight_upd_08_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_stream_no_weight_upd_08_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="ifor_stream_no_tree_replace", display_name="SAL (No Tree Replace)", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_stream_no_tree_replace_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_stream_no_tree_replace_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_stream_no_tree_replace_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="ifor_q8b3", display_name="BAL-D", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_q8b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_q8b3_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_q8b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="ifor_q9b3", display_name="BAL-E", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_q9b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_q9b3_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_q9b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="ifor_q10b3", display_name="BAL-DE", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_q10b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_q10b3_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_q10b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="ifor_stream_q8b3", display_name="SAL-D", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_stream_q8b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_stream_q8b3_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_stream_q8b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows),
                   window_indexes=ifor_stream_q8b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="window",
                                                            window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="ifor_q1b3", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_q1b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen"),
                   subdir=ifor_q1b3_d.format(dataset=args.dataset, budget=budget, trees=100),
                   queried=ifor_q1b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried")),
        ResultDefs(name="ifor_stream_q1b3", dataset=args.dataset, num_anoms=num_anoms,
                   filename=ifor_stream_q1b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=ifor_stream_q1b3_d.format(dataset=args.dataset, budget=budget, trees=100, window_size=window_size, max_windows=max_windows),
                   queried=ifor_stream_q1b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="queried", window_size=window_size, max_windows=max_windows),
                   window_indexes=ifor_stream_q1b3_f.format(dataset=args.dataset, budget=budget, trees=100, type="window",
                                                            window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="hstrees_stream", dataset=args.dataset, num_anoms=num_anoms,
                   filename=hst_stream_f.format(dataset=args.dataset, budget=budget, trees=50, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=hst_stream_d.format(dataset=args.dataset, budget=budget, trees=50, window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="hstrees_stream_no_upd", dataset=args.dataset, num_anoms=num_anoms,
                   filename=hst_stream_no_upd_f.format(dataset=args.dataset, budget=budget, trees=50, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=hst_stream_no_upd_d.format(dataset=args.dataset, budget=budget, trees=50, window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="hstrees_stream_incr", dataset=args.dataset, num_anoms=num_anoms,
                   filename=hst_stream_incr_f.format(dataset=args.dataset, budget=budget, trees=50, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=hst_stream_incr_d.format(dataset=args.dataset, budget=budget, trees=50, window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="hstrees_stream_incr_no_upd", dataset=args.dataset, num_anoms=num_anoms,
                   filename=hst_stream_incr_no_upd_f.format(dataset=args.dataset, budget=budget, trees=50, type="num_seen", window_size=window_size, max_windows=max_windows),
                   subdir=hst_stream_incr_no_upd_d.format(dataset=args.dataset, budget=budget, trees=50, window_size=window_size, max_windows=max_windows)),
        ResultDefs(name="fbonline", display_name="Feedback Guided Online", dataset=args.dataset, num_anoms=num_anoms,
                   filename=fbonline_f.format(dataset=args.dataset, budget=budget, trees=100,
                                              type="num_seen", window_size=window_size,
                                              max_windows=max_windows),
                   subdir=fbonline_d.format(dataset=args.dataset, budget=budget, trees=100,
                                            window_size=window_size, max_windows=max_windows))
    ]
    result_map = {}
    for result_list in result_lists:
        result_map[result_list.name] = result_list
    return result_lists, result_map


