import os
import numpy as np

import logging

from common.utils import *
from aad.aad_globals import *
from aad.aad_support import *

from aad.data_stream import *
from common.gen_samples import read_anomaly_dataset
from common.data_plotter import *


"""
pythonw -m aad.test_concept_drift --debug --plot --log_file=temp/test_concept_drift.log --dataset=weather
"""


def get_iforest_model(x):
    model = AadForest(n_estimators=100,  # 100,
                      max_samples=256,
                      score_type=IFOR_SCORE_TYPE_NEG_PATH_LEN, random_state=42,
                      add_leaf_nodes_only=True,
                      max_depth=100,
                      ensemble_score=ENSEMBLE_SCORE_LINEAR,
                      detector_type=AAD_IFOREST, n_jobs=4,
                      tree_update_type=TREE_UPD_INCREMENTAL,
                      feature_partitions=None)
    model.fit(x)
    model.init_weights(init_type=INIT_UNIF)
    return model


def test_kl_data_drift():
    logger = logging.getLogger(__name__)

    args = get_command_args(debug=False, debug_args=["--debug",
                                                     "--plot",
                                                     "--log_file=temp/test_concept_drift.log"])
    configure_logger(args)

    np.random.seed(42)

    stream_window = 1024
    alpha = 0.05

    logger.debug("dataset: %s, stream_window: %d, alpha: %0.3f" % (args.dataset, stream_window, alpha))

    X_full, y_full = read_anomaly_dataset(args.dataset)
    stream = DataStream(X_full, y_full, IdServer(initial=0))
    training_set = stream.read_next_from_stream(stream_window)
    x, y, ids = training_set.x, training_set.y, training_set.ids
    model = get_iforest_model(x)

    all_kl_q_alpha = list()
    all_reference_kls = list()
    all_compare_kls = list()
    trees_replaced = list()

    # compute KL replacement threshold *without* p
    ref_kls, kl_q_alpha = model.get_KL_divergence_distribution(x, p=None, alpha=alpha)
    # now initialize reference p
    p = model.get_node_sample_distributions(x)

    max_kl = np.max(ref_kls)

    window = 0  # already read the first window
    while True:
        buffer = stream.read_next_from_stream(stream_window)
        if buffer is None:
            break
        window += 1
        x, y, ids = buffer.x, buffer.y, buffer.ids
        # logger.debug("#new: %d" % x.shape[0])

        model.add_samples(X=x)

        all_kl_q_alpha.append(kl_q_alpha)
        all_reference_kls.append(ref_kls)

        # compare KL-divergence of current data dist against reference dist p
        comp_kls, _ = model.get_KL_divergence_distribution(x, p=p)
        all_compare_kls.append(comp_kls)
        max_kl = max(max_kl, np.max(comp_kls))

        # find which trees exceed alpha-level threshold
        replace_trees_by_kl = model.get_trees_to_replace(comp_kls, kl_q_alpha)
        if replace_trees_by_kl is not None and len(replace_trees_by_kl) > 0:
            logger.debug("window %d: #replace_trees_by_kl: %d\n%s" %
                         (window, len(replace_trees_by_kl), str(list(replace_trees_by_kl))))
            trees_replaced.append(len(replace_trees_by_kl))
            model.update_model_from_stream_buffer(replace_trees=replace_trees_by_kl)
            # recompute KL replacement threshold *without* p
            ref_kls, kl_q_alpha = model.get_KL_divergence_distribution(x, p=None, alpha=alpha)
            max_kl = max(max_kl, np.max(ref_kls))
            # now recompute reference p
            p = model.get_node_sample_distributions(x)
        else:
            logger.debug("window %d: model not updated; replace_trees_by_kl: %s" %
                         (window, str(list(replace_trees_by_kl)) if replace_trees_by_kl is not None else None))
            trees_replaced.append(0)

    if args.plot:
        xlim = [0, window+1]
        ylim = [0, max_kl+3]
        dp = DataPlotter(pdfpath="./temp/test_concept_drift_%s.pdf" % args.dataset,
                         rows=1, cols=1)
        pl = dp.get_next_plot()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('window')
        plt.ylabel('KL-divergence')
        for i in range(window):
            ref_label = com_label = threshold_label = replaced_label = None
            ref_kls = all_reference_kls[i]
            com_kls = all_compare_kls[i]
            mkl = max(np.max(ref_kls), np.max(com_kls))
            x_coord = i+1
            replaced_y_coord = mkl+2
            if i == 0:
                ref_label = "ref. KL dist"
                com_label = "KL-dist w.r.t ref. dist"
                threshold_label = "%0.2f-alpha KL" % alpha
                replaced_label = "(.) - number of trees replaced"
                pl.scatter([x_coord], [replaced_y_coord], color="black", marker=".", s=0, label=replaced_label)
            pl.scatter(np.ones(len(ref_kls), dtype=np.float32)*x_coord, ref_kls,
                       color="orange", marker="*", s=8, label=ref_label)
            pl.scatter([x_coord], [all_kl_q_alpha[i]], color="red", marker="+", s=30, label=threshold_label)
            pl.scatter(np.ones(len(ref_kls), dtype=np.float32)*x_coord + 0.1, com_kls,
                       color="green", marker="*", s=8, label=com_label)
            pl.text(x_coord-0.2, replaced_y_coord, "(%d)"%trees_replaced[i], fontsize=8, label=replaced_label)
        pl.legend(loc='upper left', prop={'size': 6})
        dp.close()


if __name__ == "__main__":
    test_kl_data_drift()
