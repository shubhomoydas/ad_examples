import random
import logging
import numpy as np
import numpy.random as rnd
from copy import copy

from ..common.utils import configure_logger, dir_create
from ..common.metrics import fn_auc
from ..common.gen_samples import read_anomaly_dataset
from .aad_globals import (
    AadOpts, AAD_IFOREST, AAD_HSTREES, HST_LOG_SCORE_TYPE, INIT_UNIF, get_aad_command_args, detector_types
)
from .aad_support import get_aad_model

"""
To run:
pythonw -m ad_examples.aad.test_tree_detectors
"""


def compute_n_found(scores, labels, budget=-1):
    if budget < 0:
        budget = len(scores)
    queried = np.argsort(scores)
    n_found = np.cumsum(labels[queried[0:budget]])
    return n_found


def test_tree_detectors(args):
    opts_ = AadOpts(args)
    logger.debug(opts_.str_opts())

    rng = np.random.RandomState(args.randseed)

    x, y = read_anomaly_dataset(args.dataset)

    configs = [
        # {'detector_type': AAD_IFOREST, 'forest_score_type': IFOR_SCORE_TYPE_NEG_PATH_LEN},
        {'detector_type': AAD_HSTREES, 'forest_score_type': HST_LOG_SCORE_TYPE},
        # {'detector_type': AAD_RSFOREST, 'forest_score_type': RSF_SCORE_TYPE}
    ]
    opt_array = list()
    models = list()
    aucs = np.zeros(shape=(len(configs), 2), dtype=np.float32)
    n_found_list = list()

    for i, config in enumerate(configs):
        opts = copy(opts_)
        opt_array.append(opts)
        opts.detector_type = config['detector_type']
        opts.forest_score_type = config['forest_score_type']
        # opts.forest_max_depth = 9 if detector_type == AAD_IFOREST else 9
        opts.forest_max_depth = 9
        opts.forest_n_trees = 100 if opts.detector_type == AAD_IFOREST else 50

        model = get_aad_model(x, opts, rng)
        model.fit(x)
        model.init_weights(opts.init)
        models.append(model)

        auc = 0.
        baseline_auc = 0.
        if True:
            x_new = model.transform_to_ensemble_features(x, dense=False, norm_unit=opts.norm_unit)
            baseline_w = model.get_uniform_weights()
            scores = model.get_score(x_new, baseline_w)
            auc = fn_auc(np.hstack([np.transpose([y]), np.transpose([-scores])]))
            n_found_list.append(np.transpose([compute_n_found(scores, y)]))

        baseline_scores = -model.clf.decision_function(x)
        baseline_auc = fn_auc(np.hstack([np.transpose([y]), np.transpose([-baseline_scores])]))

        n_found_list.append(np.transpose([compute_n_found(baseline_scores, y)]))

        aucs[i, :] = [auc, baseline_auc]
        logger.debug("%s %s auc/baseline: %f/%f" %
                     (args.dataset, detector_types[opts.detector_type], auc, baseline_auc))

    logger.debug("aucs:\n%s" % str(aucs))


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_aad_command_args(debug=True,
                                debug_args=["--dataset=toy2",
                                            "--detector_type=%d" % AAD_IFOREST,
                                            "--init=%d" % INIT_UNIF,
                                            "--forest_add_leaf_nodes_only",
                                            "--debug",
                                            "--log_file=temp/aad/test_tree_detectors.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    dir_create("./temp/aad")  # for logging and plots

    random.seed(42)
    rnd.seed(42)

    # datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1']  # , 'mammography']
    # datasets = ['covtype', 'kddcup', 'shuttle_1v23567']
    datasets = ['abalone']

    for dataset in datasets:
        args.dataset = dataset
        test_tree_detectors(args)
