import logging
import numpy as np
import matplotlib.pyplot as plt

from ..common.utils import configure_logger, read_data_as_matrix
from ..common.data_plotter import DataPlotter

from .aad_globals import (
    AAD_IFOREST, AAD_HSTREES, AAD_RSFOREST,
    IFOR_SCORE_TYPE_NEG_PATH_LEN, HST_LOG_SCORE_TYPE, RSF_SCORE_TYPE,
    INIT_UNIF, ENSEMBLE_SCORE_LINEAR, detector_types,
    get_aad_command_args, AadOpts
)
from .forest_aad_detector import is_forest_detector
from .aad_support import get_aad_model

logger = logging.getLogger(__name__)


def plot_value_hist(vals, dp):
    nbins = 100
    mn = np.min(vals)
    mx = np.max(vals)
    # bins = np.arange(start=mn, stop=mx, step=(mx-mn)/nbins)
    pl = dp.get_next_plot()
    n, bins = np.histogram(vals, bins=nbins, density=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, n, align='center', width=width, facecolor='green', alpha=0.90)


def plot_labeled_value_hist(vals, labels, dp):
    nom_v  = vals[np.where(labels==0)[0]]
    anom_v = vals[np.where(labels==1)[0]]
    nbins = 50
    bins = np.arange(start=np.min(vals), stop=np.max(vals), step=(np.max(vals)-np.min(vals))/nbins)
    pl = dp.get_next_plot()
    logger.debug("\n%s" % str(list(nom_v)))
    n1, bins1 = np.histogram(nom_v, bins=bins, normed=True)
    n2, bins2 = np.histogram(anom_v, bins=bins, normed=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, n1, align='center', width=width, facecolor='green', alpha=0.50)
    plt.bar(center, n2, align='center', width=width, facecolor='red', alpha=0.50)


def get_debug_args(dataset, detector_type):
    return ["--dataset=%s" % dataset,
            "--datafile=../datasets/anomaly/%s/fullsamples/%s_1.csv" % (dataset, dataset),
            "--startcol=2", "--labelindex=1", "--header",
            "--resultsdir=./temp", "--randseed=42",
            "--detector_type=%d" % detector_type,
            "--forest_score_type=%d" %
            (IFOR_SCORE_TYPE_NEG_PATH_LEN if detector_type == AAD_IFOREST
             else HST_LOG_SCORE_TYPE if detector_type == AAD_HSTREES
             else RSF_SCORE_TYPE if detector_type == AAD_RSFOREST else 0),
            "--init=%d" % INIT_UNIF,
            # "--norm_unit",
            "--forest_n_trees=100", "--forest_n_samples=256",
            "--forest_max_depth=%d" % (100 if detector_type == AAD_IFOREST else 7),
            "--forest_add_leaf_nodes_only",
            "--ensemble_score=%d" % ENSEMBLE_SCORE_LINEAR,
            "--log_file=./temp/node_values.log",
            "--debug"]


def test_node_values():

    dataset = "abalone"
    dataset = "cardiotocography_1"
    detector_type = AAD_IFOREST
    # detector_type = AAD_HSTREES
    # detector_type = AAD_RSFOREST
    args = get_aad_command_args(debug=True, debug_args=get_debug_args(dataset, detector_type))
    # print "log file: %s" % args.log_file
    configure_logger(args)

    opts = AadOpts(args)
    logger.debug(opts.str_opts())

    if opts.streaming:
        raise ValueError("Streaming not supported")

    np.random.seed(opts.randseed)

    logger.debug("loading file: %s" % opts.datafile)
    logger.debug("results dir: %s" % opts.resultsdir)
    logger.debug("forest_type: %s" % detector_types[opts.detector_type])

    X_train, labels = read_data_as_matrix(opts)

    rng = np.random.RandomState(args.randseed)

    # fit the model
    model = get_aad_model(X_train, opts, rng)
    model.fit(X_train)
    model.init_weights(opts.init)

    if False:
        if is_forest_detector(model.detector_type):
            logger.debug("total #nodes: %d" % (len(model.all_regions)))

        pdfpath = "./temp/node_values_%s_%s%s.pdf" % \
                  (args.dataset, detector_types[opts.detector_type],
                   "_norm" if opts.norm_unit else "")

        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        plot_value_hist(model.d, dp)
        dp.close()

    if True:
        X_transformed = model.transform_to_ensemble_features(X_train, norm_unit=opts.norm_unit)
        scores = model.get_score(X_transformed)
        pdfpath = "./temp/projection_values_%s_%s%s.pdf" % \
                  (args.dataset, detector_types[opts.detector_type],
                   "_norm" if opts.norm_unit else "")

        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        plot_labeled_value_hist(scores, labels, dp)
        dp.close()


if __name__ == "__main__":
    test_node_values()