from common.data_plotter import *
from common.gen_samples import *

from aad.aad_support import *

"""
pythonw -m aad.anomaly_vs_classifier
"""

logger = logging.getLogger(__name__)


def get_debug_args(detector_type=AAD_IFOREST):
    return ["--resultsdir=./temp", "--randseed=42",
            "--reruns=1",
            "--detector_type=%d" % detector_type,
            "--forest_score_type=%d" %
            (IFOR_SCORE_TYPE_NEG_PATH_LEN if detector_type == AAD_IFOREST
             else HST_LOG_SCORE_TYPE if detector_type == AAD_HSTREES
             else RSF_SCORE_TYPE if detector_type == AAD_RSFOREST else 0),
            "--init=%d" % INIT_UNIF,
            "--withprior", "--unifprior",  # use an (adaptive) uniform prior
            # ensure that scores of labeled anomalies are higher than tau-ranked instance,
            # while scores of nominals are lower
            "--constrainttype=%d" % AAD_CONSTRAINT_TAU_INSTANCE,
            # normalize is NOT required in general.
            # Especially, NEVER normalize if detector_type is anything other than AAD_IFOREST
            # "--norm_unit",
            "--forest_n_trees=100", "--forest_n_samples=256",
            "--forest_max_depth=%d" % (100 if detector_type == AAD_IFOREST else 7),
            "--forest_add_leaf_nodes_only",
            "--ensemble_score=%d" % ENSEMBLE_SCORE_LINEAR,
            "--resultsdir=./temp",
            "--log_file=./temp/anomaly_vs_classifier.log",
            "--debug"]


def get_auc(model, x, y, x_transformed=None):
    scores = model.get_score(x=x_transformed)
    auc = fn_auc(cbind(y, -scores))
    return auc


def train_anomaly_detector(x, y, opts, test_points):
    rng = np.random.RandomState(opts.randseed)

    # fit the model
    model = get_aad_model(x, opts, rng)
    model.fit(x)
    model.init_weights(INIT_UNIF)

    # train model with labeled examples
    x_transformed = model.transform_to_ensemble_features(x, dense=False, norm_unit=opts.norm_unit)
    ha = np.where(y == 1)[0]
    hn = np.where(y == 0)[0]
    # hn = np.zeros(0, dtype=int)

    # we know the true anomaly fraction from the fully labeled data
    opts.tau = len(ha) * 1.0 / len(y)

    auc = get_auc(model, x=x, y=y, x_transformed=x_transformed)
    logger.debug("AUC[0]: %f" % (auc))
    best_i = 0
    best_auc = auc
    best_w = model.w
    for i in range(opts.n_pretrain):
        model.update_weights(x_transformed, y, ha, hn, opts)
        auc = get_auc(model, x=x, y=y, x_transformed=x_transformed)
        logger.debug("AUC[%d]: %f" % (i + 1, auc))
        if best_auc <= auc:
            best_auc = auc
            best_w = np.copy(model.w)
            best_i = i + 1
    logger.debug("best_i: %d, best_auc: %f" % (best_i, best_auc))
    model.w = best_w

    pdfpath = "%s/%s.pdf" % (opts.resultsdir, "avc_train_anomaly")
    logger.debug("Plotting aad contours to %s" % (pdfpath))
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()

    xx, yy = np.meshgrid(np.linspace(-4, 8, 50), np.linspace(-4, 8, 50))
    x_test = np.c_[xx.ravel(), yy.ravel()]
    x_test_transformed = model.transform_to_ensemble_features(x_test, dense=False, norm_unit=opts.norm_unit)
    Z = model.get_score(x_test_transformed)
    Z = Z.reshape(xx.shape)
    pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))
    dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)

    # sidebar coordinates and dimensions for showing rank locations of true anomalies
    dash_xy = (-4.0, -2.0)  # bottom-left (x,y) coordinates
    dash_wh = (0.4, 8)  # width, height

    # plot the sidebar
    anom_scores = model.get_score(x_transformed)
    anom_order = np.argsort(-anom_scores)
    anom_idxs = np.where(y[anom_order] == 1)[0]
    dash = 1 - (anom_idxs * 1.0 / x.shape[0])
    plot_sidebar(dash, dash_xy, dash_wh, pl)

    dp.plot_points(test_points, pl, marker='x', defaultcol='green', s=50, linewidths=2.)
    dp.plot_points(test_points, pl, marker='o', edgecolor='green', defaultcol='green', s=60, linewidths=2.)

    dp.close()


def train_classifier(x, y, opts, test_points):
    classifier = RFClassifier.fit(x, y, n_estimators=100, max_depth=None)

    xx, yy = np.meshgrid(np.linspace(-4, 8, 50), np.linspace(-4, 8, 50))
    x_test = np.c_[xx.ravel(), yy.ravel()]
    probs_grid = classifier.predict_prob_for_class(x_test, 1)
    Z = probs_grid.reshape(xx.shape)

    pdfpath = "%s/%s.pdf" % (opts.resultsdir, "avc_train_classifier")
    logger.debug("Plotting classifier contours to %s" % (pdfpath))
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))
    dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)

    # sidebar coordinates and dimensions for showing rank locations of true anomalies
    dash_xy = (-4.0, -2.0)  # bottom-left (x,y) coordinates
    dash_wh = (0.4, 8)  # width, height

    # plot the sidebar
    anom_scores = classifier.predict_prob_for_class(x, 1)
    anom_order = np.argsort(-anom_scores)
    anom_idxs = np.where(y[anom_order] == 1)[0]
    dash = 1 - (anom_idxs * 1.0 / x.shape[0])
    plot_sidebar(dash, dash_xy, dash_wh, pl)

    dp.plot_points(test_points, pl, marker='x', defaultcol='green', s=50, linewidths=2.)
    dp.plot_points(test_points, pl, marker='o', edgecolor='green', defaultcol='green', s=60, linewidths=2.)

    dp.close()


def plot_dataset(x, y, opts, test_points):
    pdfpath = "%s/%s.pdf" % (opts.resultsdir, "avc_dataset")
    logger.debug("Plotting dataset to %s" % (pdfpath))
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    nomls = np.where(y == 0)[0]
    anoms = np.where(y == 1)[0]
    pl.scatter(x[nomls, 0], x[nomls, 1], marker='x', s=25, facecolors='blue', edgecolors="blue", label="Nominal")
    pl.scatter(x[anoms, 0], x[anoms, 1], marker='x', s=25, facecolors='red', edgecolors="red", label="Anomaly")
    pl.scatter(test_points[:, 0], test_points[:, 1], marker='x', s=50, facecolors='green', edgecolors="green", label="Unknown", linewidths=2.)
    pl.scatter(test_points[:, 0], test_points[:, 1], marker='o', s=60, facecolors='none', edgecolors="green", label=None)
    pl.legend(loc='lower right', prop={'size': 14})
    dp.close()


if __name__ == "__main__":
    args = get_aad_command_args(debug=True, debug_args=get_debug_args())
    # print "log file: %s" % args.log_file
    configure_logger(args)

    opts = AadOpts(args)
    logger.debug(opts.str_opts())

    np.random.seed(opts.randseed)

    x, y = get_synthetic_samples(stype=5)
    test_points = np.array([-3., 7.], dtype=np.float32).reshape((1,2))

    plot_dataset(x, y, opts, test_points)
    train_anomaly_detector(x, y, opts, test_points)
    train_classifier(x, y, opts, test_points)
