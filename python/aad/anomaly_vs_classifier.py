from common.data_plotter import *
from common.gen_samples import *

from aad.aad_support import *
from aad.demo_aad import describe_instances
from aad.classifier_trees import *

"""
pythonw -m aad.anomaly_vs_classifier --dataset=5 --algo=explain
"""

logger = logging.getLogger(__name__)


def get_debug_args(dataset="", detector_type=AAD_IFOREST):
    return ["--resultsdir=./temp", "--randseed=42",
            "--reruns=1",
            "--dataset=%s" % dataset,
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
            "--describe_n_top=5",
            "--resultsdir=./temp",
            "--log_file=./temp/anomaly_vs_classifier.log",
            "--debug"]


def get_auc(model, x, y, x_transformed=None):
    scores = model.get_score(x=x_transformed)
    auc = fn_auc(cbind(y, -scores))
    return auc


def plot_regions(model, region_indexes, pl):
    axis_lims = (plt.xlim(), plt.ylim())
    for i in region_indexes:
        region = model.all_regions[i].region
        plot_rect_region(pl, region, "red", axis_lims)


def train_anomaly_detector(x, y, opts, test_points, name, explain=False):
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

    pdfpath = "%s/%s.pdf" % (opts.resultsdir, "%s_anomaly" % name)
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

    if explain:
        ridxs_counts, region_extents = describe_instances(x, np.array(ha), model=model, opts=opts)
        logger.debug("selected region indexes and corresponding instance counts (among %d):\n%s" %
                     (len(ha), str(list(ridxs_counts))))
        region_indexes = [region for region, count in ridxs_counts]
        plot_dataset(x, y, opts, test_points=None, name="%s_anomaly_descriptions" % name,
                     model=model, region_indexes=region_indexes, legend=False)


def train_classifier(x, y, opts, test_points, name, explain=False):
    classifier = RFClassifier.fit(x, y, n_estimators=100, max_depth=None)

    xx, yy = np.meshgrid(np.linspace(-4, 8, 50), np.linspace(-4, 8, 50))
    x_test = np.c_[xx.ravel(), yy.ravel()]
    probs_grid = classifier.predict_prob_for_class(x_test, 1)
    Z = probs_grid.reshape(xx.shape)

    pdfpath = "%s/%s.pdf" % (opts.resultsdir, "%s_classifier" % name)
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

    if explain:
        # generate compact descriptions from the Random Forest classifier
        rfre = RandomForestAadWrapper(x, y, classifier.clf)
        ha = np.where(y == 1)[0]
        ridxs_counts, region_extents = describe_instances(x, ha, model=rfre, opts=opts)
        logger.debug("selected random forest region indexes and corresponding instance counts (among %d):\n%s" %
                     (len(ha), str(list(ridxs_counts))))
        region_indexes = [region for region, count in ridxs_counts]
        plot_dataset(x, y, opts, test_points=None, name="%s_random_forest_descriptions" % name,
                     model=rfre, region_indexes=region_indexes, legend=False)


def plot_decision_tree_descriptions(x, y, name):
    ha = np.where(y == 1)[0]
    # generate compact descriptions from the Decision Tree classifier;
    # these just correspond to the rules extracted from the tree structure
    dt = DecisionTreeAadWrapper(x, y)
    ridxs_counts, region_extents = describe_instances(x, ha, model=dt, opts=opts)
    logger.debug("selected decision tree region indexes and corresponding instance counts (among %d):\n%s" %
                 (len(ha), str(list(ridxs_counts))))
    region_indexes = [region for region, count in ridxs_counts]
    plot_dataset(x, y, opts, test_points=None, name="%s_decision_tree_descriptions" % name,
                 model=dt, region_indexes=region_indexes, legend=False)


def plot_dataset(x, y, opts, test_points=None, name=None, model=None, region_indexes=None, legend=True):
    pdfpath = "%s/%s.pdf" % (opts.resultsdir, name)
    logger.debug("Plotting dataset to %s" % (pdfpath))
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    nomls = np.where(y == 0)[0]
    anoms = np.where(y == 1)[0]
    pl.scatter(x[nomls, 0], x[nomls, 1], marker='x', s=25, facecolors='blue', edgecolors="blue", label="Nominal")
    pl.scatter(x[anoms, 0], x[anoms, 1], marker='x', s=25, facecolors='red', edgecolors="red", label="Anomaly")

    if test_points is not None:
        pl.scatter(test_points[:, 0], test_points[:, 1], marker='x', s=50, facecolors='green', edgecolors="green", label="Unknown", linewidths=2.)
        pl.scatter(test_points[:, 0], test_points[:, 1], marker='o', s=60, facecolors='none', edgecolors="green", label=None)

    if legend:
        pl.legend(loc='lower right', prop={'size': 14})

    if model is not None and region_indexes is not None:
        plot_regions(model, region_indexes, pl)

    dp.close()


if __name__ == "__main__":
    # the main program arguments from commandline
    args = get_command_args(debug=False)
    explain = args.algo == "explain"

    # the AAD arguments
    aadArgs = get_aad_command_args(debug=True,
                                   debug_args=get_debug_args(dataset=args.dataset))
    # print "log file: %s" % aadArgs.log_file
    configure_logger(aadArgs)

    opts = AadOpts(aadArgs)
    logger.debug(opts.str_opts())

    np.random.seed(opts.randseed)

    synthetic_dataset_id = int(args.dataset)
    x, y = get_synthetic_samples(stype=synthetic_dataset_id)
    test_points = np.array([-3., 7.], dtype=np.float32).reshape((1,2))

    name = "avc_dataset_%d" % synthetic_dataset_id
    plot_dataset(x, y, opts, test_points=test_points, name=name)
    train_anomaly_detector(x, y, opts, test_points, name=name, explain=explain)
    train_classifier(x, y, opts, test_points, name=name, explain=explain)

    # plot rules from a decision tree classifier
    if explain:
        plot_decision_tree_descriptions(x, y, name=name)
