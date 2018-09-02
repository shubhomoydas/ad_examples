import os
import numpy as np
import matplotlib.pyplot as plt

import logging
from pandas import DataFrame

from common.data_plotter import *

from aad.aad_globals import *
from aad.aad_support import *
from aad.forest_description import *
from aad.query_model_other import *


def plot_queries(x, labels, queried, pdfpath):
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    dp.plot_points(x, pl, labels=labels, lbl_color_map={0: "grey", 1: "red"}, s=25)
    anom_idxs = np.where(labels == 1)[0]
    dp.plot_points(x[anom_idxs, :], pl, labels=labels[anom_idxs], lbl_color_map={0: "grey", 1: "red"}, s=25)
    dp.plot_points(x[queried, :],
                   pl, labels=labels[queried], defaultcol="red",
                   lbl_color_map={0: "green", 1: "red"}, edgecolor="black",
                   marker=matplotlib.markers.MarkerStyle('o', fillstyle=None), s=35)
    dp.close()


def plot_tsne_queries(x, labels, ensemble, metrics, opts):
    tsne_path = os.path.join(opts.filedir, "%s_1_tsne.csv" % opts.dataset)
    if x.shape[1] != 2 and not os.path.exists(tsne_path):
        logger.debug("Looking for %s" % tsne_path)
        logger.debug("no t-SNE file found...")
        return
    if x.shape[1] == 2:
        tsne_data = x
    else:
        tsne_data = read_csv(tsne_path, header=True, sep=' ')
        tsne_data = np.asarray(tsne_data, dtype=np.float32)
        # logger.debug("tsne_data:\n%s" % str(tsne_data[0:10, :]))
    # logger.debug("tsne_data: %s" % str(tsne_data.shape))

    pdfpath_aad = os.path.join(opts.resultsdir, "queried_aad_q%d-p%d.pdf" %
                               (opts.qtype, opts.describe_volume_p))
    pdfpath_baseline = os.path.join(opts.resultsdir, "queried_baseline.pdf")

    plot_queries(tsne_data, labels, metrics.queried, pdfpath_aad)

    nqueried = len(metrics.queried)
    queried_baseline = ensemble.ordered_anom_idxs[0:nqueried]
    plot_queries(tsne_data, labels, queried_baseline, pdfpath_baseline)


def aad_unit_tests_battery(X_train, labels, model, metrics, opts,
                           outputdir, dataset_name=""):

    data_2D = X_train.shape[1] == 2

    regcols = ["red", "blue", "green", "brown", "cyan", "pink", "orange", "magenta", "yellow", "violet"]

    xx = None; yy = None
    if data_2D:
        # plot the line, the samples, and the nearest vectors to the plane
        xx, yy = np.meshgrid(np.linspace(-4, 8, 50), np.linspace(-4, 8, 50))

    # sidebar coordinates and dimensions for showing rank locations of true anomalies
    dash_xy = (-4.0, -2.0)  # bottom-left (x,y) coordinates
    dash_wh = (0.4, 8)  # width, height

    output_forest_original = False
    output_transformed_to_file = False
    plot_dataset = data_2D and True
    plot_rectangular_regions = plot_dataset and is_forest_detector(model.detector_type) and False
    plot_forest_contours = data_2D and is_forest_detector(model.detector_type) and False
    plot_baseline = data_2D and False
    plot_aad = metrics is not None and data_2D and opts.num_query_batch == 1 and True
    plot_anomalous_regions = plot_dataset and is_forest_detector(model.detector_type) and True
    illustrate_query_diversity = plot_dataset and is_forest_detector(model.detector_type) and True
    plot_some_regions = plot_dataset and is_forest_detector(model.detector_type) and True

    pdfpath_baseline = "%s/tree_baseline.pdf" % outputdir
    pdfpath_orig_if_contours = "%s/score_contours.pdf" % outputdir

    if is_forest_detector(model.detector_type):
        logger.debug("Number of regions: %d" % len(model.d))

    tm = Timer()
    X_train_new = model.transform_to_ensemble_features(X_train, dense=False, norm_unit=opts.norm_unit)
    logger.debug(tm.message("transformed input to ensemble features"))

    if plot_dataset:
        plot_dataset_2D(X_train, labels, model, plot_rectangular_regions, regcols, outputdir)

    if plot_anomalous_regions:
        print ("plotting most anomalous regions after feedback to folder %s" % outputdir)
        plot_anomalous_2D(X_train, labels, model, metrics, outputdir,
                          n_top=opts.describe_n_top, p=opts.describe_volume_p)

    if plot_some_regions:
        print ("plotting most anomalous regions (baseline) to folder %s" % outputdir)
        plot_top_regions(X_train, labels, model, pdf_folder=outputdir, n=50)

    if illustrate_query_diversity:
        print ("plotting query diversity to folder %s" % outputdir)
        plot_query_diversity(X_train, labels, X_train_new, model, metrics, outputdir, opts)

    if output_forest_original:
        n_found = evaluate_forest_original(X_train, labels, opts.budget, model, x_new=X_train_new)
        np.savetxt(os.path.join(outputdir, "iforest_original_num_found_%s.csv" % dataset_name),
                   n_found, fmt='%3.2f', delimiter=",")

    if plot_forest_contours:
        print ("plotting contours to file %s" % pdfpath_orig_if_contours)
        plot_forest_contours_2D(X_train, labels, xx, yy, opts.budget, model,
                                pdfpath_orig_if_contours, dash_xy, dash_wh)

    if output_transformed_to_file:
        write_sparsemat_to_file(os.path.join(outputdir, "forest_features.csv"),
                                X_train_new, fmt='%3.2f', delimiter=",")
        x_tmp = np.vstack((model.d, model.node_samples, model.frac_insts))
        write_sparsemat_to_file(os.path.join(outputdir, "forest_node_info.csv"),
                                x_tmp.T, fmt='%3.2f', delimiter=",")

    if plot_baseline:
        plot_model_baseline_contours_2D(X_train, labels, X_train_new, xx, yy, opts.budget, model,
                                        pdfpath_baseline, dash_xy, dash_wh, opts)

    if plot_aad and metrics is not None:
        print ("plotting feedback iterations in folder %s" % outputdir)
        plot_aad_2D(X_train, labels, X_train_new, xx, yy, model,
                    metrics, outputdir, dash_xy, dash_wh, opts)


def check_random_vector_angle(model, vec, samples=None):
    tmp = np.zeros(200, dtype=float)
    for i in range(len(tmp)):
        rndw = model.get_random_weights(samples=samples)
        cos_theta = vec.dot(rndw)
        tmp[i] = np.arccos(cos_theta) * 180. / np.pi
    logger.debug("random vector angles:\n%s" % (str(list(tmp))))


def plot_qval_hist(qfirst, qlast, i, outputdir):
    pdfpath = "%s/qval_hist_%d.pdf" % (outputdir, i)
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    qall = np.append(qfirst, qlast)
    bins = np.arange(start=np.min(qall), stop=np.max(qall), step=(np.max(qall)-np.min(qall))/50)
    pl = dp.get_next_plot()
    n1, bins1 = np.histogram(qfirst, bins=bins, normed=True)
    n2, bins2 = np.histogram(qlast, bins=bins, normed=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, n1, align='center', width=width, facecolor='green', alpha=0.50)
    plt.bar(center, n2, align='center', width=width, facecolor='red', alpha=0.50)
    dp.close()


def debug_qvals(samples, model, metrics, outputdir, opts):
    n = samples.shape[0]
    bt = get_budget_topK(n, opts)
    unif_w = model.get_uniform_weights()
    budget = metrics.all_weights.shape[0]
    if budget > 1:
        plot_qval_hist(metrics.all_weights[0], metrics.all_weights[budget-1], opts.runidx, outputdir)
    for i in range(metrics.all_weights.shape[0]):
        w = metrics.all_weights[i]
        s = samples.dot(w)
        qval = quantile(s, (1.0 - (bt.topK * 1.0 / float(n))) * 100.0)
        qmin = np.min(s)
        qmax = np.max(s)
        cos_theta = max(-1.0, min(1.0, unif_w.dot(w)))
        # logger.debug("cos_theta: %f" % cos_theta)
        angle = np.arccos(cos_theta) * 180. / np.pi
        logger.debug("[%d] qval: %1.6f [%1.6f, %1.6f]; angle: %2.6f" % (i, qval, qmin, qmax, angle))
    est_qval, est_qmin, est_qmax = estimate_qtau(samples, model, opts, lo=0.0, hi=1.0)
    logger.debug("[%d] estimated qval (0, 1): %1.6f [%1.6f, %1.6f]" % (opts.runidx, est_qval, est_qmin, est_qmax))
    est_qval, est_qmin, est_qmax = estimate_qtau(samples, model, opts, lo=-1.0, hi=1.0)
    logger.debug("[%d] estimated qval (-1,1): %1.6f [%1.6f, %1.6f]" % (opts.runidx, est_qval, est_qmin, est_qmax))


def plot_aad_2D(x, y, x_transformed, xx, yy, model, metrics,
                outputdir, dash_xy, dash_wh, opts):
    # use this to plot the AAD feedback
    tm = Timer()
    tm.start()

    x_test = np.c_[xx.ravel(), yy.ravel()]
    x_if = model.transform_to_ensemble_features(x_test, dense=False, norm_unit=opts.norm_unit)

    queried = np.array(metrics.queried)
    for i, q in enumerate(queried):
        pdfpath = "%s/iter_%02d.pdf" % (outputdir, i)
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()

        w = metrics.all_weights[i, :]
        Z = model.get_score(x_if, w)
        Z = Z.reshape(xx.shape)
        pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))

        dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        # print queried[np.arange(i+1)]
        # print X_train[queried[np.arange(i+1)], :]
        dp.plot_points(matrix(x[queried[np.arange(i+1)], :], nrow=i+1),
                       pl, labels=y[queried[np.arange(i+1)]], defaultcol="red",
                       lbl_color_map={0: "green", 1: "red"}, edgecolor=None, facecolors=True,
                       marker=matplotlib.markers.MarkerStyle('o', fillstyle=None), s=35)

        # plot the sidebar
        anom_scores = model.get_score(x_transformed, w)
        anom_order = np.argsort(-anom_scores)
        anom_idxs = np.where(y[anom_order] == 1)[0]
        dash = 1 - (anom_idxs * 1.0 / x.shape[0])
        plot_sidebar(dash, dash_xy, dash_wh, pl)

        dp.close()

    logger.debug(tm.message("plotted feedback iterations"))


def plot_score_contours(x, y, x_transformed, selected_x=None, model=None,
                        filename=None, outputdir=None, opts=None):
    # use this to plot the anomaly contours

    data_2D = x.shape[1] == 2
    can_plot = data_2D and opts.reruns == 1
    if not can_plot:
        return

    pdfpath = "%s/%s.pdf" % (outputdir, filename)
    logger.debug("Plotting contours to %s" % (pdfpath))

    tm = Timer()
    tm.start()

    xx = None
    yy = None

    if data_2D:
        # plot the line, the samples, and the nearest vectors to the plane
        xx, yy = np.meshgrid(np.linspace(-4, 8, 50), np.linspace(-4, 8, 50))

    # sidebar coordinates and dimensions for showing rank locations of true anomalies
    dash_xy = (-4.0, -2.0)  # bottom-left (x,y) coordinates
    dash_wh = (0.4, 8)  # width, height

    x_test = np.c_[xx.ravel(), yy.ravel()]
    x_if = model.transform_to_ensemble_features(x_test, dense=False, norm_unit=opts.norm_unit)

    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()

    w = model.w
    Z = model.get_score(x_if, w)
    Z = Z.reshape(xx.shape)
    pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))

    dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)

    if selected_x is not None and selected_x.shape[0] > 0:
        dp.plot_points(selected_x, pl, defaultcol='green', s=40)

    # plot the sidebar
    anom_scores = model.get_score(x_transformed, w)
    anom_order = np.argsort(-anom_scores)
    anom_idxs = np.where(y[anom_order] == 1)[0]
    dash = 1 - (anom_idxs * 1.0 / x.shape[0])
    plot_sidebar(dash, dash_xy, dash_wh, pl)

    dp.close()

    logger.debug(tm.message("plotted score contours"))


def evaluate_forest_original(x, y, budget, forest, x_new=None):
    original_scores = 0.5 - forest.decision_function(x)
    queried = np.argsort(-original_scores)

    n_found_orig = np.cumsum(y[queried[np.arange(budget)]])
    # logger.debug("original isolation forest:")
    # logger.debug(n_found_orig)

    if x_new is not None:
        w = np.ones(len(forest.d), dtype=float)
        w = w / w.dot(w)  # normalized uniform weights
        agg_scores = forest.get_score(x_new, w)
        queried = np.argsort(-agg_scores)
        n_found_baseline = np.cumsum(y[queried[np.arange(budget)]])
        n_found = np.vstack((n_found_baseline, n_found_orig)).T
    else:
        n_found = np.reshape(n_found_orig, (1, len(n_found_orig)))
    return n_found


def plot_model_baseline_contours_2D(x, y, x_transformed, xx, yy, budget, model,
                                    pdfpath_contours, dash_xy, dash_wh, opts):
    # use this to plot baseline query points.

    w = np.ones(model.get_num_members(), dtype=float)
    w = w / w.dot(w)  # normalized uniform weights

    baseline_scores = model.get_score(x_transformed, w)
    queried = np.argsort(-baseline_scores)

    n_found = np.cumsum(y[queried[np.arange(budget)]])
    print (n_found)

    dp = DataPlotter(pdfpath=pdfpath_contours, rows=1, cols=1)
    pl = dp.get_next_plot()

    x_test = np.c_[xx.ravel(), yy.ravel()]
    x_if = model.transform_to_ensemble_features(x_test, dense=False, norm_unit=opts.norm_unit)
    y_if = model.get_score(x_if, w)
    Z = y_if.reshape(xx.shape)

    pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))

    dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
    # print queried[np.arange(i+1)]
    # print X_train[queried[np.arange(i+1)], :]
    dp.plot_points(matrix(x[queried[np.arange(budget)], :], nrow=budget),
                   pl, labels=y[queried[np.arange(budget)]], defaultcol="red",
                   lbl_color_map={0: "green", 1: "red"}, edgecolor="black",
                   marker=matplotlib.markers.MarkerStyle('o', fillstyle=None), s=35)

    # plot the sidebar
    anom_idxs = np.where(y[queried] == 1)[0]
    dash = 1 - (anom_idxs * 1.0 / x.shape[0])
    plot_sidebar(dash, dash_xy, dash_wh, pl)

    dp.close()


def plot_forest_contours_2D(x, y, xx, yy, budget, forest, pdfpath_contours, dash_xy, dash_wh):
    # Original detector contours
    tm = Timer()
    tm.start()
    baseline_scores = 0.5 - forest.decision_function(x)
    queried = np.argsort(-baseline_scores)
    # logger.debug("baseline scores:%s\n%s" % (str(baseline_scores.shape), str(list(baseline_scores))))

    # n_found = np.cumsum(y[queried[np.arange(budget)]])
    # print n_found

    Z_if = 0.5 - forest.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_if = Z_if.reshape(xx.shape)

    dp = DataPlotter(pdfpath=pdfpath_contours, rows=1, cols=1)
    pl = dp.get_next_plot()
    pl.contourf(xx, yy, Z_if, 20, cmap=plt.cm.get_cmap('jet'))

    dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"})

    dp.plot_points(matrix(x[queried[np.arange(budget)], :], nrow=budget),
                   pl, labels=y[queried[np.arange(budget)]], defaultcol="red",
                   lbl_color_map={0: "green", 1: "red"}, edgecolor="black",
                   marker=matplotlib.markers.MarkerStyle('o', fillstyle=None), s=35)

    # plot the sidebar
    anom_idxs = np.where(y[queried] == 1)[0]
    dash = 1 - (anom_idxs * 1.0 / x.shape[0])
    plot_sidebar(dash, dash_xy, dash_wh, pl)

    dp.close()

    logger.debug(tm.message("plotted contours"))


def plot_dataset_2D(x, y, model, plot_regions, regcols, pdf_folder):
    # use this to plot the dataset
    tm = Timer()
    tm.start()
    if is_forest_detector(model.detector_type):
        treesig = "_%d_trees" % model.n_estimators if plot_regions else ""
    else:
        treesig = ""
    pdfpath_dataset = "%s/synth_dataset%s.pdf" % (pdf_folder, treesig)
    dp = DataPlotter(pdfpath=pdfpath_dataset, rows=1, cols=1)
    pl = dp.get_next_plot()

    # dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"})
    dp.plot_points(x[y==0, :], pl, labels=y[y==0], defaultcol="grey")
    dp.plot_points(x[y==1, :], pl, labels=y[y==1], defaultcol="red", s=26, linewidths=1.5)

    if plot_regions:
        # plot the isolation model tree regions
        axis_lims = (plt.xlim(), plt.ylim())
        for i, regions in enumerate(model.regions_in_forest):
            for region in regions:
                region = region.region
                plot_rect_region(pl, region, regcols[i % len(regcols)], axis_lims)
    dp.close()

    logger.debug(tm.message("plotted dataset"))


def plot_selected_regions(x, y, model, region_indexes,
                          candidate_instances=None, query_instances=None,
                          title=None, path=None, candidate_instance_marker='x'):
    dp = DataPlotter(pdfpath=path, rows=1, cols=1)

    pl = dp.get_next_plot()
    if title is not None: plt.title(title, fontsize=8)
    # dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"})
    dp.plot_points(x[y == 0, :], pl, labels=y[y == 0], defaultcol="grey")
    dp.plot_points(x[y == 1, :], pl, labels=y[y == 1], defaultcol="red", s=26, linewidths=1.5)

    if candidate_instances is not None:
        dp.plot_points(x[candidate_instances, :], pl, labels=None, defaultcol="blue",
                       edgecolor="blue" if candidate_instance_marker == 'o' else None,
                       s=70, linewidths=3.0, marker=candidate_instance_marker)

    if query_instances is not None:
        dp.plot_points(x[query_instances, :], pl, labels=None, edgecolor="green", s=150, linewidths=3.0, marker='o')

    # plot the isolation model tree regions
    axis_lims = (plt.xlim(), plt.ylim())
    for i in region_indexes:
        region = model.all_regions[i].region
        # logger.debug(str(region))
        plot_rect_region(pl, region, "red", axis_lims)

    dp.close()


def plot_top_regions(x, y, model, pdf_folder, n=50):
    tm = Timer()
    tm.start()
    if is_forest_detector(model.detector_type):
        treesig = "_%d_trees" % model.n_estimators
    else:
        logger.debug("Plotting anomalous regions only supported for tree-based models")
        return

    all_regions = np.argsort(-model.d)
    reg_idxs = all_regions[0:n]
    plot_selected_regions(x, y, model, reg_idxs,
                          path="%s/top_%d_regions%s.pdf" % (pdf_folder, n, treesig))


def plot_anomalous_2D(x, y, model, metrics, pdf_folder, n_top=-1, p=1):
    # use this to plot the dataset
    tm = Timer()
    tm.start()
    if is_forest_detector(model.detector_type):
        treesig = "_%d_trees" % model.n_estimators
    else:
        treesig = ""

    # logger.debug("queried:\n%s" % metrics.queried)
    instance_indexes = get_instances_for_description(x, labels=y, metrics=metrics)
    # logger.debug("instance_indexes:\n%s" % instance_indexes)

    if instance_indexes is None or len(instance_indexes) == 0:
        logger.debug("No true anomalies found to describe...")
        return

    feature_ranges = get_sample_feature_ranges(x)  # will be used to compute volumes
    # logger.debug("feature_ranges:\n%s" % feature_ranges)

    reg_idxs = get_regions_for_description(x, instance_indexes=instance_indexes,
                                           model=model, n_top=n_top)
    volumes = get_region_volumes(model, reg_idxs, feature_ranges)
    ordered_compact_idxs = get_compact_regions(x, model=model,
                                               instance_indexes=instance_indexes,
                                               region_indexes=reg_idxs,
                                               volumes=volumes, p=p)
    # logger.debug("#ordered_compact_idxs:%d" % len(ordered_compact_idxs))

    ordered_d_idxs = np.argsort(-model.d)[0:n_top]  # sort in reverse order
    # logger.debug("ordered_d:\n%s" % str(model.d[ordered_d_idxs]))

    plot_selected_regions(x, y, model, ordered_d_idxs,
                          # title="Baseline Top %d Regions" % n_top,
                          path="%s/top_%d_anomalous_regions%s_baseline.pdf" % (pdf_folder, n_top, treesig))
    plot_selected_regions(x, y, model, reg_idxs,
                          # title="AAD Top %d Regions" % n_top,
                          path="%s/top_%d_anomalous_regions%s_aad.pdf" % (pdf_folder, n_top, treesig))

    plot_selected_regions(x, y, model, ordered_compact_idxs,
                          # title="AAD Top %d Regions Compact" % n_top,
                          path="%s/top_%d_anomalous_regions%s_compact.pdf" % (pdf_folder, n_top, treesig))

    logger.debug(tm.message("plotted anomalous regions"))


def plot_query_diversity(x, y, x_transformed, model, metrics, pdf_folder, opts):
    tm = Timer()
    tm.start()
    if is_forest_detector(model.detector_type):
        treesig = "_%d_trees" % model.n_estimators
    else:
        raise RuntimeError("Operation supported for only tree-based detectors...")

    n_top = opts.describe_n_top
    p = opts.describe_volume_p

    # logger.debug("queried:\n%s" % metrics.queried)
    # baseline_scores = model.get_score(x_transformed, model.get_uniform_weights())
    n_selected_queries = 5
    n_candidate_instances = 15
    baseline_scores = model.get_score(x_transformed, model.w)
    ordered_instance_indexes = np.argsort(-baseline_scores)
    top_anomalous_instances = ordered_instance_indexes[0:n_candidate_instances]
    # logger.debug("top_anomalous_instances:\n%s" % top_anomalous_instances)

    if top_anomalous_instances is None or len(top_anomalous_instances) == 0:
        logger.debug("No true anomalies found to describe...")
        return

    feature_ranges = get_sample_feature_ranges(x)  # will be used to compute volumes
    # logger.debug("feature_ranges:\n%s" % feature_ranges)

    # most influential regions
    influential_region_indexes = get_regions_for_description(x,
                                                             instance_indexes=top_anomalous_instances,
                                                             model=model, region_score_only=True,
                                                             n_top=n_top)
    logger.debug("#influential_region_indexes: %d" % len(influential_region_indexes))
    plot_selected_regions(x, y, model, influential_region_indexes,
                          candidate_instances=top_anomalous_instances,
                          candidate_instance_marker='o',
                          # title="AAD Most Influential %d Regions for Gradient-based Learning" % n_top,
                          path="%s/influential_regions_ntop%d%s.pdf" % (pdf_folder, n_top, treesig))

    # first, select some top-ranked subspaces as candidate regions
    candidate_region_indexes = get_regions_for_description(x,
                                                           instance_indexes=top_anomalous_instances,
                                                           model=model, region_score_only=False, n_top=n_top)
    # get volumes of the candidate regions
    volumes = get_region_volumes(model, candidate_region_indexes, feature_ranges)
    compact_region_idxs = get_compact_regions(x, model=model,
                                              instance_indexes=top_anomalous_instances,
                                              region_indexes=candidate_region_indexes,
                                              volumes=volumes, p=p)
    # logger.debug("#compact_region_idxs:%d\n%s" % (len(compact_region_idxs), str(list(compact_region_idxs))))

    # get the region memberships of the top anomalous instances
    instance_ids, region_memberships = get_region_memberships(x,
                                                              instance_indexes=top_anomalous_instances,
                                                              region_indexes=compact_region_idxs,
                                                              model=model)
    # logger.debug("instance_ids:%s" % str(list(instance_ids)))
    # logger.debug("region_memberships:\n%s" % str(region_memberships))
    if opts.qtype == QUERY_EUCLIDEAN:
        # get a diverse subset of the top anomalous instances
        filtered_items = filter_by_euclidean_distance(x, instance_ids, n_select=n_selected_queries,
                                                      dist_type=opts.query_euclidean_dist_type)
    else:
        # get a diverse subset of the top anomalous instances
        query_model = QueryTopDiverseSubspace()
        query_model.order_by_euclidean_diversity = opts.qtype == QUERY_SUBSPACE_EUCLIDEAN
        filtered_items = query_model.filter_by_diversity(instance_ids, region_memberships,
                                                         queried=None,
                                                         n_select=n_selected_queries)
    # logger.debug("filtered_items: %s" % (str(list(filtered_items))))

    logger.debug("#candidate_region_indexes: %d" % len(candidate_region_indexes))
    dummy_y = np.zeros(len(y), dtype=int)
    plot_selected_regions(x, dummy_y, model, candidate_region_indexes,
                          candidate_instances=top_anomalous_instances,
                          # title="AAD Top %d Regions" % n_top,
                          path="%s/query_candidate_regions_ntop%d%s.pdf" % (pdf_folder, n_top, treesig))

    plot_selected_regions(x, dummy_y, model, compact_region_idxs,
                          candidate_instances=top_anomalous_instances,
                          query_instances=top_anomalous_instances[0:n_selected_queries],
                          # title="AAD Top %d Regions Compact" % n_top,
                          path="%s/query_compact_ntop%d%s_baseline.pdf" % (pdf_folder, n_top, treesig))

    plot_selected_regions(x, dummy_y, model, compact_region_idxs,
                          candidate_instances=top_anomalous_instances,
                          query_instances=filtered_items,
                          # title="AAD Top %d Regions Compact" % n_top,
                          path="%s/query_compact_ntop%d%s_aad.pdf" % (pdf_folder, n_top, treesig))

    logger.debug(tm.message("plotted query diversity"))


def test_ilp():
    """
    Problem defined in:
        https://en.wikipedia.org/wiki/Integer_programming#Example
    Solution from:
        https://stackoverflow.com/questions/33785396/python-the-integer-linear-programming-ilp-function-in-cvxopt-is-not-generati
    """
    import cvxopt
    from cvxopt import glpk

    glpk.options['msg_lev'] = 'GLP_MSG_OFF'
    c = cvxopt.matrix([0, -1], tc='d')
    G = cvxopt.matrix([[-1, 1], [3, 2], [2, 3], [-1, 0], [0, -1]], tc='d')
    h = cvxopt.matrix([1, 12, 12, 0, 0], tc='d')
    (status, x) = cvxopt.glpk.ilp(c, G.T, h, I=set([0, 1]))
    print (status)
    print ("%s, %s" % (str(x[0]), str(x[1])))
    print (sum(c.T * x))
