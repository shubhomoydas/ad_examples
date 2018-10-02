from common.gen_samples import *
from .afss import partition_instances, get_afss_batches
from .glad_support import *
from loda.loda import build_proj_hist, ProjectionVectorsHistograms, get_neg_ll_all_hist, \
    LodaModel, LodaResult


def plot_dataset(x, y, cls_cols, pl, selected=None, projections=None):
    plt.xlim([np.min(x[:, 0]), np.max(x[:, 0])])
    plt.ylim([np.min(x[:, 1]), np.max(x[:, 1])])
    for cls in cls_cols.keys():
        X = x[np.where(y == cls)[0], :]
        pl.scatter(X[:, 0], X[:, 1], c=cls_cols.get(cls, "grey"), marker='x',
                   linewidths=2.0, s=24, label="class %d (%s)" % (cls, "nominal" if cls == 0 else "anomaly"))
    if selected is not None:
        pl.scatter(selected[:, 0], selected[:, 1], edgecolors="green", facecolors='none', marker='o',
                   linewidths=2.0, s=40, label="selected")
    if projections is not None:
        for p in range(projections.shape[1]):
            u = interpolate_2D_line_by_point_and_vec(np.array([np.min(x[:, 0]), np.max(x[:, 0])]),
                                                     [0.0, 0.0], projections[:, p])
            pl.plot(u[:, 0], u[:, 1], "-", color="green", linewidth=2)
    pl.legend(loc='lower right', prop={'size': 4})


def get_grid(x=None, xx=None, yy=None, x_range=None, y_range=None, n=50):
    if xx is None or yy is None:
        if x_range is None: x_range = [np.min(x[:, 0]), np.max(x[:, 0])]
        if y_range is None: y_range = [np.min(x[:, 1]), np.max(x[:, 1])]
        xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], n), np.linspace(y_range[0], y_range[1], n))
    return xx, yy


def plot_scores(x, model, pl, ylabel='p', fn_score_transform=None, contour_levels=20,
                cmap='jet', xx=None, yy=None, x_range=None, y_range=None):
    """

    :param x: np.ndarray
    :param model: some models like LodaResult, AFSS which support the function decision_function()
    :param fn_score_transform: function to transform the scores computed by the model
    :param contour_levels: int or array-like
    :param cmap: name of matplotlib color map
    :param xx: np.ndarray
    :param yy: np.ndarray
    :param x_range: list
    :param y_range: list
    :return:
    """
    xx, yy = get_grid(x=x, xx=xx, yy=yy, x_range=x_range, y_range=y_range, n=50)
    x_test = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(x_test)
    if fn_score_transform is not None:
        scores = fn_score_transform(scores)
    scores = scores.reshape(xx.shape)
    # logger.debug(scores.shape)
    cf = pl.contourf(xx, yy, scores, contour_levels, cmap=plt.cm.get_cmap(cmap))
    cbar = plt.colorbar(cf)
    cbar.ax.set_ylabel(ylabel)
    return xx, yy


def plot_ensemble_scores(x, y, ensemble, selected=None, xx=None, yy=None, dataset=None, outpath=None):
    """ Assumes the ensemble is projection-based such as LODA """
    scores = ensemble.get_scores(x)
    logger.debug("ensemble projs: %s, scores: %s" % (str(ensemble.get_projections().shape), str(scores.shape)))
    dp = DataPlotter(pdfpath="%s/test_afss_%s_loda_%dw.pdf" %
                             (outpath, dataset, ensemble.get_projections().shape[1]), rows=1, cols=1)
    pl = dp.get_next_plot()
    xx, yy = plot_scores(x, ensemble, pl, ylabel='anomaly score', fn_score_transform=None, xx=xx, yy=yy)
    cls_cols = {0: "grey", 1: "red"}
    plot_dataset(x, y, cls_cols, pl, selected=selected, projections=ensemble.get_projections())
    dp.close()
    return xx, yy


def plot_afss_scores(x, y, ensemble, afss, selected=None, plot_ensemble=False,
                     cmap='jet_r', xx=None, yy=None, name="", dataset=None, outpath=None):
    """ Assumes the ensemble is projection-based such as LODA """
    cls_cols = {0: "grey", 1: "red"}
    xx, yy = get_grid(x=x, xx=xx, yy=yy, n=50)
    x_test = np.c_[xx.ravel(), yy.ravel()]

    score_type = "afss"
    if plot_ensemble:
        scores_all = ensemble.get_scores(x_test)
        score_type = ensemble.get_ensemble_type()
        contour_levels = 100
        ylabel = 'anomaly score'
    else:
        scores_all = afss.decision_function(x_test)
        contour_levels = np.arange(start=0.0, stop=1.02, step=0.02, dtype=np.float32)
        ylabel = 'p'

    p_rows = p_cols = 1
    if ensemble.m > 1: p_rows = p_cols = 2
    dp = DataPlotter(pdfpath="%s/test_%s_%s_active_%dw%s.pdf" %
                             (outpath, score_type, dataset, ensemble.m, name),
                     rows=p_rows, cols=p_cols)
    for i in range(scores_all.shape[1]):
        pl = dp.get_next_plot()

        scores = scores_all[:, i]
        scores = scores.reshape(xx.shape)
        # logger.debug(scores.shape)
        cf = pl.contourf(xx, yy, scores, contour_levels, cmap=plt.cm.get_cmap(cmap))
        cbar = plt.colorbar(cf)
        cbar.ax.set_ylabel(ylabel)

        plot_dataset(x, y, cls_cols, pl, selected=selected, projections=ensemble.get_projections()[:, [i]])
    dp.close()

    return xx, yy


def plot_weighted_scores(x, y, ensemble, afss, selected=None, xx=None, yy=None, contour_levels=20, name="", n_anoms=None, dataset=None, outpath=None):
    """ Assumes the ensemble is projection-based such as LODA """
    xx, yy = get_grid(x=x, xx=xx, yy=yy, n=50)
    x_test = np.c_[xx.ravel(), yy.ravel()]
    if selected is None and n_anoms is not None:
        proj_scores = ensemble.get_scores(x)
        anomaly_scores = afss.get_weighted_scores(x, proj_scores)
        sorted_indexes = np.argsort(-anomaly_scores)
        hf = sorted_indexes[0:n_anoms]
        logger.debug("weighted top scores:\n%s" % (str(list(anomaly_scores[hf]))))
        selected = x[hf]
    ensemble_scores = ensemble.get_scores(x_test)
    composite_scores = afss.get_weighted_scores(x_test, ensemble_scores)
    scores = composite_scores.reshape(xx.shape)
    dp = DataPlotter(pdfpath="%s/test_afss_%s_anomaly_scores_%dw%s.pdf" %
                             (outpath, dataset, ensemble.get_projections().shape[1], name), rows=1, cols=1)
    pl = dp.get_next_plot()
    cf = pl.contourf(xx, yy, scores, contour_levels, cmap=plt.cm.get_cmap('jet'))
    cbar = plt.colorbar(cf)
    cbar.ax.set_ylabel('anomaly score')
    cls_cols = {0: "grey", 1: "red"}
    plot_dataset(x, y, cls_cols, pl, selected=selected, projections=ensemble.get_projections())
    dp.close()
    return xx, yy


def prepare_loda_model_with_w(x, w):
    hists = build_proj_hist(x, w, verbose=False)
    pvh = ProjectionVectorsHistograms(w=w, hists=hists)
    nll = get_neg_ll_all_hist(x, w, hists, inf_replace=np.nan)
    anomranks = np.arange(x.shape[0])
    anomranks = anomranks[order(-nll)]
    loda_ = LodaModel(k=w.shape[1], pvh=pvh, sigs=None)
    loda_model = LodaResult(anomranks=anomranks, nll=nll, pvh=loda_)
    loda = Loda()
    loda.m = w.shape[1]
    loda.loda_model = loda_model
    return loda


def prepare_loda_ensemble(x, mink=2, maxk=10, debug=False, m=2):
    tm = Timer()
    if debug:
        # w = np.ones(shape=(2, 1), dtype=np.float32)
        w = np.array([[7.5, 4.0], [4, 6], [-2, 2], [-1, 8]], dtype=np.float32)[0:m]
        w = np.transpose(w)
        w2 = np.sum(w ** 2, axis=0)
        a = np.diag(1./np.sqrt(w2))
        w = np.dot(w, a)
        logger.debug("w:\n%s" % str(w))

        loda_model = prepare_loda_model_with_w(x, w)
        logger.debug("nll: %s" % (str(loda_model.loda_model.nll.shape)))
    else:
        loda_model = Loda(mink=mink, maxk=maxk)
        loda_model.fit(x)
    logger.debug(tm.message("trained LODA model:"))
    return AnomalyEnsembleLoda(loda_model)


def get_top_ranked_instances(x, ensemble, n=10):
    a_scores = ensemble.decision_function(x)
    sorted_indexes = np.argsort(-a_scores)
    top = sorted_indexes[0:n]
    return top, a_scores


def test_get_afss_batches(opts):
    x, y = read_anomaly_dataset(opts.dataset, datafile=opts.datafile)

    ensemble = prepare_loda_ensemble(x, debug=True, m=4)
    hf, _ = get_top_ranked_instances(x, ensemble, n=opts.n_anoms)
    scores = ensemble.get_scores(x)

    x_unlabeled, y_unlabeled, scores_unlabeled, x_labeled, y_labeled, scores_labeled = \
        partition_instances(x, y, scores, hf)

    for batch_x, batch_y, batch_scores, b_n_lbl, b_n_unl in get_afss_batches(x_labeled, y_labeled, scores_labeled,
                                                                             x_unlabeled,
                                                                             scores_unlabeled,
                                                                             batch_size=25):
        logger.debug("#instances: %d, b_n_lbl: %d, b_n_unl: %d" % (batch_x.shape[0], b_n_lbl, b_n_unl))
        logger.debug("batch_y: %s" % (str(list(batch_y))))


def test_loda(opts):
    set_results_dir(opts)
    dir_create(opts.results_dir)

    x, y = read_anomaly_dataset(opts.dataset, datafile=opts.datafile)
    ensemble = prepare_loda_ensemble(x, debug=opts.loda_debug and x.shape[1] == 2, m=4)

    hf, _ = get_top_ranked_instances(x, ensemble, n=opts.n_anoms)

    if opts.plot:
        plot_ensemble_scores(x, y, ensemble, selected=x[hf], dataset=opts.dataset, outpath=opts.results_dir)


