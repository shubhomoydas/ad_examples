from .afss import *
from .glad_test_support import *

"""
An implementation of:
    GLAD: *GL*ocalized *A*nomaly *D*etection via Active Feature Space Suppression

python -m glad.glad_batch --log_file=temp/glad/glad_batch.log --debug --dataset=toy2 --n_epochs=200 --afss_bias_prob=0.50 --train_batch_size=25 --budget=60 --afss_nodes=0 --afss_max_labeled_reps=5 --loda_debug --plot
"""


supported_ensemble_types = ["loda"]


def afss_active_learn_ensemble(x, y, ensemble, opts):

    data_2D = x.shape[1] == 2
    plot = opts.plot and data_2D

    # populate labels as some dummy value (-1) initially
    y_labeled = np.ones(x.shape[0], dtype=int) * -1

    scores = ensemble.get_scores(x)

    xx = yy = None

    afss = get_afss_model(opts, n_output=ensemble.m)

    afss.init_network(x, prime_network=True)

    baseline_scores = afss.get_weighted_scores(x, scores)
    baseline_queried = np.argsort(-baseline_scores)
    baseline_found = np.cumsum(y[baseline_queried[np.arange(opts.budget)]])
    logger.debug("baseline found:\n%s" % (str(list(baseline_found))))

    queried = []  # labeled instances

    for i in range(opts.budget):
        tm = Timer()
        a_scores = afss.get_weighted_scores(x, scores)
        ordered_indexes = np.argsort(-a_scores)
        items = get_first_vals_not_marked(ordered_indexes, queried, start=0, n=1)
        queried.extend(items)
        hf = np.array(queried, dtype=int)
        y_labeled[items] = y[items]
        afss.update_afss(x, y_labeled, hf, scores, tau=opts.afss_tau)
        if plot and ensemble.m < 5:
            xx, yy = plot_afss_scores(x, y, ensemble, afss, selected=x[hf], cmap='jet', xx=xx, yy=yy,
                                      name="_f%d_after" % (i+1), dataset=opts.dataset, outpath=opts.results_dir)
        logger.debug(tm.message("finished budget %d:" % (i+1)))

    if plot:
        xx, yy = plot_weighted_scores(x, y, ensemble, afss, selected=None, xx=xx, yy=yy, contour_levels=20,
                                      name="_feedback_after", n_anoms=opts.n_anoms,
                                      dataset=opts.dataset, outpath=opts.results_dir)

    afss.close_session()

    # the number of anomalies discovered within the budget while incorporating feedback
    # logger.debug("queried:\n%s" % str(queried))
    # logger.debug("y_labeled:\n%s" % str(list(y_labeled[queried])))
    found = np.cumsum(y[queried])
    logger.debug("GLAD found:\n%s" % (str(list(found))))

    # make queried indexes 1-indexed
    queried = np.array(queried, dtype=int) + 1
    baseline_queried = np.array(baseline_queried[0:opts.budget], dtype=int) + 1

    results = SequentialResults(num_seen=found, num_seen_baseline=baseline_found,
                                queried_indexes=queried,
                                queried_indexes_baseline=baseline_queried)
    return results


def glad_active_learn(opts):
    set_results_dir(opts)
    dir_create(opts.results_dir)

    opts.plot = opts.plot and opts.reruns == 1  # just in case...

    logger.debug("feedback budget: %d, batch_size: %d, afss_nodes: %d" %
                 (opts.budget, opts.train_batch_size, opts.afss_nodes))

    if opts.ensemble_type not in supported_ensemble_types:
        raise ValueError("Unsupported ensemble type '%s'. Supported ensemble types: %s." %
                         (opts.ensemble_type, str(supported_ensemble_types)))

    x, y = read_anomaly_dataset(opts.dataset)
    logger.debug("dataset: %s, shape: %s" % (opts.dataset, str(x.shape)))

    all_results = SequentialResults()
    orig_randseed = opts.randseed
    ensembles = []
    for i in range(opts.reruns):
        tm = Timer()

        opts.randseed = orig_randseed + i
        opts.runidx = i+1
        set_random_seeds(opts.randseed, opts.randseed+1, opts.randseed+2)

        ensemble = prepare_loda_ensemble(x, mink=opts.loda_mink, maxk=opts.loda_maxk,
                                         debug=opts.loda_debug and x.shape[1] == 2, m=4)
        logger.debug("#LODA projections: %d" % ensemble.m)

        ensembles.append(ensemble)

        if not opts.ensemble_only:
            results = afss_active_learn_ensemble(x, y, ensemble, opts)
            all_results.merge(results)

        logger.debug(tm.message("completed run %d/%d:" % (i+1, opts.reruns)))

    if not opts.ensemble_only:
        all_results.write_to_csv(opts)
    else:
        all_results = None

    return x, y, all_results, ensembles


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/glad")

    args = get_glad_command_args(debug=False, debug_args=["--debug",
                                                          "--plot",
                                                          "--dataset=toy",
                                                          "--budget=1",
                                                          "--n_anoms=30",
                                                          "--log_file=temp/glad/glad_batch.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    opts = GladOpts(args)
    logger.debug("running: %s" % opts.str_opts())

    glad_active_learn(opts)
