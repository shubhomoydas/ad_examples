from .glad_batch import *
from .glad_test_support import *

"""
python -m glad.test_glad --log_file=temp/glad/test_glad.log --debug --dataset=toy2 --n_anoms=60 --loda_debug --plot --op=unit

python -m glad.test_glad --log_file=temp/glad/test_glad.log --debug --dataset=toy2 --budget=30 --loda_debug --plot --op=active
"""


def test_glad(opts):
    set_results_dir(opts)
    dir_create(opts.results_dir)

    set_random_seeds(opts.randseed, opts.randseed + 1, opts.randseed + 2)

    x, y = read_anomaly_dataset(opts.dataset, datafile=opts.datafile)
    ensemble = prepare_loda_ensemble(x, mink=opts.loda_mink, maxk=opts.loda_maxk,
                                     debug=opts.loda_debug and x.shape[1] == 2, m=4)

    xx = yy = None
    hf, _ = get_top_ranked_instances(x, ensemble, n=opts.n_anoms)

    if opts.plot:
        xx, yy = plot_ensemble_scores(x, y, ensemble, selected=None, dataset=opts.dataset, outpath=opts.results_dir)

    afss = get_afss_model(opts, n_output=ensemble.m)

    afss.init_network(x, prime_network=True)

    if opts.debug: afss.log_probability_ranges(x)  # DEBUG

    hf_scores_before = afss.get_weighted_scores(x[hf], ensemble.get_scores(x[hf]))

    if opts.plot:
        xx, yy = plot_weighted_scores(x, y, ensemble, afss, selected=None, xx=xx, yy=yy, contour_levels=20, name="_before",
                                      n_anoms=opts.n_anoms, dataset=opts.dataset, outpath=opts.results_dir)

        # selected=x[hf]
        selected=None
        xx, yy = plot_afss_scores(x, y, ensemble, afss, selected=selected, plot_ensemble=False, cmap='jet', xx=xx, yy=yy,
                                  name="_before", dataset=opts.dataset, outpath=opts.results_dir)

        xx, yy = plot_afss_scores(x, y, ensemble, afss, selected=None, plot_ensemble=True, cmap='jet', xx=xx, yy=yy,
                                  name="_ensemble", dataset=opts.dataset, outpath=opts.results_dir)

    scores = ensemble.get_scores(x)
    logger.debug("scores: %s" % (str(scores.shape)))

    for i in range(10):
        # run update a few times so that the parameters stabilize
        # Note: the parameters and the tau-th ranked instance and it's score
        # have to be learned. This can only be possible if performed iteratively:
        #   first, learn the tau-th ranked instance using current parameters
        #   second, learn the parameters given the tau-th ranked instance
        afss.update_afss(x, y, hf, scores, tau=opts.afss_tau)

    if opts.debug: afss.log_probability_ranges(x)  # DEBUG

    hf_scores_after = afss.get_weighted_scores(x[hf], ensemble.get_scores(x[hf]))

    logger.debug("hf_scores_before:\n%s\nhf_scores_after:\n%s\ny:\n%s" %
                 (str(list(hf_scores_before)), str(list(hf_scores_after)), str(list(y[hf]))))

    if opts.plot:
        xx, yy = plot_weighted_scores(x, y, ensemble, afss, selected=None, xx=xx, yy=yy, contour_levels=20, name="_after",
                                      n_anoms=opts.n_anoms, dataset=opts.dataset, outpath=opts.results_dir)

        xx, yy = plot_afss_scores(x, y, ensemble, afss, selected=None, plot_ensemble=False, cmap='jet', xx=xx, yy=yy,
                                  name="_after", dataset=opts.dataset, outpath=opts.results_dir)

    afss.close_session()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/glad")

    args = get_glad_command_args(debug=False, debug_args=["--debug",
                                                          "--plot",
                                                          "--dataset=toy",
                                                          "--budget=1",
                                                          "--n_anoms=30",
                                                          "--log_file=temp/glad/test_glad.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    opts = GladOpts(args)
    logger.debug("running: %s" % opts.str_opts())

    # test_get_afss_batches(args)
    if opts.op == "unit":
        test_glad(opts)
    elif opts.op == "active":
        glad_active_learn(opts)
    elif opts.op == "loda":
        test_loda(opts)
    else:
        print("invalid algo: '%s'. options: [unit|active|loda]" % opts.op)
