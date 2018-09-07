import gzip

from common.utils import *
from common.metrics import *
from aad.aad_base import *
from aad.query_model import *
from aad.aad_loss import *

from aad.forest_aad_detector import *
from aad.loda_aad import *
from aad.precomputed_aad import *


def get_aad_model(x, opts, random_state=None):
    if opts.detector_type == LODA:
        model = AadLoda(sparsity=opts.sparsity, mink=opts.mink, maxk=opts.maxk)
    elif is_forest_detector(opts.detector_type):
        model = AadForest(n_estimators=opts.forest_n_trees,
                          max_samples=min(opts.forest_n_samples, x.shape[0]),
                          score_type=opts.forest_score_type, random_state=random_state,
                          add_leaf_nodes_only=opts.forest_add_leaf_nodes_only,
                          max_depth=opts.forest_max_depth,
                          ensemble_score=opts.ensemble_score,
                          detector_type=opts.detector_type, n_jobs=opts.n_jobs,
                          tree_update_type=opts.tree_update_type,
                          forest_replace_frac=opts.forest_replace_frac,
                          feature_partitions=opts.feature_partitions)
    elif opts.detector_type == PRECOMPUTED_SCORES:
        model = AadPrecomputed(opts, random_state=random_state)
    else:
        raise ValueError("Unsupported ensemble")
    return model


class SequentialResults(object):
    def __init__(self, num_seen=None, num_not_seen=None, num_seen_baseline=None,
                 true_queried_indexes=None, true_queried_indexes_baseline=None,
                 stream_window=None, stream_window_baseline=None,
                 aucs=None):
        self.num_seen = num_seen
        self.num_not_seen = num_not_seen
        self.num_seen_baseline = num_seen_baseline
        self.true_queried_indexes = true_queried_indexes
        self.true_queried_indexes_baseline = true_queried_indexes_baseline
        self.stream_window = stream_window
        self.stream_window_baseline = stream_window_baseline
        self.aucs = aucs


def summarize_aad_metrics(ensembles, metrics_struct):
    nqueried = len(metrics_struct.metrics[0][0].queried)
    num_seen = np.zeros(shape=(0, nqueried+2))
    num_seen_baseline = np.zeros(shape=(0, nqueried+2))
    true_queried_indexes = np.zeros(shape=(0, nqueried+2))
    true_queried_indexes_baseline = np.zeros(shape=(0, nqueried + 2))
    for i in range(len(metrics_struct.metrics)):
        # file level
        submetrics = metrics_struct.metrics[i]
        subensemble = ensembles[i]
        for j in range(len(submetrics)):
            # rerun level
            queried = submetrics[j].queried
            lbls = subensemble[j].labels

            nseen = np.zeros(shape=(1, nqueried+2))
            nseen[0, 0:2] = [metrics_struct.fids[i], metrics_struct.runidxs[j]]
            nseen[0, 2:(nseen.shape[1])] = np.cumsum(lbls[queried])
            num_seen = rbind(num_seen, nseen)

            qlbls = subensemble[j].labels[subensemble[j].ordered_anom_idxs[0:nqueried]]
            nseen = np.zeros(shape=(1, nqueried+2))
            nseen[0, 0:2] = [metrics_struct.fids[i], metrics_struct.runidxs[j]]
            nseen[0, 2:(nseen.shape[1])] = np.cumsum(qlbls)
            num_seen_baseline = rbind(num_seen_baseline, nseen)

            # the ensembles store samples in sorted order of default anomaly
            # scores. The corresponding indexes are stored in ensemble.original_indexes
            t_idx = np.zeros(shape=(1, nqueried + 2))
            t_idx[0, 0:2] = [metrics_struct.fids[i], metrics_struct.runidxs[j]]
            t_idx[0, 2:(t_idx.shape[1])] = subensemble[j].original_indexes[queried]
            # Note: make the queried indexes realive 1 (NOT zero)
            true_queried_indexes = rbind(true_queried_indexes, t_idx + 1)

            # the ensembles store samples in sorted order of default anomaly
            # scores. The corresponding indexes are stored in ensemble.original_indexes
            b_idx = np.zeros(shape=(1, nqueried + 2))
            b_idx[0, 0:2] = [metrics_struct.fids[i], metrics_struct.runidxs[j]]
            b_idx[0, 2:(b_idx.shape[1])] = subensemble[j].original_indexes[np.arange(nqueried)]
            # Note: make the queried indexes realive 1 (NOT zero)
            true_queried_indexes_baseline = rbind(true_queried_indexes_baseline, b_idx + 1)

    return SequentialResults(num_seen=num_seen, num_seen_baseline=num_seen_baseline,
                             true_queried_indexes=true_queried_indexes,
                             true_queried_indexes_baseline=true_queried_indexes_baseline)


def save_aad_summary(alad_summary, opts):
    cansave = opts.resultsdir != "" and os.path.isdir(opts.resultsdir)
    if cansave:
        save(alad_summary, filepath=opts.get_metrics_summary_path())


def get_score_ranges(x, w):
    s = x.dot(w)
    qvals = list()
    qvals.append(np.min(s))
    for i in range(1, 10):
        qvals.append(quantile(s, (i * 10.0)))
    qvals.append(np.max(s))
    return qvals


def get_linear_score_variance(x, w):
    indxs = x.nonzero()[1]  # column indexes
    x_ = x[0, indxs].todense()
    xw = np.array(x_) * w[indxs]
    # xw = x_.reshape(-1, 1) * w[indxs]
    # logger.debug("xw:\n%s" % str(list(xw)))
    #xw = np.array(x[0, indxs].multiply(w[indxs]))
    #xw_mean = xw.mean(axis=1)[0]
    #xw_sq = xw ** 2
    #var = xw_sq.mean(axis=1)[0] - xw_mean ** 2
    var = np.var(xw)
    score = np.sum(xw)
    if False:
        s = x.dot(w)
        if s != score:
            logger.debug("size of x: %s" % str(x.shape))
            logger.debug("x_: %s" % str(list(x_)))
            logger.debug("w : %s" % str(list(w[indxs])))
            logger.debug("xw: %s" % str(list(xw)))
            raise ArithmeticError("s=%f != score=%f" % (s, score))
    return score, var


def get_closest_indexes(inst, test_set, num=1, dest_set=None):
    n = test_set.shape[0]
    dists = np.zeros(n)
    for i in np.arange(n):
        ts = test_set[i, :]
        if ts.shape[0] > 1:
            # dense matrix
            ts = matrix(ts, nrow=1)
            diff = inst - ts
            dist = np.sum(diff**2)
        else:
            # sparse matrix
            diff = inst - ts
            tmp = diff * diff.T
            if tmp.shape[0] != 1:
                raise ValueError("dot product is %s" % str(tmp.shape))
            dist = tmp[0, 0]
        dists[i] = dist
    ordered = np.argsort(dists)[np.arange(num)]
    if False:
        logger.debug("last ts:\n%s" % str(ts))
        logger.debug("last diff:\n%s" % str(diff))
        logger.debug("ordered indexes: %s" % str(list(ordered)))
        logger.debug("dists: %s" % str(list(dists[ordered])))
        # logger.debug("dists: %s" % str(list(dists)))
        logger.debug("inst:\n%s" % str(inst))
        logger.debug("points:\n%s" % str(test_set[ordered, :]))
        ts = test_set[ordered[1], :]
        ts = matrix(ts, nrow=1)
        logger.debug("dist 2:\n%s" % str(np.sum((inst - ts)**2)))
    if dest_set is not None:
        for indx in ordered:
            dest_set.add(indx)
    return ordered


def get_score_variances(x, w, n_test, ordered_indexes=None, queried_indexes=None,
                        test_indexes=None,
                        eval_set=None, n_closest=9):
    if test_indexes is None:
        n_test = min(x.shape[0], n_test)
        top_ranked_indexes = ordered_indexes[np.arange(len(queried_indexes) + n_test)]
        tmp = np.array(SetList(top_ranked_indexes) - SetList(queried_indexes))
        test = tmp[np.arange(n_test)]
        # logger.debug("test:\n%s" % str(list(test)))
    else:
        test = test_indexes
        n_test = len(test)

    tm = Timer()
    vars = np.zeros(len(test))
    means = np.zeros(len(test))
    for i, idx in enumerate(test):
        means[i], vars[i] = get_linear_score_variance(x[idx], w)
    # logger.debug(tm.message("Time for score variance computation on test set:"))

    v_eval = None
    m_eval = None
    if eval_set is not None:
        tm = Timer()
        v_eval = np.zeros(eval_set.shape[0], dtype=float)
        m_eval = np.zeros(eval_set.shape[0], dtype=float)
        closest_indexes = set()  # all indexes from test_set that are closest to any unlabeled instances
        for i in range(n_test):
            test_index = test[i]
            get_closest_indexes(x[test_index, :], eval_set, num=n_closest, dest_set=closest_indexes)
        logger.debug("# Closest: %d" % len(closest_indexes))
        for i, idx in enumerate(closest_indexes):
            m_eval[idx], v_eval[idx] = get_linear_score_variance(eval_set[idx, :], w)
        logger.debug(tm.message("Time for score variance computation on eval set:"))

    return means, vars, test, v_eval, m_eval


def get_queried_indexes(scores, labels, opts):
    # logger.debug("computing queried indexes...")
    queried = np.argsort(-scores)[0:opts.budget]
    num_seen = np.cumsum(labels[queried[np.arange(opts.budget)]])
    return num_seen, queried


def write_baseline_query_indexes(queried_info, opts):
    logger.debug("writing baseline queries...")
    queried = np.zeros(shape=(len(queried_info), opts.budget + 2), dtype=int)
    num_seen = np.zeros(shape=(len(queried_info), opts.budget + 2), dtype=int)
    for i, info in enumerate(queried_info):
        num_seen[i, 2:(opts.budget + 2)] = info[0]
        num_seen[i, 0] = 1
        queried[i, 2:(opts.budget + 2)] = info[1] + 1  # make indexes relative 1, *not* 0
        queried[i, 0] = 1
    prefix = opts.get_alad_metrics_name_prefix()
    baseline_file = os.path.join(opts.resultsdir, "%s-baseline.csv" % (prefix,))
    # np.savetxt(baseline_file, num_seen, fmt='%d', delimiter=',')
    queried_idxs_baseline_file = os.path.join(opts.resultsdir, "%s-queried-baseline.csv" % (prefix,))
    np.savetxt(queried_idxs_baseline_file, queried, fmt='%d', delimiter=',')


def write_sequential_results_to_csv(results, opts):
    """

    :param results: SequentialResults
    :param opts: AadOpts
    :return:
    """
    prefix = opts.get_alad_metrics_name_prefix()
    num_seen_file = os.path.join(opts.resultsdir, "%s-num_seen.csv" % (prefix,))
    num_not_seen_file = os.path.join(opts.resultsdir, "%s-num_not_seen.csv" % (prefix,))
    num_total_anoms_file = os.path.join(opts.resultsdir, "%s-num_total_anoms.csv" % (prefix,))
    baseline_file = os.path.join(opts.resultsdir, "%s-baseline.csv" % (prefix,))
    stream_window_file = os.path.join(opts.resultsdir, "%s-window.csv" % (prefix,))
    stream_window_baseline_file = os.path.join(opts.resultsdir, "%s-window-baseline.csv" % (prefix,))
    queried_idxs_file = os.path.join(opts.resultsdir, "%s-queried.csv" % (prefix,))
    queried_idxs_baseline_file = os.path.join(opts.resultsdir, "%s-queried-baseline.csv" % (prefix,))
    aucs_file = os.path.join(opts.resultsdir, "%s-aucs.csv" % (prefix,))
    if results.num_seen is not None:
        np.savetxt(num_seen_file, results.num_seen, fmt='%d', delimiter=',')
    if results.num_not_seen is not None:
        np.savetxt(num_not_seen_file, results.num_not_seen, fmt='%d', delimiter=',')
        tmp = np.copy(results.num_seen)
        tmp[:, 2:tmp.shape[1]] += results.num_not_seen[:, 2:results.num_not_seen.shape[1]]
        np.savetxt(num_total_anoms_file, tmp, fmt='%d', delimiter=',')
    if results.num_seen_baseline is not None:
        np.savetxt(baseline_file, results.num_seen_baseline, fmt='%d', delimiter=',')
    if results.true_queried_indexes is not None:
        np.savetxt(queried_idxs_file, results.true_queried_indexes, fmt='%d', delimiter=',')
    if results.true_queried_indexes_baseline is not None:
        np.savetxt(queried_idxs_baseline_file, results.true_queried_indexes_baseline, fmt='%d', delimiter=',')
    if results.stream_window is not None:
        np.savetxt(stream_window_file, results.stream_window, fmt='%d', delimiter=',')
    if results.stream_window_baseline is not None:
        np.savetxt(stream_window_baseline_file, results.stream_window_baseline, fmt='%d', delimiter=',')
    if results.aucs is not None:
        np.savetxt(aucs_file, results.aucs, fmt='%f', delimiter=',')


def summarize_ensemble_num_seen(ensemble, metrics, fid=0, runidx=0):
    """
    IMPORTANT: returned queried_indexes and queried_indexes_baseline are 1-indexed (NOT 0-indexed)
    """
    nqueried = len(metrics.queried)
    num_seen = np.zeros(shape=(1, nqueried + 2))
    num_seen_baseline = np.zeros(shape=(1, nqueried + 2))

    num_seen[0, 0:2] = [fid, runidx]
    num_seen[0, 2:(num_seen.shape[1])] = np.cumsum(ensemble.labels[metrics.queried])

    queried_baseline = ensemble.ordered_anom_idxs[0:nqueried]
    qlbls = ensemble.labels[queried_baseline]
    num_seen_baseline[0, 0:2] = [fid, runidx]
    num_seen_baseline[0, 2:(num_seen_baseline.shape[1])] = np.cumsum(qlbls)

    # the ensembles store samples in sorted order of default anomaly
    # scores. The corresponding indexes are stored in ensemble.original_indexes
    true_queried_indexes = np.zeros(shape=(1, nqueried + 2))
    true_queried_indexes[0, 0:2] = [fid, runidx]
    # Note: make the queried indexes relative 1 (NOT zero)
    true_queried_indexes[0, 2:(true_queried_indexes.shape[1])] = ensemble.original_indexes[metrics.queried] + 1

    true_queried_indexes_baseline = np.zeros(shape=(1, nqueried + 2))
    true_queried_indexes_baseline[0, 0:2] = [fid, runidx]
    # Note: make the queried indexes relative 1 (NOT zero)
    true_queried_indexes_baseline[0, 2:(true_queried_indexes_baseline.shape[1])] = \
        queried_baseline + 1

    return num_seen, num_seen_baseline, true_queried_indexes, true_queried_indexes_baseline


def write_sparsemat_to_file(fname, X, fmt='%.18e', delimiter=','):
    if isinstance(X, np.ndarray):
        np.savetxt(fname, X, fmt='%3.2f', delimiter=",")
    elif isinstance(X, csr_matrix):
        f = open(fname, 'w')
        for i in range(X.shape[0]):
            a = X[i, :].toarray()[0]
            f.write(delimiter.join([fmt % v for v in a]))
            f.write(os.linesep)
            if (i + 1) % 10 == 0:
                f.flush()
        f.close()
    else:
        raise ValueError("Invalid matrix type")


def save_aad_model(filepath, model):
    import cPickle
    f = gzip.open(filepath, 'wb')
    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def load_aad_model(filepath):
    import cPickle
    f = gzip.open(filepath, 'rb')
    model = cPickle.load(f)
    f.close()
    return model


def save_aad_metrics(metrics, opts):
    cansave = (opts.resultsdir != "" and os.path.isdir(opts.resultsdir))
    if cansave:
        save(metrics, filepath=opts.get_metrics_path())


def load_aad_metrics(opts):
    metrics = None
    fpath = opts.get_metrics_path()
    canload = (opts.resultsdir != "" and os.path.isfile(fpath))
    if canload:
        # print "Loading metrics" + fpath
        metrics = load(fpath)
    else:
        print ("Cannot load %s" % fpath)
    return metrics

