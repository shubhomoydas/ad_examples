import os
import numpy as np

import logging

from common.utils import *
from aad.aad_globals import *
from aad.aad_support import *

from forest_aad_detector import *
from data_stream import *


class StreamingAnomalyDetector(object):
    """
    Attributes:
        model: AadForest
        stream: DataStream
        max_buffer: int
            Determines the window size
        buffer_instances_x: list
    """
    def __init__(self, stream, model, labeled_x=None, labeled_y=None,
                 unlabeled_x=None, unlabeled_y=None, opts=None, max_buffer=512):
        self.model = model
        self.stream = stream
        self.max_buffer = max_buffer
        self.opts = opts
        self.retention_type = opts.retention_type

        self.buffer_x = None
        self.buffer_y = None

        self.unlabeled_x = unlabeled_x
        self.unlabeled_y = unlabeled_y

        self.labeled_x = labeled_x
        self.labeled_y = labeled_y

        self.qstate = None

    def reset_buffer(self):
        self.buffer_x = None
        self.buffer_y = None

    def add_buffer_xy(self, x, y):
        if self.buffer_x is None:
            self.buffer_x = x
        else:
            self.buffer_x = rbind(self.buffer_x, x)

        if self.buffer_y is None:
            self.buffer_y = y
        else:
            if y is not None:
                self.buffer_y = append(self.buffer_y, y)

    def move_buffer_to_unlabeled(self):
        if self.retention_type == STREAM_RETENTION_OVERWRITE:
            missed = int(np.sum(self.unlabeled_y)) if self.unlabeled_y is not None else 0
            retained = int(np.sum(self.buffer_y)) if self.buffer_y is not None else 0
            # logger.debug("[overwriting] true anomalies: missed(%d), retained(%d)" % (missed, retained))
            self.unlabeled_x = self.buffer_x
            self.unlabeled_y = self.buffer_y
        elif self.retention_type == STREAM_RETENTION_TOP_ANOMALOUS:
            # retain the top anomalous instances from the merged
            # set of instance from both buffer and current unlabeled.
            tmp_x = self.unlabeled_x
            tmp_y = self.unlabeled_y
            if self.buffer_x is not None:
                tmp_x = np.vstack([tmp_x, self.buffer_x])
                tmp_y = np.append(tmp_y, self.buffer_y)
            n = min(tmp_x.shape[0], self.max_buffer)
            new_x = self.model.transform_to_region_features(tmp_x)
            idxs, scores = self.model.order_by_score(new_x)
            self.unlabeled_x = tmp_x[idxs[np.arange(n)]]
            self.unlabeled_y = tmp_y[idxs[np.arange(n)]]
            if n < len(tmp_y):
                missedidxs = idxs[n:len(tmp_y)]
            else:
                missedidxs = None
            missed = int(np.sum(tmp_y[missedidxs])) if missedidxs is not None else 0
            retained = int(np.sum(self.unlabeled_y)) if self.unlabeled_y is not None else 0
            # logger.debug("[top anomalous] true anomalies: missed(%d), retained(%d)" % (missed, retained))
        self.reset_buffer()

    def get_num_instances(self):
        """Returns the total number of labeled and unlabeled instances that will be used for weight inference"""
        n = 0
        if self.unlabeled_x is not None:
            n += nrow(self.unlabeled_x)
        if self.labeled_x is not None:
            # logger.debug("labeled_x: %s" % str(self.labeled_x.shape))
            n += nrow(self.labeled_x)
        return n

    def init_query_state(self, opts):
        n = self.get_num_instances()
        bt = get_budget_topK(n, opts)
        self.qstate = Query.get_initial_query_state(opts.qtype, opts=opts, qrank=bt.topK,
                                                    a=1., b=1., budget=bt.budget)

    def get_next_from_stream(self, n=0):
        if n == 0:
            n = self.max_buffer
        x, y = self.stream.read_next_from_stream(n)

        if x is None:
            return x, y

        if False:
            if self.buffer_x is not None:
                logger.debug("buffer shape: %s" % str(self.buffer_x.shape))
            logger.debug("x.shape: %s" % str(x.shape))

        self.add_buffer_xy(x, y)

        self.model.add_samples(x, current=False)

        return x, y

    def update_model_from_buffer(self):
        self.model.update_model_from_stream_buffer()

    def get_next_transformed(self, n=1):
        x, y = self.get_next_from_stream(n)
        if x is None:
            return x, y
        x_new = self.model.transform_to_region_features(x, dense=False)
        return x_new, y

    def stream_buffer_empty(self):
        return self.stream.empty()

    def get_anomaly_scores(self, x):
        x_new = self.model.transform_to_region_features(x, dense=False)
        scores = self.model.get_score(x_new)
        return scores

    def setup_data_for_feedback(self):
        """
        Prepares the input matrices/data structures for weight update. The format
        is such that the top rows of data matrix are labeled and below are unlabeled.

        :return: (np.ndarray, np.array, np.array, np.array)
            (x, y, ha, hn)
            x - data matrix, y - labels (np.nan for unlabeled),
            ha - indexes of labeled anomalies, hn - indexes of labeled nominals
        """
        x = None
        y = None
        if self.labeled_x is not None:
            x = self.labeled_x.copy()
            y = self.labeled_y.copy()
            ha = np.where(self.labeled_y == 1)[0]
            hn = np.where(self.labeled_y == 0)[0]
        else:
            ha = np.zeros(0, dtype=int)
            hn = np.zeros(0, dtype=int)
        if self.unlabeled_x is not None:
            if x is None:
                x = self.unlabeled_x.copy()
            else:
                x = np.append(x, self.unlabeled_x, axis=0)
            if self.unlabeled_y is not None:
                if y is not None:
                    y = np.append(y, self.unlabeled_y)
                else:
                    y = self.unlabeled_y.copy()
            else:
                if y is not None:
                    y = np.append(y, np.ones(nrow(self.unlabeled_x), dtype=int) * -1)
                else:
                    y = np.ones(nrow(self.unlabeled_x), dtype=int) * -1
        if False:
            logger.debug("x: %d, y: %d, ha: %d, hn:%d" % (nrow(x), len(y), len(ha), len(hn)))
        return x, y, ha, hn

    def get_instance_stats(self):
        nha = nhn = nul = 0
        if self.labeled_y is not None:
            nha = len(np.where(self.labeled_y == 1)[0])
            nhn = len(np.where(self.labeled_y == 0)[0])
        if self.unlabeled_x is not None:
            nul = nrow(self.unlabeled_x)
        return nha, nhn, nul

    def get_num_labeled(self):
        """Returns the number of instances for which we already have label feedback"""
        if self.labeled_y is not None:
            return len(self.labeled_y)
        return 0

    def get_query_data(self, x=None, y=None, ha=None, hn=None, unl=None, w=None, n_query=1):
        """Returns the best instance that should be queried, along with other data structures

        Args:
            x: np.ndarray
                input instances (labeled + unlabeled)
            y: np.array
                labels for instances which are already labeled, else some dummy values
            ha: np.array
                indexes of labeled anomalies
            hn: np.array
                indexes of labeled nominals
            unl: np.array
                unlabeled instances that should be ignored for query
            w: np.array
                current weight vector
            n_query: int
                number of instances to query
        """
        n = self.get_num_instances()
        n_feedback = self.get_num_labeled()
        if False:
            logger.debug("get_query_data() n: %d, n_feedback: %d" % (n, n_feedback))
        if n == 0:
            raise ValueError("No instances available")
        if x is None:
            x, y, ha, hn = self.setup_data_for_feedback()
        if w is None:
            w = self.model.w
        if unl is None:
            unl = np.zeros(0, dtype=int)
        # the top n_feedback instances in the instance list are the labeled items
        queried_items = append(np.arange(n_feedback), unl)
        x_transformed = self.model.transform_to_region_features(x, dense=False)
        order_anom_idxs, anom_score = self.model.order_by_score(x_transformed)
        xi = self.qstate.get_next_query(maxpos=n, ordered_indexes=order_anom_idxs,
                                        queried_items=queried_items,
                                        x=x_transformed, lbls=y, anom_score=anom_score,
                                        w=w, hf=append(ha, hn),
                                        remaining_budget=self.opts.budget - n_feedback,
                                        n=n_query)
        if False:
            logger.debug("ordered instances[%d]: %s\nha: %s\nhn: %s\nxi: %s" %
                         (self.opts.budget, str(list(order_anom_idxs[0:self.opts.budget])),
                          str(list(ha)), str(list(hn)), str(list(xi))))
        return xi, x, y, x_transformed, ha, hn, order_anom_idxs, anom_score

    def move_unlabeled_to_labeled(self, xi, yi):
        unlabeled_idx = xi - self.get_num_labeled()

        self.labeled_x = rbind(self.labeled_x, matrix(self.unlabeled_x[unlabeled_idx], nrow=1))
        if self.labeled_y is None:
            self.labeled_y = np.array([yi], dtype=int)
        else:
            self.labeled_y = np.append(self.labeled_y, [yi])
        mask = np.ones(self.unlabeled_x.shape[0], dtype=bool)
        mask[unlabeled_idx] = False
        self.unlabeled_x = self.unlabeled_x[mask]
        self.unlabeled_y = self.unlabeled_y[mask]

    def update_weights_with_feedback(self, xi, yi, x, y, x_transformed, ha, hn, opts):
        """Relearns the optimal weights from feedback and updates internal labeled and unlabeled matrices

        IMPORTANT:
            This API assumes that the input x, y, x_transformed are consistent with
            the internal labeled/unlabeled matrices, i.e., the top rows/values in
            these matrices are from labeled data and bottom ones are from internally
            stored unlabeled data.
        """

        # Add the newly labeled instance to the corresponding list of labeled
        # instances and remove it from the unlabeled set.
        self.move_unlabeled_to_labeled(xi, yi)

        if yi == 1:
            ha = append(ha, [xi])
        else:
            hn = append(hn, [xi])

        self.model.update_weights(x_transformed, y, ha, hn, opts)

    def get_score_variance(self, x, n_instances, opts, transform=False):
        """Computes variance in scores of top ranked instances
        """
        w = self.model.w
        if w is None:
            raise ValueError("Model not trained")
        if transform:
            x = self.model.transform_to_region_features(x, dense=False)
        ordered_indexes, scores = self.model.order_by_score(x, w=w)
        bt = get_budget_topK(n_instances, opts)
        tn = min(10, nrow(x))
        vars = np.zeros(tn, dtype=float)
        for i in np.arange(tn):
            vars[i] = get_linear_score_variance(x[ordered_indexes[i], :], w)
        # logger.debug("top %d vars:\n%s" % (tn, str(list(vars))))
        return vars


def get_rearranging_indexes(add_pos, move_pos, n):
    """Creates an array 0...n-1 and moves value at 'move_pos' to 'add_pos', and shifts others back

    Useful to reorder data when we want to move instances from unlabeled set to labeled.
    TODO:
        Use this to optimize the API StreamingAnomalyDetector.get_query_data()
        since it needs to repeatedly convert the data to transformed [node] features.

    Example:
        get_rearranging_indexes(2, 2, 10):
            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        get_rearranging_indexes(0, 1, 10):
            array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])

        get_rearranging_indexes(2, 9, 10):
            array([0, 1, 9, 2, 3, 4, 5, 6, 7, 8])

    :param add_pos:
    :param move_pos:
    :param n:
    :return:
    """
    if add_pos > move_pos:
        raise ValueError("add_pos must be less or equal to move_pos")
    rearr_idxs = np.arange(n)
    if add_pos == move_pos:
        return rearr_idxs
    rearr_idxs[(add_pos + 1):(move_pos + 1)] = rearr_idxs[add_pos:move_pos]
    rearr_idxs[add_pos] = move_pos
    return rearr_idxs


def read_data(opts):
    data = read_csv(opts.datafile, header=0, sep=',')
    X_train = np.zeros(shape=(data.shape[0], data.shape[1] - 1))
    for i in range(X_train.shape[1]):
        X_train[:, i] = data.iloc[:, i + 1]
    labels = np.array([1 if data.iloc[i, 0] == "anomaly" else 0 for i in range(data.shape[0])], dtype=int)
    return X_train, labels


def train_aad_model(opts, X_train):
    rng = np.random.RandomState(opts.randseed + opts.fid * opts.reruns + opts.runidx)
    # fit the model
    model = AadForest(n_estimators=opts.forest_n_trees,
                      max_samples=min(opts.forest_n_samples, X_train.shape[0]),
                      score_type=opts.forest_score_type, random_state=rng,
                      add_leaf_nodes_only=opts.forest_add_leaf_nodes_only,
                      max_depth=opts.forest_max_depth,
                      ensemble_score=opts.ensemble_score,
                      detector_type=opts.detector_type, n_jobs=opts.n_jobs)
    model.fit(X_train)
    model.init_weights(init_type=opts.init)
    return model


def prepare_aad_model(X, y, opts):
    if opts.load_model and opts.modelfile != "" and os.path.isfile(opts.modelfile):
        logger.debug("Loading model from file %s" % opts.modelfile)
        model = load_aad_model(opts.modelfile)
    else:
        model = train_aad_model(opts, X)

    logger.debug("total #nodes: %d" % (len(model.all_regions)))
    if False:
        if model.w is not None:
            logger.debug("w:\n%s" % str(list(model.w)))
        else:
            logger.debug("model weights are not set")
    return model


def run_feedback(sad, min_feedback, max_feedback, opts):
    """

    :param sad: StreamingAnomalyDetector
    :param max_feedback: int
    :param opts: Opts
    :return:
    """

    if False:
        # get baseline metrics
        x_transformed = sad.model.transform_to_region_features(sad.unlabeled_x, dense=False)
        ordered_idxs, _ = sad.model.order_by_score(x_transformed)
        seen_baseline = sad.unlabeled_y[ordered_idxs[0:max_feedback]]
        num_seen_baseline = np.cumsum(seen_baseline)
        logger.debug("num_seen_baseline:\n%s" % str(list(num_seen_baseline)))

    # baseline scores
    w_baseline = sad.model.get_uniform_weights()
    x_transformed_baseline = sad.model.transform_to_region_features(sad.unlabeled_x, dense=False)
    order_baseline, scores_baseline = sad.model.order_by_score(x_transformed_baseline, w_baseline)
    n_seen_baseline = min(max_feedback, len(sad.unlabeled_y))
    queried_baseline = order_baseline[0:n_seen_baseline]
    seen_baseline = sad.unlabeled_y[queried_baseline]
    # seen_baseline = min(max_feedback, len(sad.unlabeled_y))
    # found_baseline = np.sum(sad.unlabeled_y[order_baseline[0:seen_baseline]])

    seen = np.zeros(0, dtype=int)
    n_unlabeled = np.zeros(0, dtype=int)
    queried = np.zeros(0, dtype=int)
    unl = np.zeros(0, dtype=int)
    i = 0
    while i < max_feedback:
        i += 1
        # scores based on current weights
        xi_, x, y, x_transformed, ha, hn, order_anom_idxs, anom_score = \
            sad.get_query_data(unl=unl, n_query=max_feedback)

        order_anom_idxs_minus_ha_hn = get_first_vals_not_marked(
            order_anom_idxs, append(ha, hn), n=len(order_anom_idxs))

        bt = get_budget_topK(x_transformed.shape[0], opts)

        # Note: We will ensure that the tau-th instance is atleast 10-th (or lower) ranked
        tau_rank = min(max(bt.topK, 10), x.shape[0])

        xi = xi_[0]
        means = vars = qpos = m_tau = v_tau = None
        if opts.query_confident:
            # get the mean score and its variance for the top ranked instances
            # excluding the instances which have already been queried
            means, vars, test, v_eval, _ = get_score_variances(x_transformed, sad.model.w,
                                                               n_test=tau_rank,
                                                               ordered_indexes=order_anom_idxs,
                                                               queried_indexes=append(ha, hn))
            # get the mean score and its variance for the tau-th ranked instance
            m_tau, v_tau, _, _, _ = get_score_variances(x_transformed[order_anom_idxs_minus_ha_hn[tau_rank]],
                                                        sad.model.w, n_test=1,
                                                        test_indexes=np.array([0], dtype=int))
            qpos = np.where(test == xi)[0]  # top-most ranked instance

        if False and opts.query_confident:
            logger.debug("tau score:\n%s (%s)" % (str(list(m_tau)), str(list(v_tau))))
            strmv = ",".join(["%f (%f)" % (means[j], vars[j]) for j in np.arange(len(means))])
            logger.debug("scores:\n%s" % strmv)

        # check if we are confident that this is larger than the tau-th ranked instance
        if (not opts.query_confident) or (i <= min_feedback or
                                          means[qpos] - 3. * np.sqrt(vars[qpos]) >= m_tau):
            seen = append(seen, [y[xi]])
            queried = append(queried, xi)
            # seen += 1
            # found += y[xi]
            tm_update = Timer()
            sad.update_weights_with_feedback(xi, y[xi], x, y, x_transformed, ha, hn, opts)
            tm_update.end()
            # reset the list of queried test instances because their scores would have changed
            unl = np.zeros(0, dtype=int)
            if False:
                nha, nhn, nul = sad.get_instance_stats()
                # logger.debug("xi:%d, test indxs: %s, qpos: %d" % (xi, str(list(test)), qpos))
                # logger.debug("orig scores:\n%s" % str(list(anom_score[order_anom_idxs[0:tau_rank]])))
                logger.debug("[%d] #feedback: %d; ha: %d; hn: %d, mnw: %d, mxw: %d; update: %f sec(s)" %
                             (i, nha + nhn, nha, nhn, min_feedback, max_feedback, tm_update.elapsed()))
        else:
            # ignore this instance from query
            unl = append(unl, [xi])
            # logger.debug("skipping feedback for xi=%d at iter %d; unl: %s" % (xi, i, str(list(unl))))
            # continue
        n_unlabeled = np.append(n_unlabeled, [int(np.sum(sad.unlabeled_y))])
        # logger.debug("y:\n%s" % str(list(y)))
    # logger.debug("w:\n%s" % str(list(sad.model.w)))
    # logger.debug("\nseen   : %s\nqueried: %s" % (str(list(seen)), str(list(queried))))
    return seen, seen_baseline, None, None, n_unlabeled


def aad_stream():

    logger = logging.getLogger(__name__)

    # PRODUCTION
    args = get_aad_command_args(debug=False)
    # print "log file: %s" % args.log_file
    configure_logger(args)

    opts = AadOpts(args)
    # print opts.str_opts()
    logger.debug(opts.str_opts())

    if not opts.streaming:
        raise ValueError("Only streaming supported")

    X_full, y_full = read_data(opts)
    # X_train = X_train[0:10, :]
    # labels = labels[0:10]

    logger.debug("loaded file: (%s) %s" % (str(X_full.shape), opts.datafile))
    logger.debug("results dir: %s" % opts.resultsdir)

    all_num_seen = None
    all_num_not_seen = None
    all_num_seen_baseline = None
    all_window = None
    all_window_baseline = None

    aucs = np.zeros(0, dtype=float)

    opts.fid = 1
    for runidx in opts.get_runidxs():
        tm_run = Timer()
        opts.set_multi_run_options(opts.fid, runidx)

        stream = DataStream(X_full, y_full)
        X_train, y_train = stream.read_next_from_stream(opts.stream_window)

        # logger.debug("X_train:\n%s\nlabels:\n%s" % (str(X_train), str(list(labels))))

        model = prepare_aad_model(X_train, y_train, opts)  # initial model training
        sad = StreamingAnomalyDetector(stream, model, unlabeled_x=X_train, unlabeled_y=y_train,
                                       max_buffer=opts.stream_window, opts=opts)
        sad.init_query_state(opts)

        if False:
            # use for DEBUG only
            run_feedback(sad, 0, opts.budget, opts)
            print "This is experimental/demo code for streaming integration and will be application specific." + \
                  " Exiting after reading max %d instances from stream and iterating for %d feedback..." % \
                    (opts.stream_window, opts.budget)
            exit(0)

        all_scores = np.zeros(0)
        all_y = np.zeros(0, dtype=int)

        scores = sad.get_anomaly_scores(X_train)
        # auc = fn_auc(cbind(y_train, -scores))
        all_scores = np.append(all_scores, scores)
        all_y = np.append(all_y, y_train)
        iter = 0
        seen = np.zeros(0, dtype=int)
        n_unlabeled = np.zeros(0, dtype=int)
        seen_baseline = np.zeros(0, dtype=int)
        stream_window_tmp = np.zeros(0, dtype=int)
        stream_window_baseline = np.zeros(0, dtype=int)
        stop_iter = False
        while not stop_iter:
            iter += 1

            tm = Timer()
            seen_, seen_baseline_, queried_, queried_baseline_, n_unlabeled_ = \
                run_feedback(sad,
                             opts.min_feedback_per_window,
                             opts.max_feedback_per_window,
                             opts)
            seen = append(seen, seen_)
            n_unlabeled = append(n_unlabeled, n_unlabeled_)
            seen_baseline = append(seen_baseline, seen_baseline_)
            stream_window_tmp = append(stream_window_tmp, np.ones(len(seen_)) * iter)
            stream_window_baseline = append(stream_window_baseline, np.ones(len(seen_baseline_)) * iter)
            # queried = append(queried, queried_)
            # queried_baseline = append(queried_baseline, queried_baseline_)
            # logger.debug("seen:\n%s;\nbaseline:\n%s" % (str(list(seen)), str(list(seen_baseline))))

            x_eval, y_eval = sad.get_next_from_stream(sad.max_buffer)
            if x_eval is None or iter >= opts.max_windows:
                if iter >= opts.max_windows:
                    logger.debug("Exceeded %d iters; exiting stream read..." % opts.max_windows)
                stop_iter = True
            else:
                scores = sad.get_anomaly_scores(x_eval)  # compute scores before updating the model

                all_scores = np.append(all_scores, scores)
                all_y = np.append(all_y, y_eval)

                if opts.allow_stream_update:
                    sad.update_model_from_buffer()

                sad.move_buffer_to_unlabeled()

            logger.debug(tm.message("Stream window [%d]: algo [%d/%d]; baseline [%d/%d]; unlabeled anoms [%d]: " %
                                    (iter, int(np.sum(seen)), len(seen),
                                     int(np.sum(seen_baseline)), len(seen_baseline),
                                     int(np.sum(sad.unlabeled_y)))))

        # retained = int(np.sum(sad.unlabeled_y)) if sad.unlabeled_y is not None else 0
        # logger.debug("Final retained unlabeled anoms: %d" % retained)

        auc = fn_auc(cbind(all_y, -all_scores))
        # logger.debug("AUC: %f" % auc)
        aucs = append(aucs, [auc])

        # queried_baseline = order(all_scores, decreasing=True)[0:opts.budget]
        num_seen_tmp = np.cumsum(seen)  # np.cumsum(all_y[queried])
        # logger.debug("\nnum_seen    : %s" % (str(list(num_seen_tmp)),))

        num_seen_baseline = np.cumsum(seen_baseline)  # np.cumsum(all_y[queried_baseline])
        # logger.debug("Numseen in %d budget (overall):\n%s" % (opts.budget, str(list(num_seen_baseline))))

        stream_window_baseline = append(np.array([opts.fid, opts.runidx],
                                                 dtype=stream_window_baseline.dtype),
                                        stream_window_baseline)
        stream_window = np.ones(len(stream_window_baseline) + 2, dtype=stream_window_tmp.dtype) * -1
        stream_window[0:2] = [opts.fid, opts.runidx]
        stream_window[2:(2+len(stream_window_tmp))] = stream_window_tmp

        # queried = append(np.array([opts.fid, opts.runidx], dtype=queried.dtype), queried)
        # queried_baseline = append(np.array([opts.fid, opts.runidx], dtype=queried_baseline.dtype), queried_baseline)

        # num_seen_baseline has the uniformly maximum number of queries.
        # the number of queries in num_seen will vary under the query confidence mode
        num_seen = np.ones(len(num_seen_baseline) + 2, dtype=num_seen_tmp.dtype) * -1
        num_not_seen = np.ones(len(num_seen_baseline) + 2, dtype=num_seen.dtype) * -1
        num_seen[0:2] = [opts.fid, opts.runidx]
        num_seen[2:(2+len(num_seen_tmp))] = num_seen_tmp

        # the number of unlabeled instances in buffer. For streaming this is
        # important since this represents the potential to discover true
        # anomalies. True anomalies in unlabeled set should not get discarded
        # when a new window of data arrives.
        num_not_seen[0:2] = [opts.fid, opts.runidx]
        num_not_seen[2:(2+len(n_unlabeled))] = n_unlabeled

        num_seen_baseline = append(np.array([opts.fid, opts.runidx], dtype=num_seen_baseline.dtype), num_seen_baseline)

        # all_queried = rbind(all_queried, matrix(queried, nrow=1))
        # all_queried_baseline = rbind(all_queried_baseline, matrix(queried_baseline, nrow=1))

        all_num_seen = rbind(all_num_seen, matrix(num_seen, nrow=1))
        all_num_not_seen = rbind(all_num_not_seen, matrix(num_not_seen, nrow=1))
        all_num_seen_baseline = rbind(all_num_seen_baseline, matrix(num_seen_baseline, nrow=1))
        all_window = rbind(all_window, matrix(stream_window, nrow=1))
        all_window_baseline = rbind(all_window_baseline, matrix(stream_window_baseline, nrow=1))

        logger.debug(tm_run.message("Completed runidx: %d" % runidx))

    results = SequentialResults(num_seen=all_num_seen,
                                num_not_seen=all_num_not_seen,
                                # true_queried_indexes=all_queried,
                                num_seen_baseline=all_num_seen_baseline,
                                # true_queried_indexes_baseline=all_queried_baseline,
                                stream_window=all_window,
                                stream_window_baseline=all_window_baseline,
                                aucs=aucs)
    write_sequential_results_to_csv(results, opts)


if __name__ == "__main__":
    aad_stream()
