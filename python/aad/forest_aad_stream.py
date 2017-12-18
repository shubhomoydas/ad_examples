import os
import numpy as np

import logging

from common.utils import *
from aad.aad_globals import *
from aad.aad_support import *

from aad.forest_aad_detector import *
from aad.data_stream import *


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

        self.buffer = None

        self.labeled = None
        if labeled_x is not None:
            self.labeled = InstanceList(x=labeled_x, y=labeled_y)

        self.unlabeled = None
        if unlabeled_x is not None:
            self.unlabeled = InstanceList(x=unlabeled_x, y=unlabeled_y)
            # transform the features and cache...
            self.unlabeled.x_transformed = self.get_transformed(self.unlabeled.x)

        self.qstate = None

    def reset_buffer(self):
        self.buffer = None

    def add_to_buffer(self, instances):
        if self.buffer is not None:
            self.buffer.add_instances(instances.x, instances.y,
                                      instances.ids, instances.x_transformed)
        else:
            self.buffer = instances

    def move_buffer_to_unlabeled(self):
        if self.retention_type == STREAM_RETENTION_OVERWRITE:
            if False:
                missed = int(np.sum(self.unlabeled.y)) if self.unlabeled.y is not None else 0
                retained = int(np.sum(self.buffer.y)) if self.buffer.y is not None else 0
                logger.debug("[overwriting] true anomalies: missed(%d), retained(%d)" % (missed, retained))
            if self.buffer is not None:
                self.unlabeled = self.buffer
        elif self.retention_type == STREAM_RETENTION_TOP_ANOMALOUS:
            # retain the top anomalous instances from the merged
            # set of instance from both buffer and current unlabeled.
            if self.buffer is not None:
                tmp = append_instance_lists(self.unlabeled, self.buffer)
            else:
                tmp = self.unlabeled
            n = min(tmp.x.shape[0], self.max_buffer)
            idxs, scores = self.model.order_by_score(tmp.x_transformed)
            top_idxs = idxs[np.arange(n)]
            self.unlabeled = InstanceList(x=tmp.x[top_idxs],
                                          y=tmp.y[top_idxs],
                                          x_transformed=tmp.x_transformed[top_idxs])
            if n < len(tmp.y):
                missedidxs = idxs[n:len(tmp.y)]
            else:
                missedidxs = None
            if False:
                missed = int(np.sum(tmp.y[missedidxs])) if missedidxs is not None else 0
                retained = int(np.sum(self.unlabeled.y)) if self.unlabeled.y is not None else 0
                logger.debug("[top anomalous] true anomalies: missed(%d), retained(%d)" % (missed, retained))
        self.reset_buffer()

    def get_num_instances(self):
        """Returns the total number of labeled and unlabeled instances that will be used for weight inference"""
        n = 0
        if self.unlabeled is not None:
            n += len(self.unlabeled)
        if self.labeled is not None:
            # logger.debug("labeled_x: %s" % str(self.labeled_x.shape))
            n += len(self.labeled.x)
        return n

    def init_query_state(self, opts):
        n = self.get_num_instances()
        bt = get_budget_topK(n, opts)
        self.qstate = Query.get_initial_query_state(opts.qtype, opts=opts, qrank=bt.topK,
                                                    a=1., b=1., budget=bt.budget)

    def get_next_from_stream(self, n=0, transform=False):
        if n == 0:
            n = self.max_buffer

        instances = self.stream.read_next_from_stream(n)
        if instances is not None:
            if False:
                if self.buffer is not None:
                    logger.debug("buffer shape: %s" % str(self.buffer.x.shape))
                logger.debug("x.shape: %s" % str(instances.x.shape))

            if transform:
                instances.x_transformed = self.get_transformed(instances.x)
            self.add_to_buffer(instances)
            self.model.add_samples(instances.x, current=False)

        return instances

    def update_model_from_buffer(self):
        self.model.update_model_from_stream_buffer()

    def stream_buffer_empty(self):
        return self.stream.empty()

    def get_anomaly_scores(self, x, x_transformed=None):
        if x_transformed is None:
            x_new = self.get_transformed(x)
        else:
            if x.shape[0] != x_transformed.shape[0]:
                raise ValueError("x(%d) and x_transformed(%d) are inconsistent" % (x.shape[0], x_transformed.shape[0]))
            x_new = x_transformed
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
        if self.labeled is None:
            tmp = self.unlabeled
        elif self.unlabeled is None:
            tmp = self.labeled
        else:
            tmp = append_instance_lists(self.labeled, self.unlabeled)
        if self.labeled is not None:
            ha = np.where(self.labeled.y == 1)[0]
            hn = np.where(self.labeled.y == 0)[0]
        else:
            ha = np.zeros(0, dtype=int)
            hn = np.zeros(0, dtype=int)
        if False:
            logger.debug("x: %d, ha: %d, hn:%d" % (nrow(tmp.x), len(ha), len(hn)))
        return tmp, ha, hn

    def get_instance_stats(self):
        nha = nhn = nul = 0
        if self.labeled.y is not None:
            nha = len(np.where(self.labeled.y == 1)[0])
            nhn = len(np.where(self.labeled.y == 0)[0])
        if self.unlabeled is not None:
            nul = len(self.unlabeled)
        return nha, nhn, nul

    def get_num_labeled(self):
        """Returns the number of instances for which we already have label feedback"""
        if self.labeled is not None:
            return len(self.labeled.y)
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
        x_transformed = None
        if x is None:
            tmp, ha, hn = self.setup_data_for_feedback()
            x, y, x_transformed = tmp.x, tmp.y, tmp.x_transformed
        if w is None:
            w = self.model.w
        if unl is None:
            unl = np.zeros(0, dtype=int)
        # the top n_feedback instances in the instance list are the labeled items
        queried_items = append(np.arange(n_feedback), unl)
        if x_transformed is None:
            x_transformed = self.get_transformed(x)
            logger.debug("needs transformation")
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

    def get_transformed(self, x, opts=None):
        """Returns the instance.x_transformed

        :param instances: InstanceList
        :return: scipy sparse array
        """
        # logger.debug("transforming data...")
        if opts is None:
            opts = self.opts
        x_transformed = self.model.transform_to_ensemble_features(
            x, dense=False, norm_unit=opts.norm_unit)
        return x_transformed

    def move_unlabeled_to_labeled(self, xi, yi):
        unlabeled_idx = xi - self.get_num_labeled()
        x, _, id, x_trans = self.unlabeled.get_instance_at(unlabeled_idx)
        if self.labeled is None:
            self.labeled = InstanceList(x=matrix(self.unlabeled.x[unlabeled_idx], nrow=1),
                                        y=np.array([yi], dtype=int),
                                        ids=None if id is None else np.array([id], dtype=int),
                                        x_transformed=x_trans)
        else:
            self.labeled.add_instance(x, y=yi, id=id, x_transformed=x_trans)
        self.unlabeled.remove_instance_at(unlabeled_idx)

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
            x = self.get_transformed(x)
        ordered_indexes, scores = self.model.order_by_score(x, w=w)
        tn = min(10, nrow(x))
        vars = np.zeros(tn, dtype=float)
        for i in np.arange(tn):
            vars[i] = get_linear_score_variance(x[ordered_indexes[i], :], w)
        # logger.debug("top %d vars:\n%s" % (tn, str(list(vars))))
        return vars

    def print_instance_stats(self, msg="debug"):
        logger.debug("%s:\nlabeled: %s, unlabeled: %s" %
                     (msg,
                      '-' if self.labeled is None else str(self.labeled),
                      '-' if self.unlabeled is None else str(self.unlabeled)))


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
        x_transformed = sad.get_transformed(sad.unlabeled.x)
        ordered_idxs, _ = sad.model.order_by_score(x_transformed)
        seen_baseline = sad.unlabeled.y[ordered_idxs[0:max_feedback]]
        num_seen_baseline = np.cumsum(seen_baseline)
        logger.debug("num_seen_baseline:\n%s" % str(list(num_seen_baseline)))

    # baseline scores
    w_baseline = sad.model.get_uniform_weights()
    order_baseline, scores_baseline = sad.model.order_by_score(sad.unlabeled.x_transformed, w_baseline)
    n_seen_baseline = min(max_feedback, len(sad.unlabeled.y))
    queried_baseline = order_baseline[0:n_seen_baseline]
    seen_baseline = sad.unlabeled.y[queried_baseline]

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
        n_unlabeled = np.append(n_unlabeled, [int(np.sum(sad.unlabeled.y))])
        # logger.debug("y:\n%s" % str(list(y)))
    # logger.debug("w:\n%s" % str(list(sad.model.w)))
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

    np.random.seed(opts.randseed)

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

        stream = DataStream(X_full, y_full, IdServer(initial=0))
        training_set = stream.read_next_from_stream(opts.stream_window)
        X_train, y_train, ids = training_set.x, training_set.y, training_set.ids

        model = prepare_aad_model(X_train, y_train, opts)  # initial model training
        sad = StreamingAnomalyDetector(stream, model, unlabeled_x=X_train, unlabeled_y=y_train,
                                       max_buffer=opts.stream_window, opts=opts)
        sad.init_query_state(opts)

        all_scores = np.zeros(0)
        all_y = np.zeros(0, dtype=int)

        scores = sad.get_anomaly_scores(sad.unlabeled.x, sad.unlabeled.x_transformed)
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

            # gather metrics...
            seen = append(seen, seen_)
            n_unlabeled = append(n_unlabeled, n_unlabeled_)
            seen_baseline = append(seen_baseline, seen_baseline_)
            stream_window_tmp = append(stream_window_tmp, np.ones(len(seen_)) * iter)
            stream_window_baseline = append(stream_window_baseline, np.ones(len(seen_baseline_)) * iter)

            # get the next window of data from stream and transform features...
            instances = sad.get_next_from_stream(sad.max_buffer, transform=True)
            if instances is None or iter >= opts.max_windows:
                if iter >= opts.max_windows:
                    logger.debug("Exceeded %d iters; exiting stream read..." % opts.max_windows)
                stop_iter = True
            else:
                # compute scores before updating the model
                scores = sad.get_anomaly_scores(instances.x, instances.x_transformed)

                all_scores = np.append(all_scores, scores)
                all_y = np.append(all_y, instances.y)

                if opts.allow_stream_update:
                    sad.update_model_from_buffer()

                sad.move_buffer_to_unlabeled()

            logger.debug(tm.message("Stream window [%d]: algo [%d/%d]; baseline [%d/%d]; unlabeled anoms [%d]: " %
                                    (iter, int(np.sum(seen)), len(seen),
                                     int(np.sum(seen_baseline)), len(seen_baseline),
                                     int(np.sum(sad.unlabeled.y)))))

        # retained = int(np.sum(sad.unlabeled_y)) if sad.unlabeled_y is not None else 0
        # logger.debug("Final retained unlabeled anoms: %d" % retained)

        auc = fn_auc(cbind(all_y, -all_scores))
        # logger.debug("AUC: %f" % auc)
        aucs = append(aucs, [auc])

        num_seen_tmp = np.cumsum(seen)
        # logger.debug("\nnum_seen    : %s" % (str(list(num_seen_tmp)),))

        num_seen_baseline = np.cumsum(seen_baseline)
        # logger.debug("Numseen in %d budget (overall):\n%s" % (opts.budget, str(list(num_seen_baseline))))

        stream_window_baseline = append(np.array([opts.fid, opts.runidx],
                                                 dtype=stream_window_baseline.dtype),
                                        stream_window_baseline)
        stream_window = np.ones(len(stream_window_baseline) + 2, dtype=stream_window_tmp.dtype) * -1
        stream_window[0:2] = [opts.fid, opts.runidx]
        stream_window[2:(2+len(stream_window_tmp))] = stream_window_tmp

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
