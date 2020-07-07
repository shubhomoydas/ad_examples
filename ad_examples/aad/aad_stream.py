import os
import logging
import numpy as np
from ..common.utils import (
    logger, InstanceList, Timer, append_instance_lists, cbind, rbind, append, matrix, read_data_as_matrix,
    get_sample_feature_ranges, configure_logger
)
from ..common.metrics import fn_auc
from .aad_globals import (
    STREAM_RETENTION_OVERWRITE, STREAM_RETENTION_TOP_ANOMALOUS, get_aad_command_args, AadOpts,
    get_first_vals_not_marked
)
from .aad_base import get_budget_topK, Ensemble
from .forest_aad_detector import is_forest_detector
from .query_model import Query
from .aad_support import get_aad_model, load_aad_model, SequentialResults, write_sequential_results_to_csv

from .data_stream import DataStream, IdServer
from .aad_test_support import plot_score_contours
from .query_model_euclidean import filter_by_euclidean_distance


class StreamingAnomalyDetector(object):
    """
    Attributes:
        model: Aad
            trained AAD model
        stream: DataStream
        max_buffer: int
            Determines the window size
        labeled: InstanceList
        unlabeled: InstanceList
        buffer: InstanceList
            test set from stream
        initial_labeled: InstanceList
        initial_anomalies: InstanceList
        initial_nominals: InstanceList
        opts: AadOpts
    """
    def __init__(self, stream, model, labeled_x=None, labeled_y=None, labeled_ids=None,
                 unlabeled_x=None, unlabeled_y=None, unlabeled_ids=None, opts=None,
                 max_buffer=512, min_samples_for_update=256):
        self.model = model
        self.stream = stream
        self.max_buffer = max_buffer
        self.min_samples_for_update = min_samples_for_update
        self.opts = opts
        self.n_prelabeled_instances = 0

        self.buffer = None

        self.initial_labeled, self.initial_anomalies, self.initial_nominals = \
            self.get_initial_labeled(labeled_x, labeled_y, labeled_ids)

        self.labeled = self._get_pretrain_labeled()
        if self.labeled is not None:
            self.n_prelabeled_instances = self.labeled.x.shape[0]

        self.unlabeled = None
        if unlabeled_x is not None:
            self.unlabeled = InstanceList(x=unlabeled_x, y=unlabeled_y, ids=unlabeled_ids)
            # transform the features and cache...
            self.unlabeled.x_transformed = self.get_transformed(self.unlabeled.x)

        self.qstate = None
        self.feature_ranges = None  # required if diverse querying strategy is used

        self.current_dists = None
        self.kl_alpha = opts.kl_alpha
        self.kl_q_alpha = 0.

        if is_forest_detector(self.opts.detector_type):
            # initialize the baseline instance distributions required for evaluating KL-divergence
            all_instances = self._get_all_instances()
            self.current_dists = self.model.get_node_sample_distributions(all_instances.x)

            kl_trees, self.kl_q_alpha = self.model.get_KL_divergence_distribution(all_instances.x, alpha=self.kl_alpha)
            logger.debug("kl kl_q_alpha: %s (alpha=%0.2f), kl mean: %f, kl_trees:\n%s" %
                         (str(list(self.kl_q_alpha)), self.kl_alpha, np.mean(kl_trees), str(list(kl_trees))))

        self._pre_train(debug_auc=True)

    def _pre_train(self, debug_auc=False):
        if not self.opts.pretrain or self.initial_labeled is None or self.opts.n_pretrain == 0:
            return

        ha = np.where(self.initial_labeled.y == 1)[0]
        # hn = np.where(self.initial_labeled.y == 0)[0]
        # set hn to empty array for pre-training. Since all instances are labeled,
        # we just focus on getting the labeled anomalies ranked at the top
        hn = np.zeros(0, dtype=int)

        if len(ha) == 0 or len(ha) == len(self.initial_labeled.y):
            logger.debug("At least one example from each class (anomaly, nominal) is required for pretraining.")
            return

        logger.debug("Pre-training %d rounds with anomalies: %d, nominals: %d..." %
                     (self.opts.n_pretrain, len(ha), len(self.initial_labeled.y)-len(ha)))

        tm = Timer()
        x, y, ids, x_transformed = self.initial_labeled.x, self.initial_labeled.y, self.initial_labeled.ids, self.initial_labeled.x_transformed
        orig_tau = self.opts.tau
        self.opts.tau = len(ha)*1.0 / len(self.initial_labeled.y)
        auc = self.get_auc(x=x, y=y, x_transformed=x_transformed)
        if self.opts.dataset in ['toy', 'toy2', 'toy_hard']:
            plot_score_contours(x, y, x_transformed, model=self.model,
                                filename="baseline", outputdir=self.opts.resultsdir,
                                opts=self.opts)
        if debug_auc: logger.debug("AUC[0]: %f" % (auc))
        best_i = 0
        best_auc = auc
        best_w = self.model.w
        for i in range(self.opts.n_pretrain):
            self.model.update_weights(x_transformed, y, ha, hn, self.opts)
            auc = self.get_auc(x=x, y=y, x_transformed=x_transformed)
            if debug_auc: logger.debug("AUC[%d]: %f" % (i + 1, auc))
            if best_auc < auc:
                best_auc = auc
                best_w = np.copy(self.model.w)
                best_i = i+1
        logger.debug("best_i: %d, best_auc: %f" % (best_i, best_auc))
        self.model.w = best_w
        self.opts.tau = orig_tau

        if self.opts.dataset in ['toy', 'toy2', 'toy_hard']:
            # some DEBUG plots
            selx = None
            if self.labeled is not None:
                idxs = np.where(self.labeled.y == 0)[0]
                logger.debug("#selx: %d" % len(idxs))
                selx = self.labeled.x[idxs]
            plot_score_contours(x, y, x_transformed, selected_x=selx, model=self.model,
                                filename="pre_train", outputdir=self.opts.resultsdir,
                                opts=self.opts)

        logger.debug(tm.message("Updated weights %d times with no feedback " % self.opts.n_pretrain))

    def get_initial_labeled(self, x, y, ids):
        """Returns the labeled instances as InstanceLists

        :param x: np.ndarray
        :param y: np.array
        :param ids: np.array
        :return: InstanceList, InstanceList, InstanceList
        """
        initial_labeled = initial_anomalies = initial_nominals = None
        if x is not None:
            initial_labeled = InstanceList(x=x, y=y, ids=ids)
            # transform the features and cache...
            initial_labeled.x_transformed = self.get_transformed(initial_labeled.x)
            initial_anomalies, initial_nominals = self._separate_anomaly_nominal(initial_labeled)
        return initial_labeled, initial_anomalies, initial_nominals

    def _get_pretrain_labeled(self):
        """Returns a subset of the initial labeled data which will be utilized in future

        First, we retain all labeled anomalies since these provide vital information.
        Retaining all nominals might result in severe class imbalance if they are in
        relatively larger number compared to anomalies. Therefore, we subsample the nominals.

        We need to determine a reasonable informative set of nominals. For this, we utilize
        Euclidean-diversity based strategy. We retain the nominals which have highest
        average distance from the anomalies as well as other selected nominals.

        :return: InstanceList
        """
        l = self.initial_labeled
        if l is None:
            return None
        if self.opts.n_pretrain_nominals < 0:
            # include all nominal instances
            labeled = InstanceList(x=self.initial_labeled.x, y=self.initial_labeled.y,
                                   ids=self.initial_labeled.ids,
                                   x_transformed=self.initial_labeled.x_transformed)
        elif self.opts.n_pretrain_nominals == 0:
            # completely ignore nominals and only retain anomalies
            labeled = InstanceList(x=self.initial_anomalies.x, y=self.initial_anomalies.y,
                                   ids=self.initial_anomalies.ids,
                                   x_transformed=self.initial_anomalies.x_transformed)
        else:
            # select a subset of nominals
            tm = Timer()
            anom_idxs = np.where(l.y == 1)[0]
            noml_idxs = np.where(l.y == 0)[0]

            # set number of nominals...
            n_nominals = min(self.opts.n_pretrain_nominals, len(anom_idxs))
            if n_nominals > 0:
                selected_indexes = filter_by_euclidean_distance(l.x,
                                                                noml_idxs, init_selected=anom_idxs,
                                                                n_select=n_nominals)
            else:
                selected_indexes = anom_idxs
            selected_indexes = np.array(selected_indexes, dtype=int)
            labeled = InstanceList(x=l.x[selected_indexes], y=l.y[selected_indexes],
                                   x_transformed=l.x_transformed[selected_indexes],
                                   ids=l.ids[selected_indexes])
            logger.debug(tm.message("Total labeled: %d, anomalies: %d, nominals: %d" %
                         (labeled.x.shape[0], len(anom_idxs), len(selected_indexes)-len(anom_idxs))))
        return labeled

    def _separate_anomaly_nominal(self, labeled):
        anom_idxs = np.where(labeled.y == 1)[0]
        noml_idxs = np.where(labeled.y == 0)[0]
        anomalies = None
        nominals = None
        if len(anom_idxs) > 0:
            anomalies = InstanceList(x=labeled.x[anom_idxs], y=labeled.y[anom_idxs], ids=labeled.ids[anom_idxs],
                                     x_transformed=labeled.x_transformed[anom_idxs])
        if len(noml_idxs) > 0:
            nominals = InstanceList(x=labeled.x[noml_idxs], y=labeled.y[noml_idxs], ids=labeled.ids[noml_idxs],
                                    x_transformed=labeled.x_transformed[noml_idxs])
        return anomalies, nominals

    def _get_all_instances(self):
        if self.labeled is not None and self.unlabeled is not None:
            all_instances = append_instance_lists(self.labeled, self.unlabeled)
        elif self.labeled is not None:
            all_instances = self.labeled
        else:
            all_instances = self.unlabeled
        return all_instances

    def reset_buffer(self):
        self.buffer = None

    def add_to_buffer(self, instances):
        if self.buffer is not None:
            self.buffer.add_instances(instances.x, instances.y,
                                      instances.ids, instances.x_transformed)
        else:
            self.buffer = instances

    def move_buffer_to_unlabeled(self):
        if self.opts.retention_type == STREAM_RETENTION_OVERWRITE:
            if False:
                missed = int(np.sum(self.unlabeled.y)) if self.unlabeled.y is not None else 0
                retained = int(np.sum(self.buffer.y)) if self.buffer.y is not None else 0
                logger.debug("[overwriting] true anomalies: missed(%d), retained(%d)" % (missed, retained))
            if self.buffer is not None:
                self.unlabeled = self.buffer
        elif self.opts.retention_type == STREAM_RETENTION_TOP_ANOMALOUS:
            # retain the top anomalous instances from the merged
            # set of instance from both buffer and current unlabeled.
            if self.buffer is not None and self.unlabeled is not None:
                tmp = append_instance_lists(self.unlabeled, self.buffer)
            elif self.buffer is not None:
                tmp = self.buffer
            else:
                tmp = self.unlabeled
            n = min(tmp.x.shape[0], self.max_buffer)
            idxs, scores = self.model.order_by_score(tmp.x_transformed)
            top_idxs = idxs[np.arange(n)]
            tmp_x, tmp_y, tmp_ids, tmp_trans = tmp.get_instances_at(top_idxs)
            self.unlabeled = InstanceList(x=tmp_x, y=tmp_y, ids=tmp_ids, x_transformed=tmp_trans)
            # self.unlabeled = InstanceList(x=tmp.x[top_idxs],
            #                               y=tmp.y[top_idxs],
            #                               x_transformed=tmp.x_transformed[top_idxs])
            if n < len(tmp.y):
                missedidxs = idxs[n:len(tmp.y)]
            else:
                missedidxs = None
            if False:
                missed = int(np.sum(tmp.y[missedidxs])) if missedidxs is not None else 0
                retained = int(np.sum(self.unlabeled.y)) if self.unlabeled.y is not None else 0
                logger.debug("[top anomalous] true anomalies: missed(%d), retained(%d)" % (missed, retained))
        self.feature_ranges = get_sample_feature_ranges(self.unlabeled.x)
        self.reset_buffer()

    def get_num_instances(self):
        """Returns the total number of labeled and unlabeled instances that will be used for weight inference"""
        n = 0
        if self.unlabeled is not None:
            n += len(self.unlabeled)
        if self.labeled is not None:
            # logger.debug("labeled_x: %s" % str(self.labeled_x.shape))
            n += len(self.labeled)
        return n

    def init_query_state(self):
        n = self.get_num_instances()
        bt = get_budget_topK(n, self.opts)
        self.qstate = Query.get_initial_query_state(self.opts.qtype, opts=self.opts, qrank=bt.topK,
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

    def update_model_from_buffer(self, transform=False):
        """Updates the underlying model if it meets the criteria

        The minimum number of samples required for model update is:
            max(self.min_samples_for_update, self.opts.stream_window//2)

        We will replace trees in the following conditions:
            - if check_KL_divergence is True, then check whether the KL-divergence
                from reference distributions of 2*kl_alpha number of trees exceed
                the alpha-threshold; if so, then replace all trees which exceed their
                respective thresholds.
            - if check_KL_divergence is False, then replace the configured fraction of
                oldest trees. The fraction is configured with the command line
                parameter --forest_replace_frac.

        :param transform: bool
        :return:
        """
        model_updated = False

        min_samples_required = max(self.min_samples_for_update, self.opts.stream_window//2)
        if self.buffer is None or self.buffer.x is None or self.buffer.x.shape[0] < min_samples_required:
            logger.warning("Insufficient samples (%d) for model update. Minimum required: %d = max(%d,%d)." %
                           (0 if self.buffer is None or self.buffer.x is None else self.buffer.x.shape[0],
                            min_samples_required, self.min_samples_for_update, self.opts.stream_window//2))
        else:
            tm = Timer()
            n_trees = self.model.clf.n_estimators
            n_threshold = int(2 * self.kl_alpha * n_trees)
            replace_trees_by_kl = None

            if self.opts.check_KL_divergence:
                kl_trees, _ = self.model.get_KL_divergence_distribution(self.buffer.x, p=self.current_dists)
                replace_trees_by_kl = self.model.get_trees_to_replace(kl_trees, self.kl_q_alpha)
                logger.debug("kl kl_q_alpha: %s (alpha=%0.2f), kl_trees:\n%s\n(#replace: %d): %s" %
                             (str(list(self.kl_q_alpha)), self.kl_alpha, str(list(kl_trees)), len(replace_trees_by_kl), str(list(replace_trees_by_kl))))

            n_replace = 0 if replace_trees_by_kl is None else len(replace_trees_by_kl)

            # check whether conditions for tree-replacement are satisfied.
            do_replace = not self.opts.check_KL_divergence or (n_trees > 0 and n_replace >= n_threshold)
            if do_replace:
                self.model.update_model_from_stream_buffer(replace_trees=replace_trees_by_kl)
                if is_forest_detector(self.opts.detector_type):
                    self.current_dists = self.model.get_node_sample_distributions(self.buffer.x)
                    kl_trees, self.kl_q_alpha = self.model.get_KL_divergence_distribution(self.buffer.x, alpha=self.kl_alpha)
                    logger.debug("kl kl_q_alpha: %s, kl_trees:\n%s" % (str(list(self.kl_q_alpha)), str(list(kl_trees))))
                model_updated = True

            logger.debug(tm.message(
                "Model%s updated; n_replace: %d, n_threshold: %d, kl_q_alpha: %s (check_KL: %s, alpha: %0.2f)" %
                (" not" if not do_replace else "", n_replace, n_threshold,
                 str(list(self.kl_q_alpha)), str(self.opts.check_KL_divergence), self.kl_alpha)
            ))

        if transform:
            if self.labeled is not None and self.labeled.x is not None:
                self.labeled.x_transformed = self.get_transformed(self.labeled.x)
            if self.unlabeled is not None and self.unlabeled.x is not None:
                self.unlabeled.x_transformed = self.get_transformed(self.unlabeled.x)
            if self.buffer is not None and self.buffer.x is not None:
                self.buffer.x_transformed = self.get_transformed(self.buffer.x)

        return model_updated

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

    def get_auc(self, x, y, x_transformed=None):
        scores = self.get_anomaly_scores(x, x_transformed=x_transformed)
        auc = fn_auc(cbind(y, -scores))
        return auc

    def get_allowed_labeled_subset(self):
        """ Returns a randomly selected subset of labeled instances

        The  number of instances returned is determined by the upper limit
        specified through the optional parameters opts.labeled_to_window_ratio
        and opts.max_labeled_for_stream in the streaming mode.
        """
        # first, compute the maximum number of labeled instances allowed for
        # computing AAD losses and constraints...
        n_labeled = 0 if self.labeled is None else len(self.labeled.x)
        if n_labeled == 0 or (self.opts.labeled_to_window_ratio is None and self.opts.max_labeled_for_stream is None):
            return self.labeled
        n_allowed_labeled = self.max_buffer if self.opts.labeled_to_window_ratio is None \
            else int(self.opts.labeled_to_window_ratio * self.max_buffer)
        n_allowed_labeled = n_allowed_labeled if self.opts.max_labeled_for_stream is None \
            else min(n_allowed_labeled, self.opts.max_labeled_for_stream)
        n_allowed_labeled = min(n_allowed_labeled, n_labeled)
        if n_allowed_labeled == n_labeled:
            return self.labeled

        labeled = InstanceList(x=self.labeled.x, y=self.labeled.y,
                               ids=self.labeled.ids, x_transformed=self.labeled.x_transformed)
        n_per_type = n_allowed_labeled // 2
        anom_idxs = np.where(self.labeled.y == 1)[0]
        noml_idxs = np.where(self.labeled.y == 0)[0]
        if len(anom_idxs) > n_per_type:
            np.random.shuffle(anom_idxs)
            idxs = anom_idxs[0:n_per_type]
        else:
            idxs = anom_idxs
        n_anoms = len(idxs)
        n_nomls = n_allowed_labeled - n_anoms
        if len(noml_idxs) > n_nomls:
            np.random.shuffle(noml_idxs)
            idxs = np.append(idxs, noml_idxs[0:n_nomls])
        else:
            idxs = np.append(idxs, noml_idxs)
        n_nomls = len(idxs) - n_anoms
        if False:
            logger.debug("n_labeled: %d, n_allowed_labeled: %d, n_anoms: %d, n_nomls: %d" %
                         (n_labeled, n_allowed_labeled, n_anoms, n_nomls))
        mask = np.zeros(n_labeled, dtype=bool)
        mask[idxs[0:n_allowed_labeled]] = True
        labeled.retain_with_mask(mask)

        return labeled

    def setup_data_for_feedback(self):
        """
        Prepares the input matrices/data structures for weight update. The format
        is such that the top rows of data matrix are labeled and below are unlabeled.

        :return: (np.ndarray, np.array, np.array, np.array)
            (x, y, ha, hn)
            x - data matrix, y - labels (np.nan for unlabeled),
            ha - indexes of labeled anomalies, hn - indexes of labeled nominals
        """
        labeled = self.get_allowed_labeled_subset()
        if labeled is None:
            tmp = self.unlabeled
        elif self.unlabeled is None:
            tmp = labeled
        else:
            tmp = append_instance_lists(labeled, self.unlabeled)
        if labeled is not None:
            ha = np.where(labeled.y == 1)[0]
            hn = np.where(labeled.y == 0)[0]
        else:
            ha = np.zeros(0, dtype=int)
            hn = np.zeros(0, dtype=int)
        if False:
            logger.debug("x: %d, ha: %d, hn:%d" % (nrow(tmp.x), len(ha), len(hn)))
        return tmp, ha, hn

    def get_instance_stats(self):
        nha = nhn = nul = 0
        if self.labeled is not None and self.labeled.y is not None:
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

    def reestimate_tau(self, default_tau):
        """Re-estimate the proportion of anomalies

        The number of true anomalies discovered might end up being high
        relative to the data in the memory. We need to adjust for that...

        :param default_tau: float
            default proportion of anomalies
        :return: float
        """
        new_tau = default_tau
        nha, nhn, nul = self.get_instance_stats()
        frac_known_anom = nha * 1.0 / (nha + nhn + nul)
        if frac_known_anom >= default_tau:
            new_tau = frac_known_anom + 0.01
            logger.debug("Exceeded original tau (%f); setting tau=%f" % (default_tau, new_tau))
        return new_tau

    def update_weights_with_no_feedback(self, n_train=None, debug_auc=False):
        """Runs the weight update n times

        This is used when:
          1. There has been a significant update to the model because
             of (say) data drift and we want to iteratively estimate the
             ensemble weights and the tau-quantile value a number of times.
          2. We have an initial fully labeled set with which we want to
             pretrain the model
        """
        n = n_train if n_train is not None else self.opts.n_weight_updates_after_stream_window
        if self.opts.do_not_update_weights or n <= 0:
            return

        tm = Timer()
        tmp, ha, hn = self.setup_data_for_feedback()
        x, y, ids, x_transformed = tmp.x, tmp.y, tmp.ids, tmp.x_transformed
        orig_tau = self.opts.tau
        self.opts.tau = self.reestimate_tau(orig_tau)
        if debug_auc: logger.debug("AUC[0]: %f" % (self.get_auc(x=x, y=y, x_transformed=x_transformed)))
        for i in range(n):
            self.model.update_weights(x_transformed, y, ha, hn, self.opts)
            if debug_auc: logger.debug("AUC[%d]: %f" % (i+1, self.get_auc(x=x, y=y, x_transformed=x_transformed)))
        self.opts.tau = orig_tau
        logger.debug(tm.message("Updated weights %d times with no feedback " % n))

    def get_query_data(self, x=None, y=None, ids=None, ha=None, hn=None, unl=None, w=None, n_query=1):
        """Returns the best instance that should be queried, along with other data structures

        Args:
            x: np.ndarray
                input instances (labeled + unlabeled)
            y: np.array
                labels for instances which are already labeled, else some dummy values
            ids: np.array
                unique instance ids
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
        if self.get_num_instances() == 0:
            raise ValueError("No instances available")
        x_transformed = None
        if x is None:
            tmp, ha, hn = self.setup_data_for_feedback()
            x, y, ids, x_transformed = tmp.x, tmp.y, tmp.ids, tmp.x_transformed
        n = x.shape[0]
        if w is None:
            w = self.model.w
        if unl is None:
            unl = np.zeros(0, dtype=int)
        n_feedback = len(ha) + len(hn)
        # the top n_feedback instances in the instance list are the labeled items
        queried_items = append(np.arange(n_feedback), unl)
        if x_transformed is None:
            x_transformed = self.get_transformed(x)
            logger.debug("needs transformation")
        order_anom_idxs, anom_score = self.model.order_by_score(x_transformed)
        ensemble = Ensemble(x, original_indexes=0)
        xi = self.qstate.get_next_query(maxpos=n, ordered_indexes=order_anom_idxs,
                                        queried_items=queried_items,
                                        ensemble=ensemble,
                                        feature_ranges=self.feature_ranges,
                                        model=self.model,
                                        x=x_transformed, lbls=y, anom_score=anom_score,
                                        w=w, hf=append(ha, hn),
                                        remaining_budget=self.opts.num_query_batch, # self.opts.budget - n_feedback,
                                        n=n_query)
        if False:
            logger.debug("ordered instances[%d]: %s\nha: %s\nhn: %s\nxi: %s" %
                         (self.opts.budget, str(list(order_anom_idxs[0:self.opts.budget])),
                          str(list(ha)), str(list(hn)), str(list(xi))))
        return xi, x, y, ids, x_transformed, ha, hn, order_anom_idxs, anom_score

    def get_transformed(self, x):
        """Returns the instance.x_transformed

        Args:
            instances: InstanceList

        Returns: scipy sparse array
        """
        # logger.debug("transforming data...")
        x_transformed = self.model.transform_to_ensemble_features(
            x, dense=False, norm_unit=self.opts.norm_unit)
        return x_transformed

    def move_unlabeled_to_labeled(self, xi, yi):
        unlabeled_idxs = xi
        x, _, id, x_trans = self.unlabeled.get_instances_at(unlabeled_idxs)
        if self.labeled is None:
            self.labeled = InstanceList(x=self.unlabeled.x[unlabeled_idxs, :],
                                        y=yi,
                                        ids=None if id is None else id,
                                        x_transformed=x_trans)
        else:
            self.labeled.add_instance(x, y=yi, id=id, x_transformed=x_trans)
        self.unlabeled.remove_instance_at(unlabeled_idxs)

    def update_weights_with_feedback(self, xis, yis, x, y, x_transformed, ha, hn):
        """Relearns the optimal weights from feedback and updates internal labeled and unlabeled matrices

        IMPORTANT:
            This API assumes that the input x, y, x_transformed are consistent with
            the internal labeled/unlabeled matrices, i.e., the top rows/values in
            these matrices are from labeled data and bottom ones are from internally
            stored unlabeled data.

        Args:
            xis: np.array(dtype=int)
                indexes of instances in Union(self.labeled, self.unlabeled)
            yis: np.array(dtype=int)
                labels {0, 1} of instances (supposedly provided by an Oracle)
            x: numpy.ndarray
                set of all instances
            y: list of int
                set of all labels (only those at locations in the lists ha and hn are relevant)
            x_transformed: numpy.ndarray
                x transformed to ensemble features
            ha: list of int
                indexes of labeled anomalies
            hn: list of int
                indexes of labeled nominals
        """

        # Add the newly labeled instance to the corresponding list of labeled
        # instances and remove it from the unlabeled set.
        nhn = len(ha) + len(hn)
        self.move_unlabeled_to_labeled(xis - nhn, yis)

        for xi, yi in zip(xis, yis):
            if yi == 1:
                ha = append(ha, [xi])
            else:
                hn = append(hn, [xi])

        if not self.opts.do_not_update_weights:
            self.model.update_weights(x_transformed, y, ha, hn, self.opts)

    def run_feedback(self):
        """Runs active learning loop for current unlabeled window of data."""

        min_feedback = self.opts.min_feedback_per_window
        max_feedback = self.opts.max_feedback_per_window

        # For the last window, we query till the buffer is exhausted
        # irrespective of whether we exceed max_feedback per window limit
        if self.stream_buffer_empty() and self.opts.till_budget:
            bk = get_budget_topK(self.unlabeled.x.shape[0], self.opts)
            n_labeled = 0 if self.labeled is None else len(self.labeled.y)
            max_feedback = max(0, bk.budget - (n_labeled - self.n_prelabeled_instances))
            max_feedback = min(max_feedback, self.unlabeled.x.shape[0])

        if False:
            # get baseline metrics
            x_transformed = self.get_transformed(self.unlabeled.x)
            ordered_idxs, _ = self.model.order_by_score(x_transformed)
            seen_baseline = self.unlabeled.y[ordered_idxs[0:max_feedback]]
            num_seen_baseline = np.cumsum(seen_baseline)
            logger.debug("num_seen_baseline:\n%s" % str(list(num_seen_baseline)))

        # baseline scores
        w_baseline = self.model.get_uniform_weights()
        order_baseline, scores_baseline = self.model.order_by_score(self.unlabeled.x_transformed, w_baseline)
        n_seen_baseline = min(max_feedback, len(self.unlabeled.y))
        queried_baseline = order_baseline[0:n_seen_baseline]
        seen_baseline = self.unlabeled.y[queried_baseline]

        orig_tau = self.opts.tau
        self.opts.tau = self.reestimate_tau(orig_tau)

        seen = np.zeros(0, dtype=int)
        n_unlabeled = np.zeros(0, dtype=int)
        queried = np.zeros(0, dtype=int)
        unl = np.zeros(0, dtype=int)
        i = 0
        n_feedback = 0
        while n_feedback < max_feedback:
            i += 1

            # scores based on current weights
            xi_, x, y, ids, x_transformed, ha, hn, order_anom_idxs, anom_score = \
                self.get_query_data(unl=unl, n_query=self.opts.n_explore)

            order_anom_idxs_minus_ha_hn = get_first_vals_not_marked(
                order_anom_idxs, append(ha, hn), n=len(order_anom_idxs))

            bt = get_budget_topK(x_transformed.shape[0], self.opts)

            # Note: We will ensure that the tau-th instance is atleast 10-th (or lower) ranked
            tau_rank = min(max(bt.topK, 10), x.shape[0])

            xi = np.array(xi_, dtype=int)
            if n_feedback + len(xi) > max_feedback:
                xi = xi[0:(max_feedback - n_feedback)]
            n_feedback += len(xi)
            # logger.debug("n_feedback: %d, #xi: %d" % (n_feedback, len(xi)))
            means = vars = qpos = m_tau = v_tau = None
            if self.opts.query_confident:
                # get the mean score and its variance for the top ranked instances
                # excluding the instances which have already been queried
                means, vars, test, v_eval, _ = get_score_variances(x_transformed, self.model.w,
                                                                   n_test=tau_rank,
                                                                   ordered_indexes=order_anom_idxs,
                                                                   queried_indexes=append(ha, hn))
                # get the mean score and its variance for the tau-th ranked instance
                m_tau, v_tau, _, _, _ = get_score_variances(x_transformed[order_anom_idxs_minus_ha_hn[tau_rank]],
                                                            self.model.w, n_test=1,
                                                            test_indexes=np.array([0], dtype=int))
                qpos = np.where(test == xi[0])[0]  # top-most ranked instance

            if False and self.opts.query_confident:
                logger.debug("tau score:\n%s (%s)" % (str(list(m_tau)), str(list(v_tau))))
                strmv = ",".join(["%f (%f)" % (means[j], vars[j]) for j in np.arange(len(means))])
                logger.debug("scores:\n%s" % strmv)

            # check if we are confident that this is larger than the tau-th ranked instance
            if (not self.opts.query_confident) or (n_feedback <= min_feedback or
                                              means[qpos] - 3. * np.sqrt(vars[qpos]) >= m_tau):
                seen = np.append(seen, y[xi])
                queried_ = [ids[q] for q in xi]
                queried = np.append(queried, queried_)
                tm_update = Timer()
                self.update_weights_with_feedback(xi, y[xi], x, y, x_transformed, ha, hn)
                tm_update.end()
                # reset the list of queried test instances because their scores would have changed
                unl = np.zeros(0, dtype=int)
                if False:
                    nha, nhn, nul = self.get_instance_stats()
                    # logger.debug("xi:%d, test indxs: %s, qpos: %d" % (xi, str(list(test)), qpos))
                    # logger.debug("orig scores:\n%s" % str(list(anom_score[order_anom_idxs[0:tau_rank]])))
                    logger.debug("[%d] #feedback: %d; ha: %d; hn: %d, mnw: %d, mxw: %d; update: %f sec(s)" %
                                 (i, nha + nhn, nha, nhn, min_feedback, max_feedback, tm_update.elapsed()))
            else:
                # ignore these instances from query
                unl = np.append(unl, xi)
                # logger.debug("skipping feedback for xi=%d at iter %d; unl: %s" % (xi, i, str(list(unl))))
                # continue
            n_unlabeled = np.append(n_unlabeled, [int(np.sum(self.unlabeled.y))])
            # logger.debug("y:\n%s" % str(list(y)))
        self.opts.tau = orig_tau
        # logger.debug("w:\n%s" % str(list(sad.model.w)))
        return seen, seen_baseline, queried, None, n_unlabeled

    def print_instance_stats(self, msg="debug"):
        logger.debug("%s:\nlabeled: %s, unlabeled: %s" %
                     (msg,
                      '-' if self.labeled is None else str(self.labeled),
                      '-' if self.unlabeled is None else str(self.unlabeled)))


def train_aad_model(opts, x):
    random_state = np.random.RandomState(opts.randseed + opts.fid * opts.reruns + opts.runidx)
    # fit the model
    model = get_aad_model(x, opts, random_state)
    model.fit(x)
    model.init_weights(init_type=opts.init)
    return model


def prepare_aad_model(x, y, opts):
    if opts.load_model and opts.modelfile != "" and os.path.isfile(opts.modelfile):
        logger.debug("Loading model from file %s" % opts.modelfile)
        model = load_aad_model(opts.modelfile)
    else:
        model = train_aad_model(opts, x)

    if is_forest_detector(model.detector_type):
        logger.debug("total #nodes: %d" % (len(model.all_regions)))
    if False:
        if model.w is not None:
            logger.debug("w:\n%s" % str(list(model.w)))
        else:
            logger.debug("model weights are not set")
    return model


def prepare_stream_anomaly_detector(stream, opts):
    """Prepares an instance of the StreamingAnomalyDetector

    :param stream: DataStream
    :param opts: AadOpts
    :param pretrain: boolean
        If True, then treats the first window of data as fully *LABELED* and updates
            the weights with the labeled data. Next, fetches the next window of data
            as fully *UNLABELED* and updates tree structure if needed.
        If False, then treats the first window of data as fully unlabeled.
    :param n_pretrain: int
        Number of times to run the weight update if pre-training is required.
    :return: StreamingAnomalyDetector
    """
    training_set = stream.read_next_from_stream(opts.stream_window)
    X_train, y_train, ids = training_set.x, training_set.y, training_set.ids
    model = prepare_aad_model(X_train, y_train, opts)  # initial model training
    if opts.pretrain:
        # first window pre-trains the model as fully labeled set
        sad = StreamingAnomalyDetector(stream, model,
                                       labeled_x=X_train, labeled_y=y_train, labeled_ids=ids,
                                       max_buffer=opts.stream_window, opts=opts)

        # second window is treated as fully unlabeled
        instances = sad.get_next_from_stream(sad.max_buffer,
                                             transform=(not opts.allow_stream_update))
        if instances is not None:
            model_updated = False
            if opts.allow_stream_update:
                model_updated = sad.update_model_from_buffer(transform=True)
            sad.move_buffer_to_unlabeled()
            if model_updated:
                sad.update_weights_with_no_feedback()
            sad.feature_ranges = get_sample_feature_ranges(instances.x)
        else:
            sad.feature_ranges = get_sample_feature_ranges(X_train)
        sad.init_query_state()
    else:
        # first window is treated as fully unlabeled
        sad = StreamingAnomalyDetector(stream, model,
                                       unlabeled_x=X_train, unlabeled_y=y_train, unlabeled_ids=ids,
                                       max_buffer=opts.stream_window, opts=opts)
        sad.feature_ranges = get_sample_feature_ranges(X_train)
        sad.init_query_state()
    return sad


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

    X_full, y_full = read_data_as_matrix(opts)

    logger.debug("loaded file: (%s) %s" % (str(X_full.shape), opts.datafile))
    logger.debug("results dir: %s" % opts.resultsdir)

    all_num_seen = None
    all_num_not_seen = None
    all_num_seen_baseline = None
    all_queried = None
    all_window = None
    all_window_baseline = None

    opts.fid = 1
    for runidx in opts.get_runidxs():
        tm_run = Timer()
        opts.set_multi_run_options(opts.fid, runidx)

        stream = DataStream(X_full, y_full, IdServer(initial=0))
        # from aad.malware_aad import MalwareDataStream
        # stream = MalwareDataStream(X_full, y_full, IdServer(initial=0))

        sad = prepare_stream_anomaly_detector(stream, opts)

        if sad.unlabeled is None:
            logger.debug("No instances to label")
            continue

        iter = 0
        seen = np.zeros(0, dtype=int)
        n_unlabeled = np.zeros(0, dtype=int)
        seen_baseline = np.zeros(0, dtype=int)
        queried = np.zeros(0, dtype=int)
        stream_window_tmp = np.zeros(0, dtype=int)
        stream_window_baseline = np.zeros(0, dtype=int)
        stop_iter = False
        while not stop_iter:
            iter += 1

            tm = Timer()
            seen_, seen_baseline_, queried_, queried_baseline_, n_unlabeled_ = sad.run_feedback()

            # gather metrics...
            seen = append(seen, seen_)
            n_unlabeled = append(n_unlabeled, n_unlabeled_)
            seen_baseline = append(seen_baseline, seen_baseline_)
            queried = append(queried, queried_)
            stream_window_tmp = append(stream_window_tmp, np.ones(len(seen_)) * iter)
            stream_window_baseline = append(stream_window_baseline, np.ones(len(seen_baseline_)) * iter)

            # get the next window of data from stream and transform features...
            # Note: Since model update will automatically transform the data, we will
            # not transform while reading from stream. If however, the model is not
            # to be updated, then we transform the data while reading from stream
            instances = sad.get_next_from_stream(sad.max_buffer,
                                                 transform=(not opts.allow_stream_update))
            if instances is None or iter >= opts.max_windows or len(queried) >= opts.budget:
                if iter >= opts.max_windows:
                    logger.debug("Exceeded %d iters; exiting stream read..." % opts.max_windows)
                stop_iter = True
            else:
                model_updated = False
                if opts.allow_stream_update:
                    model_updated = sad.update_model_from_buffer(transform=True)

                sad.move_buffer_to_unlabeled()

                if model_updated:
                    sad.update_weights_with_no_feedback()

            logger.debug(tm.message("Stream window [%d]: algo [%d/%d]; baseline [%d/%d]; unlabeled anoms [%d]: " %
                                    (iter, int(np.sum(seen)), len(seen),
                                     int(np.sum(seen_baseline)), len(seen_baseline),
                                     int(np.sum(sad.unlabeled.y)))))

        # retained = int(np.sum(sad.unlabeled_y)) if sad.unlabeled_y is not None else 0
        # logger.debug("Final retained unlabeled anoms: %d" % retained)

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

        queried_ids = np.ones(len(num_seen_baseline) + 2, dtype=num_seen_tmp.dtype) * -1
        queried_ids[0:2] = [opts.fid, opts.runidx]
        # IMPORTANT:: The queried indexes are output as 1-indexed (NOT zero-indexed)
        # logger.debug("queried:\n%s\n%s" % (str(list(queried)), str(list(y_full[queried]))))
        queried_ids[2:(2 + len(queried))] = queried + 1

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
        all_queried = rbind(all_queried, matrix(queried_ids, nrow=1))
        all_window = rbind(all_window, matrix(stream_window, nrow=1))
        all_window_baseline = rbind(all_window_baseline, matrix(stream_window_baseline, nrow=1))

        logger.debug(tm_run.message("Completed runidx: %d" % runidx))

    results = SequentialResults(num_seen=all_num_seen,
                                num_not_seen=all_num_not_seen,
                                true_queried_indexes=all_queried,
                                num_seen_baseline=all_num_seen_baseline,
                                # true_queried_indexes_baseline=all_queried_baseline,
                                stream_window=all_window,
                                stream_window_baseline=all_window_baseline,
                                aucs=None)
    write_sequential_results_to_csv(results, opts)


if __name__ == "__main__":
    aad_stream()
