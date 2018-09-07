from __future__ import division

import logging
from sklearn.ensemble import IsolationForest

from aad.random_split_trees import *
from common.utils import *


"""
Builds isolation forest instances with partitions of features so that each
feature partition acts as a separate view.

TODO: Need to extend this to other tree-based models (HSTrees, RSForest), and
    perhaps to other ensemble algorithms (LODA) as well.
"""


class IForestMultiviewTree(RandomSplitTree):
    """Create ArrTree instance from an IsolationForest tree"""
    def __init__(self,
                 n_features=0,
                 splitter=None,
                 max_depth=10,
                 max_features=1,
                 random_state=None,
                 update_type=TREE_UPD_OVERWRITE,
                 incremental_update_weight=0.5,
                 ifor_tree=None):
        RandomSplitTree.__init__(self,
                                 splitter=splitter,
                                 max_depth=max_depth,
                                 max_features=max_features,
                                 random_state=random_state,
                                 update_type=update_type,
                                 incremental_update_weight=incremental_update_weight)
        self.tree_ = ArrTree(n_features=n_features, max_depth=max_depth, update_type=update_type,
                             incremental_update_weight=incremental_update_weight)
        self.tree_.children_left = np.array(ifor_tree.children_left)
        self.tree_.children_right = np.array(ifor_tree.children_right)
        self.tree_.feature = np.array(ifor_tree.feature)
        self.tree_.threshold = np.array(ifor_tree.threshold)
        self.tree_.n_node_samples = np.array(ifor_tree.n_node_samples)
        self.tree_.node_count = len(self.tree_.feature)
        self.tree_.nodes = np.arange(self.tree_.node_count, dtype=int)
        self.tree_.v = np.zeros(self.tree_.node_count, dtype=np.float32)
        self.tree_.acc_log_v = np.zeros(self.tree_.node_count, dtype=np.float32)

    def get_splitter(self, splitter=None):
        raise NotImplementedError("method not supported")

    def decision_function(self, X):
        raise NotImplementedError("method not supported")


class IForestMultiview(RandomSplitForest):

    def __init__(self,
                 feature_partitions=None,
                 n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 replace_frac=0.2,
                 random_state=None,
                 verbose=0):
        RandomSplitForest.__init__(self, n_estimators=n_estimators,
                                   max_samples=max_samples,
                                   max_features=max_features,
                                   bootstrap=bootstrap,
                                   n_jobs=n_jobs,
                                   random_state=random_state,
                                   verbose=verbose)
        assert feature_partitions is not None and len(feature_partitions) > 0
        self.feature_partitions = feature_partitions
        self.n_estimators_view = None
        self.contamination = contamination
        # The fraction of trees replaced when new window of data arrives
        # TODO: Model update by tree replacement feature is not supported yet...
        self.replace_frac = replace_frac
        self.ifor = None
        self.ifor_static = None
        self.ifor_dynamic = None
        self.estimators_features_ = None
        self.buffer = None
        self.updated = False
        self.n_estimators_view = get_tree_partitions(self.n_estimators, len(self.feature_partitions))

    def fit(self, X, y=None, sample_weight=None):
        logger.debug("IForestMultiview feature_partitions: %s" % str(list(self.feature_partitions)))
        self._fit(X, y, sample_weight, self.max_depth)
        logger.debug("IForestMultiview n_estimators: %d" % len(self.estimators_))
        self.updated = False

    def _multiview_fit(self, X, y, feature_partitions, n_estimators_view):
        n_features = X.shape[1]

        estimators_group = []
        feature_offset = 0
        logger.debug("IForestMultiview n_estimators_view: %s" % str(list(n_estimators_view)))
        for n_feats, n_est_ in zip(feature_partitions, n_estimators_view):
            estimators = []
            X_ = X[:, feature_offset:(feature_offset+n_feats)]

            if n_est_ > 0:
                # contruct isolation forest for the view containing just the feature subset
                ifor_ = IsolationForest(n_estimators=n_est_,
                                        max_samples=self.max_samples,
                                        contamination=self.contamination,
                                        max_features=self.max_features,
                                        bootstrap=self.bootstrap,
                                        n_jobs=self.n_jobs,
                                        random_state=self.random_state,
                                        verbose=self.verbose)
                ifor_.fit(X_, y, sample_weight=None)

                for tree in ifor_.estimators_:
                    # The IsolationForest trees contain read-only properties. We copy
                    # over all the properties to our custom tree structure so that we
                    # can modify them if needed.
                    ifor_mv_estimator = IForestMultiviewTree(n_features=n_features, ifor_tree=tree.tree_)

                    # adjust the feature indexes at the tree nodes.
                    ifor_mv_estimator.tree_.feature += feature_offset

                    estimators.append(ifor_mv_estimator)

            estimators_group.append(estimators)
            feature_offset += n_feats

        return estimators_group

    def _fit(self, X, y, max_samples, max_depth, sample_weight=None):
        self.estimators_ = []
        self.estimators_features_ = None  # not used, hence ignored...

        estimators_group = self._multiview_fit(X, None, self.feature_partitions, self.n_estimators_view)
        for estimators in estimators_group:
            if len(estimators) > 0:
                self.estimators_.extend(estimators)

    def decision_function(self, X):
        if self.updated:
            logger.debug("WARN: The underlying isolation forest was updated and " +
                         "using calling decision_function() on it will likely return inconsistent results.")
        return self.ifor.decision_function(X)

    def supports_streaming(self):
        return True

    def add_samples(self, X, current=True):
        if current:
            raise ValueError(
                "IForestMultiview does not support adding to current instance set.")
        if self.buffer is None:
            self.buffer=X
        else:
            self.buffer=np.vstack([self.buffer, X])

    def update_trees_by_replacement(self, X=None, replace_trees=None):
        if self.estimators_ is None:
            raise RuntimeError("Forest not trained")

        if X is None:
            X = self.buffer
        if X is None:
            logger.warning("No new data for update")
            return None

        n_estimators = np.zeros(len(self.n_estimators_view), dtype=int)
        discard_set = set(replace_trees)
        old_replaced_idxs = list()
        old_retained_idxs = list()
        retained_trees = list()
        start_tree = 0
        for i, n_trees_view in enumerate(self.n_estimators_view):
            retained_in_group = list()
            rep = list()
            ret = list()
            end_tree = start_tree + n_trees_view
            for j in range(start_tree, end_tree):
                if j not in discard_set:
                    retained_in_group.append(self.estimators_[j])
                    ret.append(j)
                else:
                    n_estimators[i] += 1
                    rep.append(j)
            retained_trees.append(retained_in_group)
            old_replaced_idxs.append(np.array(rep, dtype=int))
            old_retained_idxs.append(np.array(ret, dtype=int))
            start_tree = end_tree

        logger.debug("Number of new trees per group: %s" % (str(list(n_estimators))))
        new_trees = self._multiview_fit(X, None, self.feature_partitions, n_estimators)

        self.estimators_ = list()
        for i, n_estimators_group in enumerate(self.n_estimators_view):
            if len(retained_trees[i]) + len(new_trees[i]) != n_estimators_group:
                raise RuntimeError("retained_trees (%d) and new_trees (%d) do not add to expected (%d)" %
                                   (len(retained_trees[i]), len(new_trees[i]), n_estimators_group))
            self.estimators_.extend(retained_trees[i])
            self.estimators_.extend(new_trees[i])

        self.updated = True
        self.buffer = None

        return old_replaced_idxs, old_retained_idxs, new_trees

    def update_model_from_stream_buffer(self, replace_trees=None):
        return self.update_trees_by_replacement(X=self.buffer, replace_trees=replace_trees)
