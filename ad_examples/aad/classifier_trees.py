import numpy as np

from sklearn.tree import DecisionTreeClassifier

from ..common.utils import DTClassifier
from .aad_globals import (
    IFOR_SCORE_TYPE_CONST, ENSEMBLE_SCORE_LINEAR, AAD_IFOREST, INIT_UNIF
)
from .aad_base import Aad
from .random_split_trees import TREE_UPD_OVERWRITE
from .forest_aad_detector import AadForest


class ClassifierForest(object):
    def __init__(self, trees):
        self.estimators_ = trees
        self.n_estimators = len(trees)

    def get_node_ids(self, X, getleaves=True):
        if not getleaves:
            raise ValueError("Operation supported for leaf level only")
        forest_nodes = list()
        for estimator in self.estimators_:
            tree_nodes = estimator.apply(X)
            forest_nodes.append(tree_nodes)
        return forest_nodes


class DecisionTreeAadWrapper(AadForest):
    """ Extracts regions from a decision tree classifier which cover selected instances

    Since much of the functionality is already in AadForest, we merely initialize the
    internal structures in the constructor. Later, the requisite functions in AadForest
    will be invoked.
    """

    def __init__(self,
                 x, y,
                 max_depth=10,
                 score_type=IFOR_SCORE_TYPE_CONST,
                 ensemble_score=ENSEMBLE_SCORE_LINEAR,
                 random_state=None,
                 detector_type=AAD_IFOREST):
        Aad.__init__(self, detector_type, ensemble_score, random_state)

        self.max_depth = max_depth
        self.n_estimators = 1
        self.max_samples = x.shape[0]
        self.tree_update_type = TREE_UPD_OVERWRITE
        self.tree_incremental_update_weight = None
        self.forest_replace_frac = None
        self.feature_partitions = None

        self.score_type = score_type

        self.add_leaf_nodes_only = True

        # store all regions grouped by tree
        self.regions_in_forest = None

        # store all regions in a flattened list (ungrouped)
        self.all_regions = None

        # store maps of node index to region index for all trees
        self.all_node_regions = None

        # scores for each region
        self.d = None

        self.decision_tree = DTClassifier.fit(x, y, self.max_depth)

        # anomalies are labeled '1'
        self.anomaly_class_index = np.where(self.decision_tree.clf.classes_ == 1)[0][0]
        # logger.debug("anomaly_class_index: %d" % self.anomaly_class_index)

        self.clf = ClassifierForest([self.decision_tree.clf])

        self._init_structures()
        self.init_weights(init_type=INIT_UNIF)

    def _init_structures(self):
        self.regions_in_forest = []
        self.all_regions = []
        self.all_node_regions = []
        region_id = 0
        for i in range(len(self.clf.estimators_)):
            regions = self.extract_leaf_regions_from_tree(self.clf.estimators_[i],
                                                          self.add_leaf_nodes_only)
            self.regions_in_forest.append(regions)
            self.all_regions.extend(regions)
            node_regions = {}
            for region in regions:
                node_regions[region.node_id] = region_id
                region_id += 1  # this will monotonously increase across trees
            self.all_node_regions.append(node_regions)
            # print "%d, #nodes: %d" % (i, len(regions))
        self.d, _, _ = self.get_region_scores_wrapper()
        # self.w = self.get_uniform_weights()
        self.w_unif_prior = self.get_uniform_weights()

    def get_region_scores_wrapper(self):
        """Larger values mean more anomalous"""
        if self.clf.n_estimators == 1 and isinstance(self.clf.estimators_[0], DecisionTreeClassifier):
            return self.get_region_scores_dt(self.all_regions)
        return self.get_region_scores(self.all_regions)

    def get_region_scores_dt(self, all_regions):
        """ Assign the probability of being anomaly in a region as the score for that region

        Note: Larger values mean more anomalous
        """
        d = np.zeros(len(all_regions), dtype=np.float64)
        for i, region in enumerate(all_regions):
            # the anomaly class '1' probability is the region score
            dists = region.value[0]
            tot = np.sum(dists)
            if tot == 0:
                tot = 1.
            # logger.debug(region.value)
            d[i] = dists[self.anomaly_class_index] * 1. / tot
        # logger.debug("d:\n%s" % str(list(d)))
        return d, None, None

    def fit(self, x):
        raise NotImplementedError("fit() not implemented for DecisionTreeRegionExtractor")


class RandomForestAadWrapper(DecisionTreeAadWrapper):

    def __init__(self, x, y, clf, score_type=IFOR_SCORE_TYPE_CONST,
                 ensemble_score=ENSEMBLE_SCORE_LINEAR,
                 random_state=None,
                 detector_type=AAD_IFOREST):
        Aad.__init__(self, detector_type, ensemble_score, random_state)

        self.max_depth = clf.max_depth
        self.n_estimators = len(clf.estimators_)
        self.max_samples = x.shape[0]
        self.tree_update_type = TREE_UPD_OVERWRITE
        self.tree_incremental_update_weight = None
        self.forest_replace_frac = None
        self.feature_partitions = None

        self.score_type = score_type

        self.add_leaf_nodes_only = True

        # store all regions grouped by tree
        self.regions_in_forest = None

        # store all regions in a flattened list (ungrouped)
        self.all_regions = None

        # store maps of node index to region index for all trees
        self.all_node_regions = None

        # scores for each region
        self.d = None

        self.clf = ClassifierForest(clf.estimators_)
        self._init_structures()
        self.init_weights(init_type=INIT_UNIF)

    def fit(self, x):
        raise NotImplementedError("fit() not implemented for RandomForestRegionExtractor")