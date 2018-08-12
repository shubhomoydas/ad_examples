from aad.aad_support import *


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
        self.d, _, _ = self.get_region_scores(self.all_regions)
        # self.w = self.get_uniform_weights()
        self.w_unif_prior = self.get_uniform_weights()

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