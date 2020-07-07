import numpy as np

from .aad_globals import QUERY_EUCLIDEAN_DIST_MIN, QUERY_EUCLIDEAN_DIST_MEAN, get_first_vals_not_marked
from .query_model import Query


"""
Euclidean Query model: Diverse query strategy based on euclidean distance.
    Get the query instances such that distance between them is maximized
"""


class DistanceCache(object):
    def __init__(self, size):
        self.size = size
        self.n_dists = 0
        self.dists = [None] * size

    def add_dist(self, i, j, dist):
        if self.dists[i] is None:
            self.dists[i] = {}
        if j not in self.dists[i]:
            self.n_dists += 1
        self.dists[i][j] = dist

    def get_dist(self, i, j):
        return self.dists[i][j]

    def has_dist(self, i, j):
        if self.dists[i] is None or (j not in self.dists[i]):
            return False
        return True

    def __len__(self):
        return self.size


def get_mean_euclidean_distance(all_instances, compare_to_indexes, ref_index, cached_distances=None):
    """ Return the average euclidean distance from instance at ref_index to instances in compare_to_indexes

    :param all_instances: np.ndarray(float)
        All instances in *original* feature space
    :param compare_to_indexes: list(int)
        Instance indexes for instances to which mean euclidean distance needs to be computed
    :param ref_index: int
        Index of instance from which the euclidean distance will be computed
    :returns float
    """
    if len(compare_to_indexes) == 0:
        return 0.
    mean_dist = 0.
    inst = all_instances[ref_index]
    for i in compare_to_indexes:
        if cached_distances is not None and cached_distances.has_dist(ref_index, i):
            dist = cached_distances.get_dist(ref_index, i)
        else:
            diff = all_instances[i] - inst
            dist = np.sum(diff ** 2)
            if cached_distances is not None:
                cached_distances.add_dist(ref_index, i, dist)
        mean_dist += dist
    mean_dist = mean_dist / len(compare_to_indexes)
    return mean_dist


def get_min_euclidean_distance(all_instances, compare_to_indexes, ref_index, cached_distances=None):
    """ Return the minimum euclidean distance from instance at ref_index to instances in compare_to_indexes

    :param all_instances: np.ndarray(float)
        All instances in *original* feature space
    :param compare_to_indexes: list(int)
        Instance indexes for instances to which mean euclidean distance needs to be computed
    :param ref_index: int
        Index of instance from which the euclidean distance will be computed
    :returns float
    """
    if len(compare_to_indexes) == 0:
        return 0.
    min_dist = np.Inf
    inst = all_instances[ref_index]
    for i in compare_to_indexes:
        if cached_distances is not None and cached_distances.has_dist(ref_index, i):
            dist = cached_distances.get_dist(ref_index, i)
        else:
            diff = all_instances[i] - inst
            dist = np.sum(diff ** 2)
            if cached_distances is not None:
                cached_distances.add_dist(ref_index, i, dist)
        min_dist = min(min_dist, dist)
    return min_dist


def filter_by_euclidean_distance(x, instance_ids, init_selected=None, n_select=3,
                                 dist_type=QUERY_EUCLIDEAN_DIST_MIN):
    """ Return the n most diverse instances based on euclidean distances
    :param x: np.ndarray(float)
        All instances in *original* feature space
    :param instance_ids: np.array(int)
        The indexes to instances in the order of anomaly scores
    :param n_select: int
        Maximum number of instances to output
    :param dist_type: int
        The distance type to use {QUERY_EUCLIDEAN_DIST_MEAN | QUERY_EUCLIDEAN_DIST_MIN}
    """
    selected_instances = list()
    n_init_selected = 0
    if init_selected is not None:
        selected_instances.extend(init_selected)
        n_init_selected = len(init_selected)
    candidates = np.array(instance_ids, dtype=int)
    cached_distances = DistanceCache(x.shape[0])
    while len(selected_instances) - n_init_selected < n_select and len(candidates) > 0:
        # find average distance to all other selected instances
        dists = np.zeros(len(candidates), dtype=np.float32)
        for i, inst in enumerate(candidates):
            if dist_type == QUERY_EUCLIDEAN_DIST_MEAN:
                dists[i] = get_mean_euclidean_distance(x, selected_instances, inst,
                                                       cached_distances=cached_distances)
            elif dist_type == QUERY_EUCLIDEAN_DIST_MIN:
                dists[i] = get_min_euclidean_distance(x, selected_instances, inst,
                                                      cached_distances=cached_distances)
            else:
                raise ValueError("invalid dist_type %d" % dist_type)
        # sort in descending order and retain input order in case of equal values
        # so that most anomalous are preferred
        sorted_inst_indexes = np.argsort(-dists, kind='mergesort')
        selected_index = sorted_inst_indexes[0]
        selected = candidates[selected_index]
        selected_instances.append(selected)
        candidates = np.delete(candidates, selected_index)
    # logger.debug("Euclidean selected:\n%s\namong\n%s" % (str(list(selected_instances)), str(instance_ids)))
    return selected_instances


class QueryTopDiverseByEuclideanDistance(Query):
    """ Meaningful for only batch queries """

    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):

        ordered_indexes = kwargs.get("ordered_indexes")
        queried_items = kwargs.get("queried_items")
        remaining_budget = kwargs.get("remaining_budget")
        ensemble = kwargs.get("ensemble")
        if self.opts.n_explore < self.opts.num_query_batch:
            raise ValueError("Error: option n_explore (%d) less than n_batch (%d)" %
                             (self.opts.n_explore, self.opts.num_query_batch))
        items = get_first_vals_not_marked(ordered_indexes, queried_items, start=0,
                                          n=self.opts.n_explore)
        if len(items) == 0:
            return None

        filtered_items = filter_by_euclidean_distance(ensemble.samples, items,
                                                      n_select=min(remaining_budget, self.opts.num_query_batch),
                                                      dist_type=self.opts.query_euclidean_dist_type)

        return filtered_items


