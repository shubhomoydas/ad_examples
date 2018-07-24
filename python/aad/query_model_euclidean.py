from aad.query_model import *
from aad.forest_aad_detector import *
from aad.forest_description import *


"""
Euclidean Query model: Diverse query strategy based on euclidean distance.
    Get the query instances such that distance between them is maximized
"""


class QueryTopDiverseByEuclideanDistance(Query):
    """ Meaningful for only batch queries """

    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)

    def update_query_state(self, **kwargs):
        pass

    def get_average_euclidean_distance(self, all_instances, compare_to_indexes, ref_index):
        """ Return the average euclidean distance from instance at ref_index to instances in compare_to_inedxes

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
            diff = all_instances[i] - inst
            mean_dist += np.sum(diff ** 2)
        mean_dist = mean_dist / len(compare_to_indexes)
        return mean_dist

    def filter_by_euclidean_distance(self, x, instance_ids, n_select=3):
        """ Return the n most diverse instances based on euclidean distances
        :param x: np.ndarray(float)
            All instances in *original* feature space
        :param instance_ids: np.array(int)
            The indexes to instances in the order of anomaly scores
        :param n_select: int
            Maximum number of instances to output
        """
        selected_instances = list()
        candidates = np.array(instance_ids, dtype=int)
        while len(selected_instances) < n_select and len(candidates) > 0:
            # find average distance to all other selected instances
            dists = np.zeros(len(candidates), dtype=np.float32)
            for i, inst in enumerate(candidates):
                dists[i] = self.get_average_euclidean_distance(x, selected_instances, inst)
            # sort in descending order and retain input order in case of equal values
            # so that most anomalous are preferred
            sorted_inst_indexes = np.argsort(-dists, kind='mergesort')
            selected_instances.append(candidates[sorted_inst_indexes[0]])
            candidates = np.delete(candidates, sorted_inst_indexes[0])
        logger.debug("Euclidean selected: %s in %s" % (str(list(selected_instances)), str(instance_ids)))
        return selected_instances

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

        filtered_items = self.filter_by_euclidean_distance(ensemble.samples, items,
                                                           n_select=min(remaining_budget, self.opts.num_query_batch))

        return filtered_items


