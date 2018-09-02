from aad.query_model import *
from aad.query_model_euclidean import filter_by_euclidean_distance
from aad.forest_aad_detector import *
from aad.forest_description import *


"""
Custom Query models kept here to decouple the base APIs from custom APIs.
"""


class QueryTopDiverseSubspace(Query):
    """ Batch queries """

    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)
        self.order_by_euclidean_diversity = False

    def update_query_state(self, **kwargs):
        pass

    def filter_by_diversity(self, instance_ids, region_memberships, queried=None, n_select=3):
        """ Return the n most diverse instances
        :param instance_ids: np.array(int)
            The indexes to instances in the order they appear in the training dataset.
        :param region_memberships: np.ndarray(int)
            Bit-map of region memberships
        :param queried: np.array(int)
            The instance indexes (might be included in instance_ids) whose labels have
            already been queried and therefore should be ignored in output
        :param n_select: int
            Maximum number of instances to output
        """
        regions = region_memberships
        # logger.debug("regions: %s\n%s" % (str(regions.shape), str(regions)))
        selected_regions = np.zeros(regions.shape[1])
        selected_instances = list()
        if queried is not None and len(queried) > 0:
            # record all region memberships for instances which have already been queried
            queried_set = set(queried)
            ignore_indexes = list()
            for i, inst in enumerate(instance_ids):
                if inst in queried_set:
                    ignore_indexes.append(i)
                    selected_regions = selected_regions + regions[i, :]
            if len(ignore_indexes) > 0:
                # remove all queried instances from consideration further down
                regions = np.delete(regions, ignore_indexes, axis=0)
                instance_ids = np.delete(instance_ids, ignore_indexes)
            if False:
                logger.debug("queried:\n%s\nselected_regions:\n%s\nIgnored:\n%s" %
                             (str(queried), str(selected_regions), str(ignore_indexes)))
        curr_inst_ids = instance_ids
        for i in range(n_select):
            if len(selected_instances) == n_select or regions.shape[0] == 0:
                break
            regs = np.dot(regions, selected_regions)
            # logger.debug("regs:\n%s" % str(regs))
            # We need stable sort for which we use mergesort.
            # The parameter instance_ids contains instance indexes in
            # sorted order of anomaly scores.
            sorted_inst_indexes = np.argsort(regs, kind='mergesort')
            inst = curr_inst_ids[sorted_inst_indexes[0]]
            selected_regions = selected_regions + regions[sorted_inst_indexes[0], :]
            # logger.debug("selected_regions:\n%s" % str(selected_regions))
            curr_inst_ids = np.delete(curr_inst_ids, sorted_inst_indexes[0])
            regions = np.delete(regions, sorted_inst_indexes[0], axis=0)
            selected_instances.append(inst)
        return selected_instances

    def get_next_query(self, **kwargs):

        consider_queried = False

        ordered_indexes = kwargs.get("ordered_indexes")
        queried_items = kwargs.get("queried_items")
        remaining_budget = kwargs.get("remaining_budget")
        ensemble = kwargs.get("ensemble")
        feature_ranges = kwargs.get("feature_ranges")
        model = kwargs.get("model")
        if self.opts.n_explore < self.opts.num_query_batch:
            raise ValueError("Error: option n_explore (%d) less than n_batch (%d)" %
                             (self.opts.n_explore, self.opts.num_query_batch))
        items = get_first_vals_not_marked(ordered_indexes, queried_items, start=0,
                                          n=self.opts.n_explore)
        if len(items) == 0:
            return None

        if is_forest_detector(self.opts.detector_type) and True:
            # feature_ranges will be used to compute volumes
            if feature_ranges is None:
                feature_ranges = get_sample_feature_ranges(ensemble.samples)

            instance_ids = items
            queried_anom_indexes = None
            if consider_queried and queried_items is not None and len(queried_items) > 0:
                queried_items = np.array(queried_items, dtype=int)
                # Filter only the labeled anomalies for diversity computation
                queried_anom_indexes = np.where(ensemble.labels[queried_items] == 1)[0]
                if len(queried_anom_indexes) > 0:
                    if len(queried_anom_indexes) > 50:
                        # The ILP to select compact regions could be expensive.
                        # Therefore, take a subsample.
                        np.random.shuffle(queried_anom_indexes)
                        queried_anom_indexes = queried_anom_indexes[0:50]
                    instance_ids = append(instance_ids, queried_items[queried_anom_indexes])

            reg_idxs = get_regions_for_description(ensemble.samples, instance_indexes=instance_ids,
                                                   model=model, n_top=self.opts.describe_n_top)
            volumes = get_region_volumes(model, reg_idxs, feature_ranges)
            # logger.debug("reg_idxs:%d\n%s" % (len(reg_idxs), str(list(reg_idxs))))
            compact_reg_idxs = get_compact_regions(ensemble.samples, instance_indexes=instance_ids,
                                                   region_indexes=reg_idxs, model=model, volumes=volumes,
                                                   p=self.opts.describe_volume_p)
            if True:
                logger.debug("#reg_idxs:%d, #compact regions:%d, #queried_anoms: %d, p: %d, n_top: %d" %
                             (len(reg_idxs), len(compact_reg_idxs),
                              0 if queried_anom_indexes is None else len(queried_anom_indexes),
                              self.opts.describe_volume_p, self.opts.describe_n_top))
                # logger.debug("compact regions:%d\n%s" % (len(reg_idxs), str(list(reg_idxs))))

            instance_ids, region_memberships = get_region_memberships(ensemble.samples,
                                                                      instance_indexes=instance_ids,
                                                                      region_indexes=compact_reg_idxs,
                                                                      model=model)

            if self.order_by_euclidean_diversity:
                # arrange instance_ids by euclidean diversity first
                # logger.debug("ordering by euclidean diversity")
                init_ordered_items = filter_by_euclidean_distance(ensemble.samples, instance_ids,
                                                                  n_select=len(instance_ids),
                                                                  dist_type=QUERY_EUCLIDEAN_DIST_MIN)
                logger.debug("\ninstance_ids:\n%s\ninit_ordered_items:\n%s" % (str(instance_ids), str(init_ordered_items)))
            else:
                init_ordered_items = instance_ids

            filtered_items = self.filter_by_diversity(init_ordered_items, region_memberships,
                                                      queried=queried_items,
                                                      n_select=min(remaining_budget, self.opts.num_query_batch))
            # logger.debug("\nitems:\n%s\nfiltered_items:\n%s\ninstance_ids:\n%s" % (str(items), str(filtered_items), str(instance_ids)))
        else:
            raise RuntimeError("QueryTopBatch supported only for forest-based models")

        return filtered_items


