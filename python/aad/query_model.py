import importlib
from aad.aad_globals import *


class Query(object):
    def __init__(self, opts=None, **kwargs):
        self.opts = opts
        self.test_indexes = None

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):
        pass

    @staticmethod
    def get_custom_query_model(opts, **kwargs):
        module_name = opts.query_module_name
        class_name = opts.query_class_name
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(opts, **kwargs)

    @staticmethod
    def get_initial_query_state(querytype, opts, **kwargs):
        if querytype == QUERY_DETERMINISIC:
            return QueryTop(opts=opts, **kwargs)
        elif querytype == QUERY_TOP_RANDOM:
            return QueryTopRandom(opts=opts, **kwargs)
        elif querytype == QUERY_QUANTILE:
            return QueryQuantile(opts=opts, **kwargs)
        elif querytype == QUERY_RANDOM:
            return QueryRandom(opts=opts, **kwargs)
        elif querytype == QUERY_CUSTOM_MODULE:
            return Query.get_custom_query_model(opts, **kwargs)
        elif querytype == QUERY_EUCLIDEAN:
            # doing it this round-about way else there will be a circular module dependency
            module = importlib.import_module("aad.query_model_euclidean")
            class_ = getattr(module, "QueryTopDiverseByEuclideanDistance")
            return class_(opts, **kwargs)
        elif querytype == QUERY_SUBSPACE_EUCLIDEAN:
            obj = Query.get_custom_query_model(opts, **kwargs)
            obj.order_by_euclidean_diversity = True
            return obj
        else:
            raise ValueError("Invalid/unsupported query type %d" % (querytype,))


class QueryTop(Query):
    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):
        ordered_indexes = kwargs.get("ordered_indexes")
        queried_items = kwargs.get("queried_items")
        items = get_first_vals_not_marked(ordered_indexes, queried_items, start=0,
                                          n=self.opts.num_query_batch)
        if len(items) == 0:
            return None
        return items


class QueryTopRandom(Query):
    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):
        """Select n items from top opts.n_explore ranked items"""
        ordered_indexes = kwargs.get("ordered_indexes")
        queried_items = kwargs.get("queried_items")
        # logger.debug("n_explore: %d, n: %d" % (self.opts.n_explore, self.opts.num_query_batch))
        choose_from_items = get_first_vals_not_marked(ordered_indexes, queried_items, start=0,
                                                      n=self.opts.n_explore)
        if len(choose_from_items) == 0:
            return None
        q = sample(range(self.opts.n_explore), self.opts.num_query_batch)
        return choose_from_items[q]


class QueryQuantile(Query):
    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):
        pass


class QueryRandom(Query):
    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):
        maxpos = kwargs.get("maxpos")
        ordered_indexes = kwargs.get("ordered_indexes")
        queried_items = kwargs.get("queried_items")
        q = sample(range(maxpos), self.opts.num_query_batch)
        items = get_first_vals_not_marked(ordered_indexes, queried_items, start=q,
                                          n=self.opts.num_query_batch)
        return items

