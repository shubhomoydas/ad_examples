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
    def get_initial_query_state(querytype, opts, **kwargs):
        if querytype == QUERY_DETERMINISIC:
            return QueryTop(opts=opts, **kwargs)
        elif querytype == QUERY_TOP_RANDOM:
            return QueryTopRandom(opts=opts, **kwargs)
        elif querytype == QUERY_QUANTILE:
            return QueryQuantile(opts=opts, **kwargs)
        elif querytype == QUERY_RANDOM:
            return QueryRandom(opts=opts, **kwargs)
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
        # n = kwargs.get("n", 1)
        n = 1
        items = get_first_vals_not_marked(ordered_indexes, queried_items, start=0, n=n)
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
        n_explore = self.opts.n_explore
        n = kwargs.get("n", 1)
        # logger.debug("n_explore: %d, n: %d" % (n_explore, n))
        choose_from_items = get_first_vals_not_marked(ordered_indexes, queried_items, start=0, n=n_explore)
        if len(choose_from_items) == 0:
            return None
        q = sample(range(n_explore), n)
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
        n = kwargs.get("n", 1)
        q = sample(range(maxpos), n)
        items = get_first_vals_not_marked(ordered_indexes, queried_items, start=q, n=n)
        return items

