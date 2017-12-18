import numpy as np
import scipy as sp

from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix, vstack

from common.utils import *


class IdServer(object):
    def __init__(self, initial=0):
        self.curr = initial

    def get_next(self, n=1):
        """Returns n ids and adjusts self.curr"""
        ids = np.arange(self.curr, self.curr+n)
        self.curr += n
        return ids


class DataStream(object):
    def __init__(self, X, y=None, id_server=None):
        self.X = X
        self.y = y
        self.id_server = id_server

    def read_next_from_stream(self, n=1):
        """Returns first n instances from X and removes these instances from X"""
        n = min(n, self.X.shape[0])
        # logger.debug("DataStream.read_next_from_stream n: %d" % n)
        if n == 0:
            return None
        mask = np.zeros(self.X.shape[0], dtype=bool)
        mask[np.arange(n)] = True
        instances = self.X[mask]
        self.X = self.X[~mask]
        labels = None
        if self.y is not None:
            labels = self.y[mask]
            self.y = self.y[~mask]
        ids = None
        if self.id_server is not None:
            ids = self.id_server.get_next(n)
        # logger.debug("DataStream.read_next_from_stream instances: %s" % str(instances.shape))
        return InstanceList(instances, labels, ids)

    def empty(self):
        return self.X is None or self.X.shape[0] == 0


def get_rearranging_indexes(add_pos, move_pos, n):
    """Creates an array 0...n-1 and moves value at 'move_pos' to 'add_pos', and shifts others back

    Useful to reorder data when we want to move instances from unlabeled set to labeled.
    TODO:
        Use this to optimize the API StreamingAnomalyDetector.get_query_data()
        since it needs to repeatedly convert the data to transformed [node] features.

    Example:
        get_rearranging_indexes(2, 2, 10):
            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        get_rearranging_indexes(0, 1, 10):
            array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])

        get_rearranging_indexes(2, 9, 10):
            array([0, 1, 9, 2, 3, 4, 5, 6, 7, 8])

    :param add_pos:
    :param move_pos:
    :param n:
    :return:
    """
    if add_pos > move_pos:
        raise ValueError("add_pos must be less or equal to move_pos")
    rearr_idxs = np.arange(n)
    if add_pos == move_pos:
        return rearr_idxs
    rearr_idxs[(add_pos + 1):(move_pos + 1)] = rearr_idxs[add_pos:move_pos]
    rearr_idxs[add_pos] = move_pos
    return rearr_idxs

