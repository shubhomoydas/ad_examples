import numpy as np
import scipy as sp

from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix, vstack

from common.utils import *


class DataStream(object):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def read_next_from_stream(self, n=1):
        n = min(n, self.X.shape[0])
        # logger.debug("DataStream.read_next_from_stream n: %d" % n)
        if n == 0:
            return None, None
        mask = np.zeros(self.X.shape[0], dtype=bool)
        mask[np.arange(n)] = True
        instances = self.X[mask]
        self.X = self.X[~mask]
        labels = None
        if self.y is not None:
            labels = self.y[mask]
            self.y = self.y[~mask]
        # logger.debug("DataStream.read_next_from_stream instances: %s" % str(instances.shape))
        return instances, labels

    def empty(self):
        return self.X is None or self.X.shape[0] == 0

