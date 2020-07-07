import logging
import numpy as np

from ..common.utils import normalize


logger = logging.getLogger(__name__)


class Perceptron(object):
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.w = None

    def fit(self, x, y, w0=None, epochs=1):
        n = x.shape[0]
        d = x.shape[1]

        if w0 is not None:
            w = np.copy(w0)
        elif self.w is None:
            w = np.zeros(d, dtype=float)
        else:
            w = self.w

        # cos_theta = w0.dot(w0)
        # logger.debug("initial angle: %f (%f)" % (np.arccos(cos_theta) * 180. / np.pi, cos_theta))

        last_epoch = epochs
        for epoch in range(epochs):
            errors1 = 0
            errors2 = 0
            for i in range(n):
                v = x[i].dot(w)
                if y[i] * v < 0:
                    w -= 2 * self.learning_rate * v * x[i]
                    if y[i] == 1:
                        errors1 += 1
                    else:
                        errors2 += 1
                    cos_theta = w.dot(w0)
                    # logger.debug("epoch %d[%d] angle: %f" % (epoch, i, np.arccos(cos_theta)*180./np.pi))
            errors = errors1 + errors2
            if errors == 0:
                last_epoch = epoch
                break
            logger.debug("epoch: %d, errors: %d (+1=%d / -1=%d)" % (epoch, errors, errors1, errors2))
        logger.debug("last_epoch: %d" % last_epoch)
        self.w = normalize(w)
        return self.w
