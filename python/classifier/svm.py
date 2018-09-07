import numpy.random as rnd

from common.utils import *
from common.gen_samples import *
from common.sgd_optimization import sgdRMSProp

"""
Simple SVM implementations in primal form, solved with only gradient descent
and no other linear optimization libraries.
"""


class Classifier(object):
    def __init__(self, C=1.):
        self.C = C
        self.w_ = None
        self.b_ = None
        self.cls2index_ = None
        self.index2cls_ = None
        self.w_names = None

    def set_classes(self, y):
        c_ = np.unique(y)
        self.cls2index_ = dict()
        self.index2cls_ = dict()
        for i, v in enumerate(c_):
            self.cls2index_[v] = i
            self.index2cls_[i] = v

    def fit(self, x, y):
        pass

    def decision_function(self, x):
        return x.dot(self.w_) + self.b_

    def predict(self, x):
        pass


class BinaryLinearSVMClassifier(Classifier):
    """A very simple binary SVM classifier trained using gradient descent

    Note: The optimization solves the objective in primal
    """

    def __init__(self, C=1.):
        Classifier.__init__(self, C)

    def f(self, Wb, x, y):
        """Loss function

        hinge_loss(x) = max(0, x)

        loss = 0.5 * w.w / C
               + hinge_loss(1 - 1[y == 1] * (x.w + b))
               + hinge_loss(1 + 1[y == 0] * (x.w + b))
        """
        w = Wb[0:(len(Wb)-1)]
        b = Wb[len(Wb)-1]
        loss_r = 0.5 * w.dot(w) / self.C
        y_ = np.array([1. if yy == 1 else -1. for yy in y])
        e_ = 1. - y_ * (x.dot(w) + b)
        loss_h = np.mean(np.maximum(e_, 0.))
        return loss_r + loss_h

    def g(self, Wb, x, y):
        """Gradient of loss function

        d_loss = w / C
                 + 1[1 - 1[y == 1] * (x.w + b) > 0] * (-1) * 1[y == 1] * x
                 + 1[1 + 1[y == 0] * (x.w + b) > 0] * ( 1) * 1[y == 0] * x
        """
        w = Wb[0:(len(Wb)-1)]
        b = Wb[len(Wb)-1]
        y_ = np.array([1 if yy == 1 else -1 for yy in y])
        e_ = 1. - y_ * (x.dot(w) + b)
        egz = e_ > 0
        a1 = np.transpose([np.logical_and(egz, y_ ==  1.).astype(int)])
        a2 = np.transpose([np.logical_and(egz, y_ == -1.).astype(int)])
        dlossW = np.mean(np.multiply(a1, -x) + np.multiply(a2,  x), axis=0)
        dlossb = np.mean(np.multiply(a1, -1) + np.multiply(a2,  1.))
        return np.append(w / self.C, [0.]) + np.append(dlossW, dlossb)

    def fit(self, x, y):

        def wf(w, x, y):
            return self.f(w, x, y)

        def wg(w, x, y):
            return self.g(w, x, y)

        self.w_ = None
        self.b_ = None

        self.set_classes(y)

        y_ = np.array([self.cls2index_[v] for v in y])

        w0 = np.zeros(x.shape[1], dtype=float)
        b0 = 0.
        Wb0 = np.append(w0, [b0])
        Wb = sgdRMSProp(Wb0, x, y_, wf, wg, learning_rate=0.001, max_epochs=15000)

        self.w_ = Wb[0:(len(Wb) - 1)]
        self.b_ = Wb[len(Wb) - 1]

        return self.w_, self.b_

    def predict(self, x):
        des = self.decision_function(x)
        # logger.debug("decision_function:\n%s" % str(des))
        pred_y = np.sign(des)
        y = np.array([self.index2cls_[0] if k == -1 else self.index2cls_[1] for k in pred_y])
        return y


class MultiClassLinearSVMClassifier(Classifier):
    """A very simple multi-class SVM classifier trained using gradient descent.
    We solve the classification problem simultaneously for all classes instead
    of treating the task as multiple independent binary classification tasks.

    Define m as the predicted class for x:
        m = argmax_{k=1..M} x.w_k + b_k

    The objective is:
      minimize_w 0.5 * w.w + C * hinge_loss(2 + x.w_m + b_m - x.w_y - b_y)

    Note: The optimization solves the objective in primal.
    """

    def __init__(self, C=1., penalty_type='L1', penalize_bias=False):
        Classifier.__init__(self, C)

        if not (penalty_type == 'L1' or penalty_type == 'L2'):
            raise ValueError("Invalid penalty type.")
        self.penalty_type = penalty_type  # 'L1'/'L2'

        self.penalize_bias = penalize_bias

    def f(self, Wb, x, y):
        """Loss function

        hinge_loss(x) = max(0, x)

        Define m as the predicted class for x:
            m = argmax_{k=1..M} x.w_k + b_k

        loss = 0.5 * w.w / C
               + hinge_loss(2 + x.w_m + b_m - x.w_y + b_y)^p
            where p = 1 or 2 depending on penalty_type

        Args:
            Wb: np.array(size=(M x (d+1),))
                Where M is the number of classes, d is the data dimension,
                includes one bias for each class. The weight vectors for all
                classes are concatenated one after the other to form a 1d array.
        """
        n = x.shape[0]
        d = x.shape[1]
        M = len(Wb) / (d+1) # The d plus one for bias term
        Wb_ = np.reshape(Wb, (M, d+1), order='C')
        # logger.debug("Wb_:\n%s" % str(Wb_))
        wT = Wb_[:, 0:d]
        b = Wb_[:, d]
        w = np.transpose(wT)  # weights are column vectors

        pv = x.dot(w) + b
        py = np.argmax(pv, axis=1)
        # logger.debug("pv:\n%s" % str(pv))
        # logger.debug("f() py:\n%s" % str(py))

        loss_h = 0.0
        errors = 0.
        for i, yy in enumerate(py):
            if y[i] != yy:
                # true y does not match predicted y
                loss_x = np.maximum(0, 2. + x[i].dot(wT[yy] - wT[y[i]]) + b[yy] - b[y[i]])
                if self.penalty_type == 'L2':
                    loss_x = loss_x ** 2
                loss_h += loss_x
                errors += 1
        if self.penalty_type == 'L2':
            loss_h *= 0.5
        loss_h = (1. / n) * loss_h
        loss_r = 0.5 * np.trace(np.transpose(w).dot(w)) / self.C
        if self.penalize_bias:
            loss_r += 0.5 * b.dot(b) / self.C
        # logger.debug("loss_h: %s" % str(loss_h))
        # logger.debug("loss_r: %s" % str(loss_r))
        # logger.debug("loss: %s" % str(loss_r + loss_h))
        # logger.debug("f(): errors: %f" % errors)
        return loss_r + loss_h

    def g(self, Wb, x, y):
        """Gradient of loss function

        Define m as the predicted class for x:
            m = argmax_{k=1..M} x.w_k + b_k

        and define margin loss v as:
            v = max(0, 2 + x.w_m + b_m - x.w_y + b_y)^p
        where p = 1 or 2 depending on penalty_type

        for p = 1, gradient of loss is:
        d_loss_k = w_k / C
                 + 1[k == m] * 1[m != y] * 1[v > 0] * ( 1) * x
                 + 1[k == y] * 1[m != y] * 1[v > 0] * (-1) * x

        Args:
            Wb: np.array(size=(M x (d+1),))
                Where M is the number of classes, d is the data dimension,
                includes one bias for each class. The weight vectors for all
                classes are concatenated one after the other to form a 1d array.
        """
        n = x.shape[0]
        d = x.shape[1]
        M = len(Wb) / (d+1) # The d plus one for bias term
        Wb_ = np.reshape(Wb, (M, d+1), order='C')
        wT = Wb_[:, 0:d]
        b = Wb_[:, d]
        w = np.transpose(wT)  # weights are column vectors

        pv = x.dot(w) + b
        py = np.argmax(pv, axis=1)  # predicted y
        # logger.debug("pv:\n%s" % str(pv))
        # logger.debug("g() py:\n%s" % str(py))

        # get all incorrect predictions
        dlossW = np.zeros(shape=w.shape, dtype=w.dtype)
        dlossb = np.zeros(len(b), dtype=b.dtype)
        for i, m in enumerate(py):
            yi = y[i]
            if yi != m:  # true != predicted
                vx = x[i].dot(wT[m] - wT[yi])
                v = np.maximum(0, 2. + vx + b[m] - b[yi])
                if v > 0:
                    if self.penalty_type == 'L2':
                        dlossW[:,  m] += v * x[i]
                        dlossW[:, yi] -= v * x[i]
                        dlossb[ m] += v
                        dlossb[yi] -= v
                    else:
                        dlossW[:,  m] += x[i]
                        dlossW[:, yi] -= x[i]
                        dlossb[ m] += 1
                        dlossb[yi] -= 1
        dlossW = w / self.C + (1. / n) * dlossW
        dlossb = (1. / n) * dlossb
        if self.penalize_bias:
            dlossb += b / self.C
        dlossWb = np.transpose(np.vstack([dlossW, dlossb]))
        # logger.debug("dlossWb:\n%s" % str(dlossWb))

        return np.ravel(dlossWb, order='C')

    def fit(self, x, y):

        def wf(w, x, y):
            return self.f(w, x, y)

        def wg(w, x, y):
            return self.g(w, x, y)

        self.w_ = None
        self.b_ = None

        self.set_classes(y)

        y_ = np.array([self.cls2index_[v] for v in y])

        d = x.shape[1]
        M = len(self.cls2index_)
        Wb0 = np.reshape(rnd.uniform(-1., 1., (d+1)*M), (d+1, M))
        Wb0[d, :] = 0.  # bias is initialized to zero
        Wb0_ = np.ravel(np.transpose(Wb0), order='C')
        Wb_ = sgdRMSProp(Wb0_, x, y_, wf, wg, learning_rate=0.001, max_epochs=15000)
        Wb = np.transpose(np.reshape(Wb_, (M, d+1), order='C'))

        self.w_ = Wb[0:d]
        self.b_ = Wb[d]

        self.w_names = []
        for i in self.index2cls_.keys():
            self.w_names.append("class %s" % str(self.index2cls_[i]))

        # loss = self.f(Wb_, x, y_)
        # logger.debug("Final loss: %f" % loss)

        return self.w_, self.b_

    def predict(self, x):
        des = self.decision_function(x)
        # logger.debug("decision_function:\n%s" % str(des))
        pred_y = np.argmax(des, axis=1)
        y = np.array([self.index2cls_[k] for k in pred_y])
        return y


class PairwiseLinearSVMClassifier(Classifier):
    def __init__(self, C):
        Classifier.__init__(self, C)
        self.svms = None

    def fit(self, x, y):
        self.set_classes(y)
        d = x.shape[1]
        M = len(self.index2cls_)
        pairs = int((M*(M-1))/2)
        # logger.debug("M: %d" % M)

        self.svms = []
        self.w_ = np.zeros(shape=(d, pairs), dtype=float)
        self.b_ = np.zeros(pairs, dtype=float)
        self.w_names = []
        cls = list(self.cls2index_.keys())
        pi = 0
        for i in range(len(cls)-1):
            k1 = cls[i]
            for j in range(i+1, len(cls)):
                k2 = cls[j]
                self.w_names.append("%s vs %s" % (str(k1), str(k2)))
                idxs = np.array([l for l, yy in enumerate(y) if yy == k1 or yy == k2])
                # logger.debug("pairwise between %d and %d (%d)" % (k1, k2, len(idxs)))
                svm = BinaryLinearSVMClassifier(C=self.C)
                x_ = x[idxs]
                y_ = y[idxs]
                w, b = svm.fit(x_, y_)
                pred_y = svm.predict(x_)
                errors = np.sum([1. if p[0] != p[1] else 0. for p in zip(pred_y, y_)])
                # logger.debug("errors:%f" % errors)
                self.svms.append(svm)
                self.w_[:, pi] = w
                self.b_[pi] = b
                pi += 1

        # logger.debug("w_:\n%s" % str(self.w_))
        # logger.debug("b_:\n%s" % str(self.b_))
        return self.w_, self.b_

    def predict(self, x):
        n = x.shape[0]
        des = self.decision_function(x)
        # logger.debug("decision_function:\n%s" % str(des))
        des_p = np.maximum(0, np.sign(des)).astype(int)
        tmp_cls = np.zeros(shape=(n, len(self.cls2index_)), dtype=int)
        for i in range(n):
            for j, svm in enumerate(self.svms):
                cls = svm.index2cls_[des_p[i, j]]  # get predicted class for svm
                idx = self.cls2index_[cls]  # lookup class index in this classifier
                tmp_cls[i, idx] += 1
        # logger.debug("tmp_cls:\n%s" % str(tmp_cls))
        pred_y = np.argmax(tmp_cls, axis=1)
        y = np.array([self.index2cls_[k] for k in pred_y])
        return y
