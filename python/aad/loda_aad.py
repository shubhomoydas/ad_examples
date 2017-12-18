import os
import numpy as np

import logging

from common.utils import *
from aad.aad_globals import *
from aad.aad_support import *

from loda.loda import loda
from loda_support import *
from classifier.perceptron import Perceptron
from common.data_plotter import *


class AadLoda(object):
    """ Wrapper over LODA

    Attributes:
         sparsity: float
         mink: int
         maxk: int
         loda_model: LodaResult
            The LODA model containing all projection vectors and histograms
    """
    def __init__(self, sparsity=np.nan, mink=1, maxk=0):
        self.sparsity = sparsity
        self.mink = mink
        self.maxk = maxk
        self.loda_model = None
        self.w = None
        self.m = 0

    def fit(self, x):
        self.loda_model = loda(x, self.sparsity, mink=self.mink, maxk=self.maxk)
        self.m = self.loda_model.pvh.pvh.w.shape[1]
        w = np.ones(self.m, dtype=float)
        self.w = normalize(w)

    def get_uniform_weights(self):
        if self.m == 0:
            raise ValueError("weights not initialized")
        w = np.ones(self.m, dtype=float)
        return normalize(w)

    def transform_to_loda_features(self, x, norm_unit=False):
        hpdfs = get_all_hist_pdfs(x, self.loda_model.pvh.pvh.w, self.loda_model.pvh.pvh.hists)
        nlls = -np.log(hpdfs)
        if norm_unit:
            norms = np.sqrt(np.power(nlls, 2).sum(axis=1))
            # logger.debug("norms before [%d/%d]:\n%s" % (start_batch, end_batch, str(list(norms.T))))
            nlls = (nlls.T * 1 / norms).T
            if False:
                norms = np.sqrt(np.power(nlls, 2).sum(axis=1))
                logger.debug("norms after:\n%s" % (str(list(norms))))
        return nlls

    def get_score(self, x, w=None):
        if w is None:
            w = self.w
        if w is None:
            raise ValueError("weights not initialized")
        score = x.dot(w)
        return score

    def get_auc(self, scores, labels):
        n = len(scores)
        tmp = np.empty(shape=(n, 2), dtype=float)
        tmp[:, 0] = labels
        tmp[:, 1] = -scores
        auc = fn_auc(tmp)
        return auc


def get_angles(x, w):
    n = x.shape[0]
    a = np.zeros(n, dtype=float)
    # logger.debug("x[0]:\n%s" % str(list(x[0].todense())))
    # logger.debug("x[0]:\n%s" % str(list(x[0].data)))
    for i in range(n):
        cos_theta = x[i].dot(w)
        a[i] = np.arccos(cos_theta)*180./np.pi
    return a


def plot_angle_hist(x, labels, model, opts):
    dir_create("./temp/angles")
    pdfpath = "./temp/angles/angles_%s_%s.pdf" % (opts.dataset, detector_types[opts.detector_type])
    unif_w = model.get_uniform_weights()
    vals = get_angles(x, unif_w)

    logger.debug(pdfpath)

    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)

    nom_v  = vals[np.where(labels==0)[0]]
    anom_v = vals[np.where(labels==1)[0]]
    bins = np.arange(start=np.min(vals), stop=np.max(vals), step=(np.max(vals)-np.min(vals))/50)
    pl = dp.get_next_plot()
    n1, bins1 = np.histogram(nom_v, bins=bins, normed=True)
    n2, bins2 = np.histogram(anom_v, bins=bins, normed=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, n1, align='center', width=width, facecolor='green', alpha=0.50)
    plt.bar(center, n2, align='center', width=width, facecolor='red', alpha=0.50)

    dp.close()


def compute_hyperplane(x, labels, model):
    unif_w = model.get_uniform_weights()
    perc = Perceptron(learning_rate=1.)
    perc.fit(x, labels, unif_w, epochs=200)
    cos_theta = unif_w.dot(perc.w)
    angle = np.arccos(cos_theta) * 180. / np.pi
    logger.debug("angle: %f (%f)" % (angle, cos_theta))


def loda_aad_batch():

    logger = logging.getLogger(__name__)

    args = get_aad_command_args(debug=False)
    # print "log file: %s" % args.log_file
    configure_logger(args)

    opts = AadOpts(args)
    # print opts.str_opts()
    logger.debug(opts.str_opts())

    if opts.detector_type != LODA:
        raise ValueError("Only detector type %d supported" % opts.detector_type)

    np.random.seed(opts.randseed)

    compute_angles = True
    compute_optimal_plane = True

    data = read_csv(opts.datafile, header=0, sep=',')
    X_train = np.zeros(shape=(data.shape[0], data.shape[1] - 1))
    for i in range(X_train.shape[1]):
        X_train[:, i] = data.iloc[:, i + 1]
    labels = np.array([1 if data.iloc[i, 0] == "anomaly" else 0 for i in range(data.shape[0])], dtype=int)

    logger.debug("loaded file: %s" % opts.datafile)
    logger.debug("results dir: %s" % opts.resultsdir)
    logger.debug("(mink, maxk): (%d, %d), sparsity: %f" % (opts.mink, opts.maxk, opts.sparsity))

    model = AadLoda(sparsity=opts.sparsity, mink=opts.mink, maxk=opts.maxk)
    model.fit(X_train)
    logger.debug("Projections shape: %s" % str(model.loda_model.pvh.pvh.w.shape))
    if True:
        X_train_new = model.transform_to_loda_features(X_train, norm_unit=False)
        scores = model.get_score(X_train_new)
        ordered_scores = -np.sort(-scores)  # sort descending
        logger.debug("scores:\n%s" % str(list(ordered_scores)))
    X_train_new = model.transform_to_loda_features(X_train, norm_unit=opts.norm_unit)
    logger.debug("X_train_new.shape: %s" % str(X_train_new.shape))

    scores = model.get_score(X_train_new)
    ordered_scores = -np.sort(-scores)  # sort descending
    auc = model.get_auc(scores, labels)
    logger.debug("AUC: %f" % auc)
    logger.debug("scores:\n%s" % str(list(ordered_scores)))

    if compute_angles:
        plot_angle_hist(X_train_new, labels, model, opts)

    if compute_optimal_plane:
        # perceptron expects {-1, +1} labels
        y = 2 * labels - 1
        compute_hyperplane(X_train_new, y, model)


if __name__ == "__main__":
    loda_aad_batch()
