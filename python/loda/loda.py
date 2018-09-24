import numpy as np
from common.utils import *


"""
A non-streaming implementation of LODA. Please refer to:
    Tomas Pevny, Loda: Lightweight on-line detector of anomalies, Machine Learning 2015.
    http://webdav.agents.fel.cvut.cz/data/projects/stegodata/pdfs/Pev15-Loda.pdf
"""


class HistogramR(object):
    def __init__(self, counts, density, breaks):
        self.counts = counts
        self.density = density
        self.breaks = breaks


class ProjectionVectorsHistograms(object):
    def __init__(self, w=None, hists=None):
        """

        Args:
            w: numpy.ndarray(dtype=int)
                projection vectors
            hists: list of HistogramR
                histogram pdfs
        """
        self.w = w
        self.hists = hists


class LodaModel(object):
    def __init__(self, k=0, pvh=None, sigs=None):
        """
        Args:
            k: int
            pvh: ProjectionVectorsHistograms
            sigs: numpy.array(dtype=float)
        """
        self.k = k
        self.pvh = pvh
        self.sigs = sigs


class LodaResult(object):
    def __init__(self, anomranks=None, nll=None, pvh=None):
        """ Computed LODA model

        Args:
            anomranks: numpy.array(dtype=int)
            nll: numpy.array(dtype=float)
            pvh: LodaModel
        """
        self.anomranks = anomranks
        self.nll = nll
        self.pvh = pvh


def histogram_r(x, g1=1., g2=1., g3=-1., verbose=False):
    """Construct histograms that mimic behavior of R histogram package

    The type of histogram is 'regular', and right-open
    Note: the number of breaks is being computed as in:
    L. Birge, Y. Rozenholc, How many bins should be put in a regular histogram? 2006
    """
    # compute the number of bins based on the formula used by R package
    n = len(x)
    nbinsmax = int(g1 * (n ** g2) * (np.log(n) ** g3))
    if verbose:
        logger.debug("max bins: %d" % (nbinsmax,))
    # density, breaks = np.histogram(x, bins=nbinsmax, density=True)

    # the below implements Birge technique that recomputes the bin sizes...
    # y = np.sort(x)
    likelihood = np.zeros(nbinsmax, dtype=float)
    pen = np.arange(1, nbinsmax + 1, dtype=float) + ((np.log(np.arange(1, nbinsmax + 1, dtype=float))) ** 2.5)

    for d in range(1, nbinsmax + 1):
        #counts, breaks = np.histogram(x, bins=(y[0] + (np.arange(0, d+1, dtype=float)/d) * (y[n-1]-y[0])),
        #                              density=False)
        counts, breaks = np.histogram(x, bins=d, density=False)
        density = counts / (n * (breaks[1] - breaks[0]))
        like = np.zeros(d, dtype=float)
        like2 = np.zeros(d, dtype=float)
        tmp = np.where(counts > 0)[0]
        if len(tmp) > 0:
            like2[tmp] = np.log(density[tmp])
        finite_map = np.isfinite(like2)
        like[finite_map] = like2[finite_map]
        likelihood[d-1] = np.sum(counts * like)
        if np.min(counts) < 0:
            likelihood[d-1] = -np.Inf
    penlike = likelihood - pen
    optd = np.argmax(penlike)
    if verbose:
        logger.debug("optimal num bins: %d" % (int(optd)+1,))

    counts, breaks = np.histogram(x, bins=optd+1, density=False)
    density = counts / (n * (breaks[1] - breaks[0]))

    hist = HistogramR(counts=np.array(counts, float), density=np.array(density, float),
                      breaks=np.array(breaks, dtype=float))
    return hist


def histogram_r_mod(x, g1=1., g2=1., g3=-1., max_tries=500, verbose=False):
    """Construct histograms that mimic behavior of R histogram package

    The type of histogram is 'regular', and right-open
    Note: the number of breaks is being computed as in:
    L. Birge, Y. Rozenholc, How many bins should be put in a regular histogram? 2006
    """
    tm = Timer()
    # compute the number of bins based on the formula used by R package
    n = len(x)
    nbinsmax = int(g1 * (n ** g2) * (np.log(n) ** g3))
    if verbose:
        logger.debug("max bins: %d" % (nbinsmax,))
    # density, breaks = np.histogram(x, bins=nbinsmax, density=True)

    # the below implements Birge technique that recomputes the bin sizes...
    # y = np.sort(x)
    # likelihood = np.zeros(nbinsmax, dtype=float)
    # pen = np.arange(1, nbinsmax + 1, dtype=float) + ((np.log(np.arange(1, nbinsmax + 1, dtype=float))) ** 2.5)

    # instead of trying each bin size 1...nbinsmax, we will only try a few
    # as determined by the parameter max_tries.
    step_size = 1
    if nbinsmax > max_tries:
        # we will try maximum <max_tries> different values and take the best.
        step_size = int(nbinsmax * 1. / max_tries)
    if verbose:
        logger.debug("nbinsmax: %d, max_tries: %d, step_size: %d" % (nbinsmax, max_tries, step_size))
    d_list = list()
    likelihood = list()
    pen = list()
    for d in range(1, nbinsmax + 1, step_size):
        #counts, breaks = np.histogram(x, bins=(y[0] + (np.arange(0, d+1, dtype=float)/d) * (y[n-1]-y[0])),
        #                              density=False)
        counts, breaks = np.histogram(x, bins=d, density=False)
        density = counts / (n * (breaks[1] - breaks[0]))
        like = np.zeros(d, dtype=float)
        like2 = np.zeros(d, dtype=float)
        tmp = np.where(counts > 0)[0]
        if len(tmp) > 0:
            like2[tmp] = np.log(density[tmp])
        finite_map = np.isfinite(like2)
        like[finite_map] = like2[finite_map]
        # The below commented code is original. It is expensive.
        # likelihood[d-1] = np.sum(counts * like)
        # if np.min(counts) < 0:
        #     likelihood[d-1] = -np.Inf
        if np.min(counts) < 0:
            likelihood.append(-np.Inf)
        else:
            likelihood.append(np.sum(counts * like))
        pen.append(d + np.log(d)**2.5)
        d_list.append(d)
    likelihood = np.array(likelihood, dtype=np.float64)
    pen = np.array(pen, dtype=np.float64)
    penlike = likelihood - pen
    d_list = np.array(d_list, dtype=int)
    optd = d_list[np.argmax(penlike)]
    if verbose:
        logger.debug("optimal num bins: %d" % (int(optd)+1,))

    counts, breaks = np.histogram(x, bins=optd+1, density=False)
    density = counts / (n * (breaks[1] - breaks[0]))

    hist = HistogramR(counts=np.array(counts, float), density=np.array(density, float),
                      breaks=np.array(breaks, dtype=float))
    if verbose:
        logger.debug(tm.message("histogram_r():"))
    return hist


def get_bin_for_equal_hist(breaks, x):
    if x < breaks[0]:
        return 0
    if x > breaks[len(breaks)-1]:
        return len(breaks)-1
    i = np.trunc((x - breaks[0]) / (breaks[1] - breaks[0]))  # get integral value
    return int(i)


def pdf_hist_equal_bins(x, h, minpdf=1e-8):
    # here we are assuming a regular histogram where
    # h.breaks[1] - h.breaks[0] would return the width of the bin
    p = (x - h.breaks[0]) / (h.breaks[1] - h.breaks[0])
    ndensity = len(h.density)
    p = np.array([min(int(np.trunc(v)), ndensity-1) for v in p])
    d = h.density[p]
    # quick hack to make sure d is never 0
    d = np.array([max(v, minpdf) for v in d])
    return d


def pdf_hist(x, h, minpdf=1e-8):
    n = len(x)
    pd = np.zeros(n)
    for j in range(n):
        # use simple index lookup in case the histograms are equal width
        # this returns the lower index
        i = get_bin_for_equal_hist(h.breaks, x[j])
        if i >= len(h.density):
            i = len(h.density)-1  # maybe something else should be done here
        # i might be 0 if the value does not lie within the
        # histogram range. We will assume for now that it cannot happen.

        # More accurately, we should also multiply by diff(h$breaks)[i];
        # however, all breaks are equal in length in this algorithm,
        # hence, ignoring that for now.
        # also, hack to make sure that density is not zero
        pd[j] = max(h.density[i], minpdf)
    return pd


# Get the random projections
def get_random_proj(nproj, d, sp, keep=None, exclude=None):
    nzeros = int(np.floor(d * sp))
    idxs = np.arange(d)  # set of dims that will be sampled to be set to zero
    marked = []
    if keep is not None:
        marked.extend(keep)
    if exclude is not None:
        # since 'exclude' contains the dims that are
        # predetermined to be zero, adjust the number
        # of zero dims that need to be further determined
        # by sampling
        nzeros -= len(exclude)
        marked.extend(exclude)
    if len(marked) > 0:
        # remove from the known set -- the dims that have been
        # marked for keeping or excluding. There is no uncertainty in
        # the selection/rejection of marked dims.
        idxs = np.delete(idxs, marked)
    w = np.zeros(shape=(d, nproj), dtype=float)
    for i in range(nproj):
        w[:, i] = np.random.randn(d)
        if nzeros > 0:
            z = sample(idxs, min(nzeros, len(idxs)))
            if exclude is not None:
                z.extend(exclude)
            w[z, i] = 0
        w[:, i] = w[:, i] / np.sqrt(sum(w[:, i] *w[:, i]))

    return w


# Build histogram for each projection
def build_proj_hist(a, w, verbose=False):
    d = ncol(w)  # number of columns
    x = a.dot(w)
    hists = []
    for j in range(d):
        hists_j = histogram_r_mod(x[:, j], verbose=verbose)
        hists.append(hists_j)
    return hists


# a - (n x d) matrix
# w - (n x 1) vector
def get_neg_ll(a, w, hist, inf_replace=np.nan):
    x = a.dot(w)
    pdfs = np.zeros(shape=(len(x), 1), dtype=float)
    pdfs[:, 0] = pdf_hist_equal_bins(x, hist)
    pdfs[:, 0] = np.log(pdfs)[:, 0]
    if inf_replace is not np.nan:
        pdfs[:, 0] = [max(v, inf_replace) for v in pdfs[:, 0]]
    return -pdfs  # neg. log-lik of pdf


# get all pdfs from individual histograms.
def get_all_hist_pdfs(a, w, hists):
    x = a.dot(w)
    hpdfs = np.zeros(shape=(len(x), len(hists)), dtype=float)
    for i in range(len(hists)):
        hpdfs[:, i] = pdf_hist(x[:, i], hists[i])
    return hpdfs


# Compute negative log-likelihood using random projections and histograms
def get_neg_ll_all_hist(a, w, hists, inf_replace=np.nan):
    pds = get_all_hist_pdfs(a, w, hists)
    pds = np.log(pds)
    if inf_replace is not np.nan:
        vfunc = np.vectorize(max)
        pds = vfunc(pds, 1.0 * inf_replace) # [max(v, inf_replace) for v in pds[:, i]]
    ll = -np.mean(pds, axis=1)  # neg. log-lik
    return ll


# Determine k - no. of dimensions
# sp=1 - 1 / np.sqrt(ncol(a)),
def get_best_proj(a, mink=1, maxk=10, sp=0.0, keep=None, exclude=None, verbose=False):
    """ Computes random projections and histogram pdfs along each projection

    Args:
        a: numpy.ndarray
        mink: int
            minimum number of projections
        maxk: int
            maximum number of projections
        sp: float
            sparsity, i.e., number of dimensions which should be zero.
        keep: list
            columns of original feature space which must be included
        exclude:
            columns of original feature space which must be excluded
        verbose:
            whether to output detail execution logs when building projections

    """
    t = 0.01
    n = nrow(a)
    d = ncol(a)

    w = np.zeros(shape=(d, maxk + 1), dtype=float)
    hists = []
    fx_k = np.zeros(shape=(n, 1), dtype=float)
    fx_k1 = np.zeros(shape=(n, 1), dtype=float)

    w_ = get_random_proj(nproj=1, d=d, sp=sp, keep=keep, exclude=exclude)
    w[:, 0] = w_[:, 0]
    hists.append(build_proj_hist(a, w_, verbose=verbose)[0])
    fx_k[:, 0] = get_neg_ll(a, w_, hists[0])[:, 0]

    sigs = np.ones(maxk) * np.Inf
    k = 0
    # logger.debug("mink: %d, maxk: %d" % (mink, maxk))
    while k < maxk:
        w_ = get_random_proj(nproj=1, d=d, sp=sp, keep=keep, exclude=exclude)
        w[:, k+1] = w_[:, 0]
        hists.append(build_proj_hist(a, w_, verbose=verbose)[0])

        ll = get_neg_ll(a, w[:, k+1], hists[k+1])

        fx_k1[:, 0] = fx_k[:, 0] + ll[:, 0]

        diff_ll = abs(fx_k1 / (k+2.0) - fx_k / (k+1.0))
        # logger.debug(diff_ll)
        diff_ll = diff_ll[np.isfinite(diff_ll)]
        if len(diff_ll) > 0:
            sigs[k] = np.mean(diff_ll)
        else:
            raise(ValueError("Log-likelihood was invalid for all instances"))
        tt = sigs[k] / sigs[0]
        if tt < t and k >= mink:
            break

        fx_k[:, 0] = fx_k1[:, 0]

        # if (debug) print(paste("k =",k,"; length(sigs)",length(sigs),"; sigs_k=",tt))

        k += 1

    bestk = np.where(sigs == np.min(sigs))[0][0]  # np.where returns tuple of arrays
    if bestk < mink:
        # we retain at least the minimum
        bestk = mink
    # logger.debug("get_best_proj: bestk: %d, sigs:\n%s" % (bestk, str(list(sigs))))
    return LodaModel(bestk, ProjectionVectorsHistograms(matrix(w[:, 0:(bestk+1)], nrow=nrow(w)),
                                                        hists[0:(bestk+1)]),
                     sigs)


def get_original_proj(a, maxk=10, sp=0, keep=None, exclude=None):
    d = ncol(a)
    w = np.zeros(shape=(d, d - (0 if exclude is None else len(exclude))), dtype=float)
    hists = []
    k = 0
    for l in range(d):
        if exclude is not None and len(np.where(exclude == l)[0]) > 0:
            continue
        w_ = np.zeros(shape=(d, 1), dtype=float)
        w_[l, 0] = 1  # important: the 'l'-th (not 'k'-th) dim is 1
        w[:, k] = w_[:, 0]
        hists.append(build_proj_hist(a, w_)[0])
        k += 1

    return LodaModel(k=k, pvh=ProjectionVectorsHistograms(w=w, hists=hists), sigs=None)


def loda(a, sparsity=np.nan, mink=1, maxk=0, keep=None, exclude=None, original_dims=False, verbose=False):
    l = nrow(a)
    d = ncol(a)

    maxk = 3 * d if maxk == 0 else maxk
    maxk = max(mink, maxk)

    if sparsity is None or sparsity is np.nan:
        sp = 0 if d == 1 else 1 - 1 / np.sqrt(d)
    else:
        sp = sparsity

    tm = Timer()

    if original_dims:
        pvh = get_original_proj(a, maxk=maxk, sp=sp, keep=keep, exclude=exclude)
    else:
        pvh = get_best_proj(a, mink=mink, maxk=maxk, sp=sp, keep=keep, exclude=exclude, verbose=verbose)

    logger.debug(tm.message("loda: sparsity: %f:" % (sp,)))

    nll = get_neg_ll_all_hist(a, pvh.pvh.w, pvh.pvh.hists, inf_replace=np.nan)

    anomranks = np.arange(l)
    anomranks = anomranks[order(-nll)]

    return LodaResult(anomranks=anomranks, nll=nll, pvh=pvh)


class Loda(object):
    def __init__(self, sparsity=np.nan, mink=1, maxk=0, random_state=None, verbose=False):
        self.sparsity = sparsity
        self.mink = mink
        self.maxk = maxk
        self.random_state = random_state
        self.loda_model = None
        self.m = None  # number of projections
        self.verbose = verbose

    def get_projections(self):
        if self.loda_model is None:
            raise RuntimeError("Loda model not initialized")
        return self.loda_model.pvh.pvh.w

    def fit(self, x):
        self.loda_model = loda(x, self.sparsity, mink=self.mink, maxk=self.maxk, verbose=self.verbose)
        self.m = self.loda_model.pvh.pvh.w.shape[1]

    def get_projection_scores(self, x):
        """ Smaller values mean more anomalous """
        if self.loda_model is None:
            raise RuntimeError("Loda model not initialized")
        hpdfs = get_all_hist_pdfs(x, self.loda_model.pvh.pvh.w, self.loda_model.pvh.pvh.hists)
        nlls = np.log(hpdfs)
        return nlls

    def decision_function(self, x):
        """ Smaller values mean more anomalous """
        if self.loda_model is None:
            raise RuntimeError("Loda model not initialized")
        # hpdfs = get_all_hist_pdfs(x, self.loda_model.pvh.pvh.w, self.loda_model.pvh.pvh.hists)
        # nlls = np.log(hpdfs)
        nlls = self.get_projection_scores(x)
        scores = np.sum(nlls, axis=1)
        return scores


def get_zero_var_features(x):
    d = ncol(x)
    zcols = []
    for i in range(d):
        if np.var(x[:, i]) == 0.0:
            zcols.append(i)
    return zcols
