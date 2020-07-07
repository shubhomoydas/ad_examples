import os

from ..common.utils import logger, ncol, save, load
from ..common.metrics import *
from ..loda.loda import loda, get_all_hist_pdfs, HistogramR, ProjectionVectorsHistograms, LodaResult, LodaModel


"""
Methods in this file are closely tied to LODA.
"""


class HistogramPDFs(object):
    def __init__(self, hpdfs, nlls):
        self.hpdfs = hpdfs
        self.nlls = nlls


def get_hpdfs_for_samples(allsamples, w, hists):
    samples_hpdfs = []
    for i in range(len(allsamples)):
        hpdfs = get_all_hist_pdfs(allsamples[i].fmat, w, hists)
        nlls = -np.log(hpdfs)
        samples_hpdfs.append(HistogramPDFs(hpdfs=hpdfs, nlls=nlls))
    return samples_hpdfs


def get_avg_auc_for_samples(allsamples, samples_hpdfs, proj_wts, ignore):
    aucs = []
    n = samples_hpdfs[0].nlls.shape[0]
    tmp = np.empty(shape=(n, 2), dtype=float)
    for i in range(len(samples_hpdfs)):
        if i != ignore:
            anom_score = samples_hpdfs[i].nlls.dot(proj_wts)
            tmp[:, 0] = allsamples[i].lbls
            tmp[:, 1] = -anom_score
            auc = fn_auc(tmp)
            aucs.append(auc)
    return np.mean(aucs)


def get_avg_precs_for_samples(k, allsamples, samples_hpdfs, proj_wts, ignore):
    n_k = len(k)
    m = 0.
    n = samples_hpdfs[0].nlls.shape[0]
    tmp = np.empty(shape=(n, 2), dtype=float)
    precs = np.zeros(n_k + 1, dtype=float)  # the last is the APR
    for i in range(len(samples_hpdfs)):
        if i != ignore:
            m += 1
            anom_score = samples_hpdfs[i].nlls.dot(proj_wts)
            tmp[:, 0] = allsamples[i].lbls
            tmp[:, 1] = -anom_score
            prec = fn_precision(tmp, k)
            precs = precs + prec[1:(n_k + 1)]
    return precs / m


class ModelManager(object):
    def can_save_model(self, opts):
        pass

    def is_model_saved(self, opts):
        pass

    def save_model(self, lodares, opts):
        pass

    def load_model(self, opts):
        pass

    def get_model(self, samples, opts):
        if self.is_model_saved(opts):
            algo_result = self.load_model(opts)
            logger.debug("Loaded saved model")
        else:
            algo_result = loda(samples, sparsity=opts.sparsity,
                               mink=max(int(ncol(samples)/2), opts.mink), maxk=opts.maxk,
                               keep=None, exclude=opts.exclude,
                               original_dims=opts.original_dims)
            if self.can_save_model(opts):
                logger.debug("Saving model")
                self.save_model(algo_result, opts)
        return algo_result

    @staticmethod
    def get_model_manager(type):
        if type == "csv":
            logger.debug("Using CsvModelManager...")
            return CsvModelManager()
        elif type == "pydata":
            logger.debug("Using PyDataModelManager...")
            return PyDataModelManager()
        else:
            raise ValueError("Invalid ModelManager type: %s" % (type,))


class PyDataModelManager(ModelManager):
    def can_save_model(self, opts):
        return opts.cachedir != "" and os.path.isdir(opts.cachedir)

    def is_model_saved(self, opts):
        prefix = opts.model_file_prefix()
        return os.path.isfile(os.path.join(opts.cachedir, "%s_lodares.pydata" % (prefix,)))

    def save_model(self, lodares, opts):
        prefix = opts.model_file_prefix()
        fname = "%s_lodares.pydata" % (prefix,)
        save(lodares, filepath=os.path.join(opts.cachedir, fname))

    def load_model(self, opts):
        prefix = opts.model_file_prefix()
        fname = "%s_lodares.pydata" % (prefix,)
        logger.debug("Loading model: %s" % (fname,))
        lodares = load(os.path.join(opts.cachedir, fname))
        return lodares


class CsvModelManager(ModelManager):
    def can_save_model(self, args, fid=0, runidx=0):
        if True: return False  # currently not allowing to save model as csv
        projs_path = os.path.join(args.cachedir, 'loda_projs')
        if not os.path.isdir(projs_path):
            return False
        return True

    def is_model_saved(self, opts):
        prefix = opts.model_file_prefix()
        projs_path = opts.cached_loda_projections_path()
        if not os.path.isdir(projs_path):
            return False
        wpath = os.path.join(projs_path, "%s-w.csv" % (prefix,))
        histpath = os.path.join(projs_path, "%s-hists.csv" % (prefix,))
        rnpath = os.path.join(projs_path, "%s-ranks_nll.csv" % (prefix,))
        if not os.path.isfile(wpath) or not os.path.isfile(histpath) or not os.path.isfile(rnpath):
            return False
        return True

    def save_model(self, lodares, opts):
        raise NotImplementedError("save_model_csv() not supported currently")

    def load_model(self, opts):
        """Loads models saved into csv files by R"""
        BREAKS_INDEX, COUNTS_INDEX, DENSITY_INDEX, MIDS_INDEX = (0, 1, 2, 3)
        prefix = opts.model_file_prefix()
        projs_path = opts.cached_loda_projections_path()
        if not os.path.isdir(projs_path):
            raise ValueError("%s does not exist. 'loda_projs' folder must exist in cache dir" % (projs_path,))
        wpath = os.path.join(projs_path, "%s-w.csv" % (prefix,))
        histpath = os.path.join(projs_path, "%s-hists.csv" % (prefix,))
        rnpath = os.path.join(projs_path, "%s-ranks_nll.csv" % (prefix,))
        if not os.path.isfile(wpath) or not os.path.isfile(histpath) or not os.path.isfile(rnpath):
            raise ValueError("One or more cached model files do not exist.")
        w = np.loadtxt(wpath, dtype=float, delimiter=',', skiprows=0)
        histdata = np.loadtxt(histpath, dtype=float, delimiter=',', skiprows=0)
        ranks_nll = np.loadtxt(rnpath, dtype=float, delimiter=',', skiprows=0)
        anomranks = np.asarray(ranks_nll[:, 0], dtype=int)
        nll = np.asarray(ranks_nll[:, 1], dtype=float)
        if 4 * ncol(w) != nrow(histdata):
            raise ValueError(
                "Inconsistent model: 4 * #cols(%d) in w != #rows(%d) in hists" % (ncol(w) * 4, nrow(histdata)))
        k = ncol(w)
        hists = []
        for i in range(k):
            # the order of rows for histograms are:
            #   breaks  <- 4 * i + BREAKS_INDEX
            #   counts  <- 4 * i + COUNTS_INDEX
            #   density <- 4 * i + DENSITY_INDEX
            #   mids    <- 4 * i + MIDS_INDEX
            nbreaks = np.int(histdata[4 * i + BREAKS_INDEX, 2])
            breaks = histdata[4 * i + BREAKS_INDEX, range(3, 3 + nbreaks)]
            density = histdata[4 * i + DENSITY_INDEX, range(3, 3 + nbreaks - 1)]
            counts = histdata[4 * i + COUNTS_INDEX, range(3, 3 + nbreaks - 1)]
            hist = HistogramR(counts=counts, density=density, breaks=breaks)
            hists.append(hist)

        pvh = ProjectionVectorsHistograms(w=w, hists=hists)
        return LodaResult(anomranks=anomranks, nll=nll, pvh=LodaModel(k=k, pvh=pvh))


