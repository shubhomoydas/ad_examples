import numpy as np

from ..common.utils import logger, normalize
from .aad_globals import LODA
from .aad_base import Aad

from ..loda.loda import loda, get_all_hist_pdfs
from .data_stream import StreamingSupport


class AadLoda(Aad, StreamingSupport):
    """ Wrapper over LODA

    Attributes:
         sparsity: float
         mink: int
         maxk: int
         loda_model: LodaResult
            The LODA model containing all projection vectors and histograms
    """
    def __init__(self, sparsity=np.nan, mink=1, maxk=0, random_state=None):
        Aad.__init__(self, LODA, random_state=random_state)
        self.sparsity = sparsity
        self.mink = mink
        self.maxk = maxk
        self.loda_model = None
        self.m = None

    def get_num_members(self):
        """Returns the number of ensemble members"""
        return self.m

    def fit(self, x):
        self.loda_model = loda(x, self.sparsity, mink=self.mink, maxk=self.maxk)
        self.m = self.loda_model.pvh.pvh.w.shape[1]
        w = np.ones(self.m, dtype=float)
        self.w = normalize(w)
        logger.debug("LODA m: %d" % self.m)

    def transform_to_ensemble_features(self, x, dense=False, norm_unit=False):
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

    def supports_streaming(self):
        return False

    def add_samples(self, X, current=False):
        logger.warning("Model does not support stream update. Retaining old model.")

    def update_model_from_stream_buffer(self):
        # LODA implementation currently does not support this
        pass
