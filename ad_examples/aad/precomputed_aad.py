import os
import numpy as np

import logging

from ..common.utils import read_csv, normalize, power, configure_logger, read_data_as_matrix

from .aad_globals import PRECOMPUTED_SCORES, get_aad_command_args, AadOpts, detector_types
from .aad_base import Aad, Ensemble


class AadPrecomputed(Aad):
    """ AAD for precomputed ensemble scores

    Attributes:
    """
    def __init__(self, opts=None, random_state=None):
        Aad.__init__(self, PRECOMPUTED_SCORES, random_state=random_state)
        if opts is None:
            raise ValueError("Invalid options passed")
        if opts.scoresfile is None or not os.path.exists(opts.scoresfile):
            raise ValueError("Invalid scores file: %s" %
                             ("<missing>" if opts.scoresfile is None else opts.scoresfile))
        self.opts = opts
        self.m = None
        self.scores = None

    def get_num_members(self):
        """Returns the number of ensemble members"""
        return self.m

    def fit(self, x):
        """Load the score file and check if dimensions match"""
        data = read_csv(self.opts.scoresfile, header=0, sep=',')
        if data.shape[0] != x.shape[0]:
            raise ValueError("Data file and scores file are incompatible")

        self.scores = np.zeros(shape=(data.shape[0], data.shape[1] - 1))
        for i in range(self.scores.shape[1]):
            self.scores[:, i] = data.iloc[:, i + 1]

        self.m = self.scores.shape[1]
        w = np.ones(self.m, dtype=float)
        self.w = normalize(w)

    def transform_to_ensemble_features(self, x, dense=False, norm_unit=False):
        tmp = self.scores
        if norm_unit:
            norms = power(tmp, 2)
            tmp = tmp.multiply(1 / norms)
        return tmp


def test_precomputed_scores():
    logger = logging.getLogger(__name__)

    args = get_aad_command_args(debug=False)
    configure_logger(args)

    opts = AadOpts(args)
    logger.debug(opts.str_opts())

    X_train, labels = read_data_as_matrix(opts)

    logger.debug("loaded file: %s" % opts.datafile)
    logger.debug("results dir: %s" % opts.resultsdir)
    logger.debug("detector_type: %s" % detector_types[opts.detector_type])

    random_state = np.random.RandomState(opts.randseed)

    # fit the model
    model = AadPrecomputed(opts, random_state=random_state)
    model.fit(X_train)
    X_train_new = model.transform_to_ensemble_features(X_train, norm_unit=opts.norm_unit)

    baseline_w = model.get_uniform_weights()

    agg_scores = model.get_score(X_train_new, baseline_w)

    ensemble = Ensemble(X_train, labels, X_train_new, baseline_w,
                        agg_scores=agg_scores, original_indexes=np.arange(X_train.shape[0]),
                        auc=0.0, model=None)

    model.init_weights(init_type=opts.init)

    metrics = model.aad_learn_ensemble_weights_with_budget(ensemble, opts)

    nqueried = len(metrics.queried)
    num_seen = np.cumsum(ensemble.labels[metrics.queried])
    qlbls = ensemble.labels[ensemble.ordered_anom_idxs[0:nqueried]]
    num_seen_baseline = np.cumsum(qlbls)

    logger.debug("baseline (w/o  feedback): \n%s" % str(list(num_seen_baseline)))
    logger.debug("num_seen (with feedback): \n%s" % str(list(num_seen)))


if __name__ == "__main__":
    test_precomputed_scores()
