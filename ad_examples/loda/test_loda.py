import logging
import random
import numpy.random as rnd

from ..common.utils import Timer, dir_create, configure_logger
from ..common.gen_samples import read_anomaly_dataset
from ..aad.aad_globals import get_aad_command_args
from .loda import Loda


"""
pythonw -m ad_examples.loda.test_loda --log_file=temp/loda/test_loda.log --debug --dataset=mammography
"""


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/loda")  # for logging and plots

    args = get_aad_command_args(debug=False,
                                debug_args=["--dataset=mammography",
                                            "--debug",
                                            "--log_file=temp/loda/test_loda.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    random.seed(args.randseed)
    rnd.seed(args.randseed)

    x, y = read_anomaly_dataset(args.dataset)

    mink = 100
    maxk = 200
    logger.debug("dataset: %s, mink: %d, maxk: %d, randseed: %d" % (args.dataset, mink, maxk, args.randseed))
    mdl = Loda(mink=mink, maxk=maxk, random_state=args.randseed, verbose=False)
    tm = Timer()
    mdl.fit(x)
    logger.debug(tm.message("Time in fit():"))
    logger.debug("Projections: %d, %s" % (mdl.m, str((mdl.get_projections()).shape)))

