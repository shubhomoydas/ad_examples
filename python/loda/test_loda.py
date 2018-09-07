import numpy as np
import numpy.random as rnd
from aad.aad_globals import *
from common.gen_samples import *
from loda.loda import *


"""
pythonw -m loda.test_loda --log_file=temp/loda/test_loda.log --debug --dataset=mammography
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

