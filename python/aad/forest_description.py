import os
import numpy as np

import logging

from aad.aad_globals import *
from aad.aad_support import *


def get_most_anomalous_subspace_indexes(model, n_top=30):
    wd = np.multiply(model.w, model.d)
    ordered_wd_idxs = np.argsort(-wd)[0:n_top]  # sort in reverse order
    # logger.debug("ordered_wd:\n%s" % str(wd[ordered_wd_idxs]))
    return ordered_wd_idxs


def get_region_volumes(model, region_indexes, feature_ranges):
    volumes = np.zeros(len(region_indexes), dtype=np.float32)
    d = feature_ranges.shape[0]  # number of features
    for i, ridx in enumerate(region_indexes):
        region = model.all_regions[ridx].region
        # logger.debug(str(region))
        region_ranges = np.zeros(d, dtype=np.float32)
        for j in range(feature_ranges.shape[0]):
            rmin = feature_ranges[j][0] if np.isinf(region[j][0]) else region[j][0]
            rmax = feature_ranges[j][1] if np.isinf(region[j][1]) else region[j][1]
            # logger.debug("%d: %f, %f" % (j, rmin, rmax))
            region_ranges[j] = rmax - rmin
            volumes[i] = np.prod(region_ranges)
    # logger.debug("volumes:\n%s" % str(volumes))
    return volumes


def get_region_memberships(x, y, model, metrics, region_indexes):
    if metrics is not None and metrics.queried is not None:
        queried = np.array(metrics.queried)
        inst_indexes = np.where(y[queried] == 1)[0]
        anom_indexes = queried[inst_indexes]
    else:
        anom_indexes = np.where(y == 1)[0]
    nregions = len(region_indexes)
    member_anoms = list()
    member_inds = list()
    for i in anom_indexes:
        inds = np.zeros(nregions, dtype=int)
        for j, ridx in enumerate(region_indexes):
            inds[j] = is_in_region(x[i, :], model.all_regions[ridx].region)
        if np.sum(inds) > 0:
            member_anoms.append(i)
            member_inds.append(np.reshape(inds, newshape=(1, nregions)))
    member_anoms = np.array(member_anoms, dtype=int)
    member_inds = np.vstack(member_inds)
    return member_anoms, member_inds


def get_compact_regions(x, y, model, metrics, region_indexes, volumes, p=1):
    import cvxopt
    from cvxopt import glpk

    if metrics is None or metrics.queried is None or len(metrics.queried) == 0:
        logger.debug("No labeled anomalies found for description")
        return None

    member_anoms, member_inds = get_region_memberships(x, y, model, metrics, region_indexes)
    # logger.debug("anom indexes in selected regions (%d):\n%s" % (len(member_anoms), str(list(member_anoms))))
    # logger.debug("member_inds (%s):\n%s" % (str(member_inds.shape), str(member_inds)))

    nvars = member_inds.shape[1]
    glpk.options['msg_lev'] = 'GLP_MSG_OFF'

    c = cvxopt.matrix([float(v**p) for v in volumes], tc='d')  # minimize total volume**p

    # below states that each anomaly should be included in atleast one region
    G = cvxopt.matrix(-member_inds, tc='d')
    h = cvxopt.matrix([-1] * member_inds.shape[0], tc='d')

    bin_vars = [i for i in range(nvars)]
    (status, x) = cvxopt.glpk.ilp(c, G, h, B=set(bin_vars))
    logger.debug("ILP status: %s" % status)
    if x is not None:
        x = np.reshape(np.array(x), newshape=(nvars,))
        logger.debug("ILP solution:\n%s" % str(x))
        idxs = np.where(x == 1)[0]
        # logger.debug("idxs: %s" % str(list(idxs)))
        return region_indexes[idxs]
    else:
        return None


def get_anomaly_descriptions(x, y, model, metrics, opts):
    """ Returns the most compact anomalous region indexes """
    feature_ranges = get_sample_feature_ranges(x)  # will be used to compute volumes
    ordered_wd_idxs = get_most_anomalous_subspace_indexes(model, opts.describe_n_top)
    volumes = get_region_volumes(model, ordered_wd_idxs, feature_ranges)
    ordered_compact_idxs = get_compact_regions(x, y, model, metrics, ordered_wd_idxs, volumes, p=opts.describe_volume_p)
    if ordered_compact_idxs is not None and len(ordered_compact_idxs) > 0:
        logger.debug("logging compact descriptions:")
        for i, ridx in enumerate(ordered_compact_idxs):
            logger.debug("Desc %d: %s" % (i, str(model.all_regions[ridx])))
    return ordered_compact_idxs
