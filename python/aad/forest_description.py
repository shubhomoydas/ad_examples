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


def get_region_indexes_for_instances(x, model=None, n_top=-1):
    region_idxs = np.array(model.get_region_ids(x))
    # logger.debug("#region_idxs: %d" % len(region_idxs))
    # logger.debug("region_idxs:\n%s" % str(list(region_idxs)))
    if n_top < 0:
        n_top = len(region_idxs)
    wd = np.multiply(model.w[region_idxs], model.d[region_idxs])
    ordered_wd_idxs = np.argsort(-wd)[0:min(n_top, len(wd))]  # sort in reverse order
    # logger.debug("ordered_wd_idxs:\n%s" % str(list(ordered_wd_idxs)))
    return region_idxs[ordered_wd_idxs]


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


def get_instances_for_description(x=None, labels=None, metrics=None, instance_indexes=None):
    """ Returns indexes of instances for which we need descriptions

    The instances are selected as follows:
      - If the instance indexes are directly passed, then select those
      - If instance indexes are not passed, then the queried indexes which are
        labeled as true anomalies will be selected.
      - If there are no queried instances, then indexes of all true anomalies will
        be selected.

    :param x: np.ndarray
        The instances in *original feature* space
    :param labels: np.array(int)
        True labels
    :param metrics: MetricsStructure
    :param instance_indexes: np.array(int)
        Indexes of instances whose region memberships need to be checked
    :return:
    """
    if instance_indexes is not None:
        return instance_indexes
    elif metrics is not None and metrics.queried is not None:
        queried = np.array(metrics.queried)
        eval_indexes = np.where(labels[queried] == 1)[0]
        instance_indexes = queried[eval_indexes]
    else:
        instance_indexes = np.where(labels == 1)[0]
    # logger.debug("instance_indexes: %d\n%s" % (len(instance_indexes), str(list(instance_indexes))))
    return instance_indexes


def get_regions_for_description(x, instance_indexes=None, model=None, region_score_only=False, n_top=-1):
    """ Get the set of candidate regions for describing the instances

    Ensures that atmost n_top most anomalous regions that an instance belongs to
    will be present in the output list of regions

    :param x: np.ndarray
        The instances in *original feature* space
    :param instance_indexes: np.array(int)
        Indexes of instances whose region memberships need to be checked
    :param model: Aad
        AAD model
    :param region_score_only: bool
        If False, score regions by the multiplying region anomaly scores with corresponding weights
        If True, score regions by only their anomaly scores
    :param n_top: int
        Number of top ranked regions (by score) per data instance to use
    :return: np.array(int)
    """
    # instance_region_idxs = np.array(model.get_region_ids(x[instance_indexes, :]))
    # logger.debug(instance_region_idxs)
    if region_score_only:
        nwd = -model.d
    else:
        nwd = -np.multiply(model.w, model.d)
    regions = set()
    if n_top < 0:
        n_top = len(nwd)
    for i, inst in enumerate(instance_indexes):
        inst_regs = np.array(model.get_region_ids(x[[inst], :]))
        idxs = np.argsort(nwd[inst_regs])
        ordered_inst_regs = inst_regs[idxs]
        ordered_inst_regs = ordered_inst_regs[0:min(len(ordered_inst_regs), n_top)]
        regions.update(ordered_inst_regs)
    regions = list(regions)
    # logger.debug("selected regions: %d\n%s" % (len(regions), str(regions)))
    return np.array(regions, dtype=int)


def get_region_memberships(x, model=None,
                           instance_indexes=None, region_indexes=None):
    """ Returns which regions the required instances belong to.

    :param x: np.ndarray
        The instances in *original feature* space
    :param labels: np.array(int)
        True labels
    :param model: Aad
        AAD model
    :param metrics: MetricsStructure
    :param instance_indexes: np.array(int)
        Indexes of instances whose region memberships need to be checked
    :param region_indexes: np.array(int)
        Indexes of the candidate regions within which to contain the instances
    :return: np.array(int), np.ndarray(int)
        The first value is the list of instances which belong to any of the regions passed.
        The second is a matrix of binary values, one row per instance. The columns correspond
        to the regions. '1' indicates that an instance belongs to the corresponding region,
        '0' otherwise.
    """
    if instance_indexes is None or len(instance_indexes) == 0:
        return None, None
    nregions = len(region_indexes)
    member_insts = list()
    region_membership_indicators = list()
    for i in instance_indexes:
        inds = np.zeros(nregions, dtype=int)
        for j, ridx in enumerate(region_indexes):
            inds[j] = is_in_region(x[i, :], model.all_regions[ridx].region)
        if np.sum(inds) > 0:
            member_insts.append(i)
            region_membership_indicators.append(np.reshape(inds, newshape=(1, nregions)))
        else:
            logger.debug("No region selected for instance %d" % i)
    member_insts = np.array(member_insts, dtype=int)
    # logger.debug("#region_indexes: %d, #instance_indexes: %d, #region_membership_indicators: %d" %
    #              (len(region_indexes), len(instance_indexes), len(region_membership_indicators)))
    region_membership_indicators = np.vstack(region_membership_indicators)
    return member_insts, region_membership_indicators


def get_compact_regions(x, model=None, instance_indexes=None, region_indexes=None, volumes=None, p=1):
    """ Returns the most compact set of regions among region_indexes that contain the required instances

    :param x: np.ndarray
        The instances in *original feature* space
    :param model: Aad
        AAD model
    :param region_indexes: np.array(int)
        Indexes of the candidate regions within which to contain the instances
    :param volumes: np.array(float)
        The volumes of the regions whose indexes are provided in region_indexes
    :param p: int
        Determines how much to penalize the size of the regions (based on their volumes).
        If this is large, then bigger regions will be strongly discouraged from getting selected.
    :return:
    """
    import cvxopt
    from cvxopt import glpk

    member_insts, member_inds = get_region_memberships(x, model=model,
                                                       instance_indexes=instance_indexes,
                                                       region_indexes=region_indexes)
    # logger.debug("anom indexes in selected regions (%d):\n%s" % (len(member_anoms), str(list(member_anoms))))
    # logger.debug("member_inds (%s):\n%s" % (str(member_inds.shape), str(member_inds)))

    nvars = member_inds.shape[1]
    glpk.options['msg_lev'] = 'GLP_MSG_OFF'

    c = cvxopt.matrix([float(v**p) for v in volumes], tc='d')  # minimize total volume**p

    # below states that each anomaly should be included in atleast one region
    G = cvxopt.matrix(-member_inds, tc='d')
    h = cvxopt.matrix([-1] * member_inds.shape[0], tc='d')

    bin_vars = [i for i in range(nvars)]
    (status, soln) = cvxopt.glpk.ilp(c, G, h, B=set(bin_vars))
    # logger.debug("ILP status: %s" % status)
    if soln is not None:
        soln = np.reshape(np.array(soln), newshape=(nvars,))
        # logger.debug("ILP solution:\n%s" % str(soln))
        idxs = np.where(soln == 1)[0]
        if False:
            logger.debug("\nregion_indexes: %d\n%s\nmember_insts: %d\n%s" %
                         (len(idxs), str(list(region_indexes[idxs])),
                          len(member_insts), str(list(member_insts))))
        return region_indexes[idxs]
    else:
        return None


def get_anomaly_descriptions(x, y, model, metrics, opts):
    """ Returns the most compact anomalous region indexes """
    instance_indexes = get_instances_for_description(x, labels=y, metrics=metrics)
    feature_ranges = get_sample_feature_ranges(x)  # will be used to compute volumes
    reg_idxs = get_regions_for_description(x, instance_indexes=instance_indexes,
                                           model=model, n_top=opts.describe_n_top)
    volumes = get_region_volumes(model, reg_idxs, feature_ranges)
    ordered_compact_idxs = get_compact_regions(x, instance_indexes=instance_indexes,
                                               model=model,
                                               region_indexes=reg_idxs, volumes=volumes,
                                               p=opts.describe_volume_p)
    if ordered_compact_idxs is not None and len(ordered_compact_idxs) > 0:
        logger.debug("logging compact descriptions:")
        for i, ridx in enumerate(ordered_compact_idxs):
            logger.debug("Desc %d: %s" % (i, str(model.all_regions[ridx])))
    return ordered_compact_idxs
