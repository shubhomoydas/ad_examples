import numpy as np

from ..common.utils import get_sample_feature_ranges
from ..common.expressions import get_feature_meta_default, convert_feature_ranges_to_rules, \
    get_max_len_in_rules, convert_conjunctive_rules_to_feature_ranges
from .forest_aad_detector import is_in_region
from ..bayesian_ruleset.bayesian_ruleset import BayesianRuleset


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
            if rmax == rmin:
                # logger.debug("%d: %f, %f" % (j, rmin, rmax))
                # If the range of a variable is a single value, we just ignore it.
                region_ranges[j] = 1.0
            else:
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
            # logger.debug("No region selected for instance %d" % i)
            pass
    member_insts = np.array(member_insts, dtype=int)
    # logger.debug("#region_indexes: %d, #instance_indexes: %d, #region_membership_indicators: %d" %
    #              (len(region_indexes), len(instance_indexes), len(region_membership_indicators)))
    if len(region_membership_indicators) > 0:
        region_membership_indicators = np.vstack(region_membership_indicators)
    else:
        region_membership_indicators = None
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


class InstancesDescriber(object):
    def __init__(self, x, y, model, opts, sample_negative=False):
        """

        :param x: np.ndarray
            The instance matrix with ALL instances
        :param y: np.array
        :param model: Aad
        :param opts: AadOpts
        :param sample_negative: bool
        """
        self.x = x
        self.y = y
        self.model = model
        self.opts = opts
        self.sample_negative = sample_negative

        self.meta = None

    def sample_instances(self, exclude, n):
        s = np.ones(self.x.shape[0], dtype=np.int32)
        s[exclude] = 0
        s = np.where(s == 1)[0]
        np.random.shuffle(s)
        return s[:n]

    def convert_regions_to_rules(self, regions, region_indexes=None):
        if self.meta is None:
            raise ValueError("must set metadata before calling this function")
        rules, str_rules = convert_feature_ranges_to_rules(regions, self.meta)
        if region_indexes is not None:
            for rule, index in zip(rules, region_indexes):
                rule.id = index
        return rules, str_rules

    def get_top_regions(self, instance_indexes):
        """ Gets the regions having highest anomaly scores

        :param instance_indexes: np.array
        :return: tuple, list(map)
            tuple: (region indexes, #instances among instance_indexes that fall in the region)
            list(map): list of region extents where each region extent is a
                map {feature index: feature range}
        """
        region_indexes = get_regions_for_description(self.x, instance_indexes=instance_indexes,
                                                     model=self.model,
                                                     n_top=self.opts.describe_n_top
                                                     )
        regions = [self.model.all_regions[ridx].region for ridx in region_indexes]
        return region_indexes, regions

    def describe(self, instance_indexes):
        """ Generates descriptions for positive instances among those passed as input

        :param instance_indexes: indexes of instances
        :param sample_negative: Sample random instances and mark them as negative
            Might help in avoiding false positives.
            Number of sampled instances will be len(instance_indexes)
        :return: Rules
        """
        pass


class CompactDescriber(InstancesDescriber):
    """ Generates compact descriptions for instances

    This is different from the method get_compact_regions() in that it
    reduces false positives by excluding negative examples while always
    including positive examples.
    """
    def __init__(self, x, y, model, opts, sample_negative=False):
        InstancesDescriber.__init__(self, x, y, model, opts, sample_negative)

        self.prec_threshold = 0.4
        self.neg_penalty = 1.0

        self.meta = get_feature_meta_default(x, y)

        # will be used to compute volumes
        self.feature_ranges = get_sample_feature_ranges(self.x)

    def get_complexity(self, regions):
        """ Gets the complexity of rules derived from feature ranges that define the regions

        Compute the finite values in defining the regions. These finite values
        become part of the rule. Fewer such values, smaller the rule in length.
        E.g.: let a region be:
            {0: (-inf, 2), 1: (3, 5), 2: (-inf, inf)}
        This region will become a rule of length 3 and complexity = 2^(3-1) = 4:
            feature0 <= 2 & feature1 > 3 & feature1 <= 5

        :param regions: list of dict
        :return: np.array
        """
        complexity = np.zeros(len(regions), dtype=np.float32)
        # logger.debug("regions:\n%s" % str(regions))
        for i, region in enumerate(regions):
            c = 0
            for key, range in region.items():
                if np.isfinite(range[0]):
                    c += 1
                if np.isfinite(range[1]):
                    c += 1
            complexity[i] = c

        # We take (complexity-1) because we assume a rule of length 1 has no complexity
        complexity = np.power(2, np.minimum(complexity-1, 9))
        return complexity

    def describe(self, instance_indexes):
        """ Generates descriptions for positive instances among those passed as input

        :param instance_indexes: indexes of instances
        :param sample_negative: Sample random instances and mark them as negative
            Might help in avoiding false positives.
            Number of sampled instances will be len(instance_indexes)
        :return: Rules
        """

        # separate positive and negative instances
        positive_indexes = instance_indexes[np.where(self.y[instance_indexes] == 1)[0]]
        negative_indexes = instance_indexes[np.where(self.y[instance_indexes] == 0)[0]]

        if self.sample_negative:
            n_neg_samples = len(instance_indexes)
            neg_samples = self.sample_instances(exclude=instance_indexes, n=n_neg_samples)
            negative_indexes = np.append(negative_indexes, neg_samples)

        # get most anomalous regions that cover the positives
        region_indexes, _ = self.get_top_regions(positive_indexes)

        volumes = get_region_volumes(self.model, region_indexes, self.feature_ranges)
        total_vol = np.sum(volumes)
        # logger.debug("Total vol: %f" % total_vol)
        if total_vol > 0:
            volumes = volumes / np.sum(volumes)

        complexities = self.get_complexity([self.model.all_regions[ridx].region for ridx in region_indexes])
        # logger.debug("complexities: %s" % str(list(complexities)))

        # solve the set-covering problem where every positive instance is
        # covered by *some* region.
        ordered_compact_idxs, n_pos, n_neg, vol = self.find_compact(positive_indexes=positive_indexes,
                                                                    negative_indexes=negative_indexes,
                                                                    region_indexes=region_indexes,
                                                                    volumes=volumes, complexities=complexities)

        # filter out less precise regions which probably arose due to noise
        prec = n_pos / (n_pos + n_neg)
        selected = np.where(prec >= self.prec_threshold)[0]
        if len(selected) == 0:
            selected = np.where(prec == max(prec))[0]
        if len(selected) < len(prec):
            # logger.debug("prec: %s" % str(list(prec[selected])))
            ordered_compact_idxs, n_pos, n_neg, vol = \
                ordered_compact_idxs[selected], n_pos[selected], n_neg[selected], vol[selected]

        if ordered_compact_idxs is not None:
            feature_ranges = [self.model.all_regions[ridx].region for ridx in ordered_compact_idxs]
            rules, str_rules = self.convert_regions_to_rules(feature_ranges, region_indexes=ordered_compact_idxs)
            # logger.debug("logging compact descriptions: (%d/%d)" % (len(feature_ranges), len(rules)))
        else:
            # logger.debug("No description found")
            feature_ranges = rules = None

        return ordered_compact_idxs, feature_ranges, rules

    def find_compact(self, positive_indexes, negative_indexes, region_indexes, volumes, complexities):
        """ Returns the most compact set of region_indexes that contain positive examples

            Tries to find regions that include all positive instances while
            minimizing the presence of negative instances.

            :param positive_indexes: np.array
                Indexes of positive instances
            :param negative_indexes: np.array
                Indexes of negative instances
            :param region_indexes: np.array(int)
                Indexes of the candidate regions within which to contain the instances
            :param volumes: np.array(float)
                The volumes of the regions whose indexes are provided in region_indexes
            :return: np.array
                Selected regions
            """
        import cvxopt
        from cvxopt import glpk

        m_positives, m_positive_inds = get_region_memberships(self.x, model=self.model,
                                                              instance_indexes=positive_indexes,
                                                              region_indexes=region_indexes)
        n_positives = np.asarray(np.sum(m_positive_inds, axis=0), dtype=np.float32)
        if negative_indexes is not None and len(negative_indexes) > 0:
            m_negatives, m_negative_inds = get_region_memberships(self.x, model=self.model,
                                                                  instance_indexes=negative_indexes,
                                                                  region_indexes=region_indexes)
            if len(m_negatives) > 0:
                n_negatives = np.asarray(np.sum(m_negative_inds, axis=0), dtype=np.float32)
            else:
                n_negatives = np.zeros(len(region_indexes), dtype=np.float32)
        else:
            n_negatives = np.zeros(len(region_indexes), dtype=np.float32)

        # logger.debug("anom indexes in selected regions (%d):\n%s" % (len(m_positives), str(list(m_positives))))
        # logger.debug("m_positive_inds (%s):\n%s" % (str(m_positive_inds.shape), str(m_positive_inds)))
        # logger.debug("m_negative_inds (%s):\n%s" % (str(m_negative_inds.shape), str(m_negative_inds)))

        nvars = m_positive_inds.shape[1]
        glpk.options['msg_lev'] = 'GLP_MSG_OFF'

        # minimize total volume and the number of negative examples, i.e.:
        #   (volume**p + n_negatives)
        # logger.debug("volumes:\n%s" % str(list(volumes)))
        # median_volume = np.median(volumes)
        # logger.debug("median volume: %f" % median_volume)
        vol_neg = ((volumes ** self.opts.describe_volume_p)
                   + self.neg_penalty * np.multiply(n_negatives, volumes)
                   + complexities)
        # logger.debug("vol_neg:\n%s" % str(list(vol_neg)))
        c = cvxopt.matrix([float(v) for v in vol_neg], tc='d')

        # below states that each anomaly should be included in atleast one region
        G = cvxopt.matrix(-m_positive_inds, tc='d')
        h = cvxopt.matrix([-1] * m_positive_inds.shape[0], tc='d')

        bin_vars = [i for i in range(nvars)]
        (status, soln) = cvxopt.glpk.ilp(c, G, h, B=set(bin_vars))
        # logger.debug("ILP status: %s" % status)
        if soln is not None:
            soln = np.reshape(np.array(soln), newshape=(nvars,))
            # logger.debug("ILP solution:\n%s" % str(soln))
            idxs = np.where(soln == 1)[0]
            if False:
                logger.debug("\nregion_indexes: %d\n%s\nm_positives: %d\n%s" %
                             (len(idxs), str(list(region_indexes[idxs])),
                              len(m_positives), str(list(m_positives))))
            return region_indexes[idxs], n_positives[idxs], n_negatives[idxs], volumes[idxs]
        else:
            return None


class MinimumVolumeCoverDescriber(CompactDescriber):
    def __init__(self, x, y, model, opts):
        CompactDescriber.__init__(self, x, y, model, opts)

    def describe(self, instance_indexes):
        """ Returns descriptions that cover all input instances

        Finds all subspaces which together contain all input instances
        and have the minimum volume. No effort is made to exclude
        false positives.
        """

        instance_indexes = np.array(instance_indexes)
        # get most anomalous regions that cover the positives
        region_indexes = get_regions_for_description(self.x, instance_indexes=instance_indexes,
                                                     model=self.model,
                                                     n_top=self.opts.describe_n_top
                                                     )

        volumes = get_region_volumes(self.model, region_indexes, self.feature_ranges)

        # get the smallest set of smallest regions that together cover all instances
        compact_region_idxs = get_compact_regions(self.x, model=self.model,
                                                  instance_indexes=instance_indexes,
                                                  region_indexes=region_indexes,
                                                  volumes=volumes, p=self.opts.describe_volume_p)
        regions = [self.model.all_regions[ridx].region for ridx in compact_region_idxs]
        rules, str_rules = self.convert_regions_to_rules(regions, region_indexes=compact_region_idxs)
        return compact_region_idxs, regions, rules


class BayesianRulesetsDescriber(InstancesDescriber):
    def __init__(self, x, y, model=None, opts=None, meta=None, candidate_rules=None):
        InstancesDescriber.__init__(self, x, y, model, opts, sample_negative=True)
        self.meta = meta
        if self.meta is None:
            self.meta = get_feature_meta_default(x, y)
        self.candidate_rules = candidate_rules

    def describe(self, instance_indexes):
        instance_indexes = np.array(instance_indexes, dtype=np.int32)

        if self.sample_negative:
            frac_more = 1.0
        else:
            frac_more = 0.0

        if self.sample_negative and frac_more > 0:
            # add additional unlabeled instances as negative examples
            n_neg_samples = int(frac_more * len(instance_indexes))
            neg_samples = self.sample_instances(exclude=instance_indexes, n=n_neg_samples)
            selected_indexes = np.append(instance_indexes, neg_samples)
            x_br = self.x[selected_indexes]
            y_br = np.zeros(len(selected_indexes), dtype=self.y.dtype)  # all negative by default
            y_br[0:len(instance_indexes)] = self.y[instance_indexes]  # known labels
        else:
            x_br = self.x[instance_indexes]
            y_br = self.y[instance_indexes]

        if self.candidate_rules is not None:
            # If candidate rules were provided, use them
            rules = self.candidate_rules
        else:
            # If candidate rule were not provided, then get the top anomalous
            # regions which cover all positive instances and create candidate
            # rules from them.

            # separate positive and negative instances
            positive_indexes = instance_indexes[np.where(self.y[instance_indexes] == 1)[0]]

            # get most anomalous regions that cover the positives
            region_indexes, regions = self.get_top_regions(positive_indexes)

            rules, str_rules = self.convert_regions_to_rules(regions, region_indexes=region_indexes)

        br = BayesianRuleset(meta=self.meta, opts=None, max_iter=200,
                             maxlen=get_max_len_in_rules(rules),
                             n_min_support_stop=int(0.1 * len(y_br)))
        br.fit(x_br, y_br, rules)

        bayesian_rules = [br.rules[idx] for idx in br.predicted_rules]
        if self.candidate_rules is not None:
            bayesian_region_idxs = np.array([br.rules[idx].id if br.rules[idx].id is not None else -1
                                                for idx in br.predicted_rules],
                                            dtype=np.int32)
            feature_ranges = convert_conjunctive_rules_to_feature_ranges(bayesian_rules, self.meta)
        else:
            bayesian_region_idxs = np.array([br.rules[idx].id for idx in br.predicted_rules], dtype=np.int32)
            feature_ranges = [self.model.all_regions[ridx].region for ridx in bayesian_region_idxs]

        return bayesian_region_idxs, feature_ranges, bayesian_rules

