from common.expressions import *
from .aad_globals import *
from aad.forest_description import get_regions_for_description, is_forest_detector, \
    get_region_volumes, get_compact_regions
from bayesian_ruleset.bayesian_ruleset import BayesianRuleset, get_max_len_in_rules


def get_top_regions(x, instance_indexes=None, model=None, opts=None):
    """ Gets the regions having highest anomaly scores

    :param x: np.ndarray
        The instance matrix with ALL instances
    :param instance_indexes: np.array(dtype=int)
        Indexes for the instances which need to be described
    :param model: Aad
        Trained Aad model
    :param opts: AadOpts
    :return: tuple, list(map)
        tuple: (region indexes, #instances among instance_indexes that fall in the region)
        list(map): list of region extents where each region extent is a
            map {feature index: feature range}
    """
    if not is_forest_detector(opts.detector_type):
        raise ValueError("Descriptions only supported by forest-based detectors")

    # get top region indexes which will be candidates for rule sets
    reg_idxs = get_regions_for_description(x, instance_indexes=instance_indexes,
                                           model=model, n_top=opts.describe_n_top)
    desc_regions = [model.all_regions[ridx].region for ridx in reg_idxs]
    return reg_idxs, desc_regions


def get_compact_descriptions(x, instance_indexes, model, opts):
    # get feature ranges which will be used to compute volumes
    feature_ranges = get_sample_feature_ranges(x)

    instance_indexes = np.array(instance_indexes)
    top_region_idxs, top_regions = get_top_regions(x, instance_indexes=instance_indexes, model=model, opts=opts)

    # get volume of each candidate region
    volumes = get_region_volumes(model, top_region_idxs, feature_ranges)

    # get the smallest set of smallest regions that together cover all instances
    compact_region_idxs = get_compact_regions(x, model=model,
                                              instance_indexes=instance_indexes,
                                              region_indexes=top_region_idxs,
                                              volumes=volumes, p=opts.describe_volume_p)
    compact_regions = [model.all_regions[ridx].region for ridx in compact_region_idxs]
    return compact_region_idxs, compact_regions


def prepare_conjunctive_rulesets(x, y, meta, region_extents=None, str_rules=None, opts=None):
    """ Prepares ConjunctiveRule structures from the input

    :param x: np.ndarray
    :param y: np.array
    :param meta: FeatureMetadata
    :param region_extents: list of dict
    :param str_rules: list of strings
    :param opts: AadOpts
    :return: list of ConjunctiveRule, list of strings
    """
    if region_extents is not None:
        rules, str_rules = convert_feature_ranges_to_rules(region_extents, meta)
    else:
        rules = convert_strings_to_conjunctive_rules(str_rules, meta)
    for i, rule in enumerate(rules):
        idxs = rule.where_satisfied(x, y)
        rule.set_confusion_matrix(idxs, y)
        n_anom = np.sum(y[idxs])
        logger.debug("Rule %d: %d/%d; %s" % (i, n_anom, len(idxs), str(rule)))
    return rules, [str(rule) for rule in rules]


def get_subset_for_rule_mining(x, y, must_include=None, frac_more=1.0, negative_label=0):
    """ Returns a subset of instances

    The output will set the labels of must_include instances to
    the original labels. Labels on other sampled instances will
    be assumed to be the negative class.

    The returned instances will be input to a fully supervised;
    therefore, we are assigning the negative class to unlabeled
    instances. We assume that labeling instances as negative
    (i.e., nominal) is reasonable for anomaly detection.

    :param x:
    :param y:
    :param must_include: np.array
        Indexes of instances which *must* be included in the output.
        These instances are likely labeled. Other sub-sampled instances
        are probably unlabeled.
    :param frac_more:
    :param negative_label:
    :return:
    """
    idxs = np.zeros(len(y), dtype=np.int32)
    idxs[must_include] = 1
    idxs_to_select = np.where(idxs == 0)[0]
    np.random.shuffle(idxs_to_select)
    n_more = min(len(idxs_to_select), int(len(must_include) * frac_more))
    more_select = idxs_to_select[:n_more]
    selected = list(must_include)
    selected.extend(more_select)

    # assume that labels of must_select instances are known,
    # while others are negative
    y_o = np.ones(len(y), dtype=y.dtype) * negative_label
    y_o[must_include] = y[must_include]

    return x[selected], y_o[selected]


def get_bayesian_rulesets(x, y, queried, rules, meta, opts):
    queried = np.array(queried, dtype=np.int32)
    x_br, y_br = get_subset_for_rule_mining(x, y, queried, frac_more=1.0)

    br = BayesianRuleset(meta=meta, opts=None, max_iter=200,
                         maxlen=get_max_len_in_rules(rules),
                         n_min_support_stop=int(0.1 * len(y_br)))
    br.fit(x_br, y_br, rules)

    rules_bayesian = [br.rules[idx] for idx in br.predicted_rules]
    regions_bayesian = convert_conjunctive_rules_to_feature_ranges(rules_bayesian, meta)
    str_rules_bayesian = convert_conjunctive_rules_to_strings(rules_bayesian)

    return rules_bayesian, regions_bayesian, str_rules_bayesian


def get_rulesets(x, y, queried, model, meta, opts, bayesian=False):
    if meta is None:
        meta = get_feature_meta_default(x, y)

    queried = np.array(queried, dtype=np.int32)
    discovered_anomalies = queried[np.where(y[queried] == 1)[0]]

    _, regions_top = get_top_regions(x, instance_indexes=discovered_anomalies, model=model, opts=opts)
    _, regions_compact = get_compact_descriptions(x, instance_indexes=discovered_anomalies, model=model, opts=opts)

    rules_top, str_rules_top = prepare_conjunctive_rulesets(x, y, meta=meta, region_extents=regions_top, opts=opts)

    rules_compact, str_rules_compact = prepare_conjunctive_rulesets(x, y, meta=meta, region_extents=regions_compact,
                                                                    opts=opts)

    r_bayesian = None
    if bayesian:
        r_bayesian = get_bayesian_rulesets(x, y, queried, rules_top, meta, opts)

    return ((rules_top, regions_top, str_rules_top),
            (rules_compact, regions_compact, str_rules_compact),
            r_bayesian)

