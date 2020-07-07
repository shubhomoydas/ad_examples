import numpy as np
from ..common.expressions import (
    convert_feature_ranges_to_rules, convert_strings_to_conjunctive_rules,
    convert_conjunctive_rules_to_strings, get_feature_meta_default
)
from .forest_description import CompactDescriber, BayesianRulesetsDescriber


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
        # logger.debug("Rule %d: %d/%d; %s" % (i, n_anom, len(idxs), str(rule)))
    return rules, [str(rule) for rule in rules]


def get_rulesets(x, y, queried, model, meta, opts, bayesian=False):
    if meta is None:
        meta = get_feature_meta_default(x, y)

    queried = np.array(queried, dtype=np.int32)
    discovered_anomalies = queried[np.where(y[queried] == 1)[0]]

    if len(discovered_anomalies) == 0:
        return None, None, None

    compact_describer = CompactDescriber(x, y, model, opts, sample_negative=True)

    regids_top, regions_top = compact_describer.get_top_regions(instance_indexes=discovered_anomalies)

    regids_compact, regions_compact, _ = compact_describer.describe(instance_indexes=queried)

    rules_top, str_rules_top = prepare_conjunctive_rulesets(x, y, meta=meta, region_extents=regions_top, opts=opts)

    rules_compact, str_rules_compact = prepare_conjunctive_rulesets(x, y, meta=meta, region_extents=regions_compact,
                                                                    opts=opts)

    r_bayesian = None
    if bayesian:
        bayesian_describer = BayesianRulesetsDescriber(x, y, model=model, opts=opts,
                                                       meta=meta, candidate_rules=rules_top)
        regids_bayesian, regions_bayesian, rules_bayesian = bayesian_describer.describe(instance_indexes=queried)
        str_rules_bayesian = convert_conjunctive_rules_to_strings(rules_bayesian)
        r_bayesian = rules_bayesian, regions_bayesian, str_rules_bayesian, regids_bayesian

    return ((rules_top, regions_top, str_rules_top, regids_top),
            (rules_compact, regions_compact, str_rules_compact, regids_compact),
            r_bayesian)

