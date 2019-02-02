import math
import random
import operator
from bisect import bisect_left
from collections import defaultdict
import numpy as np
from numpy.random import random as np_random
from random import sample

from ..common.expressions import get_rule_satisfaction_matrix, get_feature_meta_default, \
    convert_strings_to_conjunctive_rules, get_max_len_in_rules
from ..common.utils import logger, get_command_args, configure_logger

"""
Bayesian Rule Set mining By Tong Wang and Peter (Zhen) Li

Reference:
    Wang, Rudin, et al. "Bayesian rule sets for interpretable classification." 
    Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016.
    
This code is based on: https://github.com/zli37/bayesianRuleSet

Bugs in *this* adaptation, if any, were introduced entirely during the
adaptation and are *NOT* due to the original authors Wang, Rudin, et al.

To run tests:
    pythonw -m ad_examples.bayesian_ruleset.bayesian_ruleset --log_file=./temp/bayesian_ruleset.log --debug

"""


def get_confusion(yhat, y):
    if len(yhat) != len(y):
        raise NameError('yhat has different length')
    TP = sum(np.array(yhat) & np.array(y))
    predict_pos = np.sum(yhat)
    FP = predict_pos - TP
    TN = len(y) - np.sum(y) - FP
    FN = len(yhat) - predict_pos - TN
    return TP, FP, TN, FN


def log_betabin(k, n, alpha, beta):
    try:
        c = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
        # logger.debug("c: %f" % c)
    except:
        print('alpha = {}, beta = {}'.format(alpha, beta))
        raise
    if isinstance(k, (list, np.ndarray)):
        if len(k) != len(n):
            print('length of k in %d and length of n is %d' % (len(k), len(n)))
            raise ValueError
        lbeta = []
        for ki, ni in zip(k, n):
            lbeta.append(math.lgamma(ki + alpha) + math.lgamma(ni - ki + beta) - math.lgamma(ni + alpha + beta) + c)
        return np.array(lbeta)
    else:
        return math.lgamma(k + alpha) + math.lgamma(max(1, n - k + beta)) - math.lgamma(n + alpha + beta) + c


def accumulate(iterable, func=operator.add):
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


def find_lt(a, x):
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    else:
        return 0


class BayesianRuleset(object):
    """ Implementation of Bayesian Rule Set mining By Tong Wang and Peter (Zhen) Li

    Significant fragments of this code were borrowed from:
        https://github.com/zli37/bayesianRuleSet

    Reference:
        Wang, Rudin, et al. "Bayesian rule sets for interpretable classification."
        Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016.
    """
    def __init__(self, meta=None, opts=None,
                 support=5, maxlen=3, max_iter=200, chains=1,
                 greedy_initialization=False, greedy_threshold=0.05,
                 propose_threshold=0.1, n_min_support_stop=100):
        self.meta = meta
        self.opts = opts
        self.support = support
        self.maxlen = maxlen
        self.max_iter = max_iter
        self.chains = chains
        self.greedy_initialization = greedy_initialization
        self.greedy_threshold = greedy_threshold
        self.propose_threshold = propose_threshold
        self.n_min_support_stop = n_min_support_stop

        self.alpha_l = None
        self.beta_l = None
        self.alpha_1 = 100
        self.beta_1 = 1
        self.alpha_2 = 100
        self.beta_2 = 1
        self.level = 4

        # support threshold
        self.C = [1]

        self.pattern_space = None
        self.Lup = None
        self.const_denominator = None
        self.P0 = None

        self.rules = None
        self.rules_len = None
        self.supp = None

        self.binary_input = False
        self.predicted_rules = []

    def set_parameters(self, x):
        # number of possible rules, i.e. rule space italic(A) prior
        self.pattern_space = np.ones(self.maxlen + 1)
        # This pattern_space is an approximation
        # because the original code allows
        # the following situation, take tic-tac-toe
        # 1_O == 1 and 1_O_neg == 1, which is impossible
        num_attributes = self.meta.num_features()
        for i in range(1, self.maxlen + 1):
            tmp = 1
            for j in range(num_attributes - i + 1, num_attributes + 1):
                tmp *= j
            self.pattern_space[i] = tmp / math.factorial(i)

        if self.alpha_l is None:
            self.alpha_l = [1 for i in range(self.maxlen + 1)]
        if self.beta_l is None:
            self.beta_l = [(self.pattern_space[i] * 100 + 1) for i in range(self.maxlen + 1)]

        # logger.debug("pattern_space: %s" % str(self.pattern_space))
        # logger.debug("alpha_l: %s" % str(self.alpha_l))
        # logger.debug("beta_l: %s" % str(self.beta_l))

    def precompute(self, y):
        TP, FP, TN, FN = sum(y), 0, len(y)-sum(y), 0
        # self.Lup : p(S|A;alpha_+,beta_+,alpha_-,beta_-)
        # conference paper formula(6)
        self.Lup = (log_betabin(TP, TP+FP, self.alpha_1, self.beta_1)
                    + log_betabin(TN, FN+TN, self.alpha_2, self.beta_2))
        # self.const_denominator : log((|Al|+beta_l-1)/(alpha_l+|Al|-1))
        # conference paper formula(9) denominator
        # logger.debug([max(1., self.pattern_space[i] + self.beta_l[i] - 1) for i in range(self.maxlen+1)])
        # logger.debug([max(1., self.pattern_space[i] + self.alpha_l[i] - 1) for i in range(self.maxlen+1)])
        self.const_denominator = [np.log(max(1., self.pattern_space[i] + self.beta_l[i] - 1)
                                         / max(1., self.pattern_space[i] + self.alpha_l[i] - 1))
                                         for i in range(self.maxlen+1)]
        Kn_count = np.zeros(self.maxlen+1, dtype=int)
        # P0 : maximum prior
        # Ml=0, |Al|= rule space
        # conference paper formula(3)
        # because of log property, + is *
        self.P0 = sum([log_betabin(Kn_count[i], self.pattern_space[i], self.alpha_l[i],
                                   self.beta_l[i]) for i in range(1, self.maxlen+1)])
        # logger.debug("const_denominator: %s" % str(self.const_denominator))
        # logger.debug("P0: %s" % str(self.P0))

    def compute_prob(self, r_matrix, y, rules):
        yhat = self.check_satisfies_at_least_one_rule(r_matrix, rules)
        self.yhat = yhat
        TP, FP, TN, FN = get_confusion(yhat, y)
        # logger.debug("rules (%d): %s" % (len(rules), str(rules)))
        # logger.debug("rules_len: %s" % (str([self.rules_len[x] for x in rules])))
        Kn_count = list(np.bincount([self.rules_len[x] for x in rules], minlength=self.maxlen+1))
        # logger.debug("Kn_count: %s" % str(Kn_count))
        # logger.debug("pattern_space: %s" % str(self.pattern_space))
        # logger.debug("alpha_l: %s" % str(self.alpha_l))
        # logger.debug("beta_l: %s" % str(self.beta_l))
        prior_ChsRules= sum([log_betabin(Kn_count[i],
                                         self.pattern_space[i], self.alpha_l[i],
                                         self.beta_l[i]) for i in range(1, len(Kn_count), 1)])
        likelihood_1 = log_betabin(TP, TP+FP, self.alpha_1, self.beta_1)
        likelihood_2 = log_betabin(TN, FN+TN, self.alpha_2, self.beta_2)
        return [TP, FP, TN, FN], [prior_ChsRules, likelihood_1, likelihood_2]

    def propose(self, rules_curr, y, r_matrix):
        """ Propose a modification to the current set of rules

        :param rules_curr: np.array
            indexes of rules currently in play
        :param y: np.array
        :param r_matrix: np.ndarray
            satisfaction matrix for all the rules in play
        :return: np.array
            proposed set of rules
        """

        # ex is an instance selected at random
        ex = None

        yhat = self.check_satisfies_at_least_one_rule(r_matrix, rules_curr)
        incorr = np.where(y != yhat)[0]
        rules_curr_len = len(rules_curr)

        move = ['clean']
        if len(incorr) > 0:
            ex = sample(list(incorr), 1)[0]
            t = np_random()
            if y[ex] == 1 or rules_curr_len == 1:
                if t < 1.0 / 2 or rules_curr_len == 1:
                    move = ['add']
                else:
                    move = ['cut', 'add']
            else:
                if t < 1.0 / 2:
                    move = ['cut']
                else:
                    move = ['cut', 'add']
        # logger.debug("move: %s" % str(move))

        # 'cut' a rule
        if move[0] == 'cut':
            try:
                if np_random() < self.propose_threshold:
                    candidate = []
                    for rule in rules_curr:
                        if r_matrix[ex, rule]:
                            candidate.append(rule)
                    if len(candidate) == 0:
                        candidate = rules_curr
                    cut_rule = sample(candidate, 1)[0]
                else:
                    p = []
                    all_sum = np.zeros(r_matrix.shape[0], dtype=int)
                    for rule in rules_curr:
                        all_sum = all_sum + r_matrix[:, rule]

                    for ith_rule, rule in enumerate(rules_curr):
                        yhat = (all_sum - r_matrix[:, rule]) > 0
                        TP, FP, TN, FN = get_confusion(yhat, y)
                        p.append(TP.astype(float) / (TP + FP + 1))
                    p = [x - min(p) for x in p]
                    p = np.exp(p)
                    p = np.insert(p, 0, 0)
                    p = np.array(list(accumulate(p)))
                    if p[-1] == 0:
                        index = sample(list(range(len(rules_curr))), 1)[0]
                    else:
                        p = p / p[-1]
                        index = find_lt(p, np_random())
                    cut_rule = rules_curr[index]
                rules_curr.remove(cut_rule)
                move.remove('cut')
            except:
                move.remove('cut')

        # 'add' a rule
        if len(move) > 0 and move[0] == 'add':
            if y[ex] == 1:
                select = np.where((self.supp > self.C[-1]) & ~r_matrix[ex] > 0)[0]
            else:
                select = np.where((self.supp > self.C[-1]) & r_matrix[ex] > 0)[0]
            if len(select) > 0:
                if np_random() < self.propose_threshold:
                    add_rule = sample(select.tolist(), 1)[0]
                else:
                    Yhat_neg_index = np.where(~self.check_satisfies_at_least_one_rule(r_matrix, rules_curr))[0]
                    # In case Yhat_neg_index is []
                    if Yhat_neg_index.shape[0] == 0:
                        return rules_curr
                    mat = r_matrix[Yhat_neg_index.reshape(-1, 1), select].transpose() & y[Yhat_neg_index].astype(int)
                    TP = np.sum(mat, axis=1)
                    FP = np.array(np.sum(r_matrix[Yhat_neg_index.reshape(-1, 1), select], axis=0) - TP)
                    p = (TP.astype(float) / (TP + FP + 1))
                    add_rule = select[sample(list(np.where(p == max(p))[0]), 1)[0]]
                try:
                    if add_rule not in rules_curr:
                        rules_curr.append(add_rule)
                except:
                    pass

        return rules_curr

    def bayesian_pattern_based(self, y, r_matrix, init_rules):

        # |A| : min((rule_space)/2,(rule_space+beta_l-alpha_l)/2)
        self.Asize = [[min(self.pattern_space[l] / 2,
                           0.5 * (self.pattern_space[l] + self.beta_l[l] - self.alpha_l[l])) for l in
                       range(self.maxlen + 1)]]
        # support threshold
        self.C = [1]

        self.maps = defaultdict(list)
        T0 = 1000

        rules_curr = init_rules
        pt_curr = -1000000000
        # now only consider 1 chain
        # it should have been maps[chain]
        self.maps[0].append([-1, [pt_curr / 3, pt_curr / 3, pt_curr / 3],
                             rules_curr, [self.rules[i] for i in rules_curr],
                             []])
        alpha = np.inf
        for ith_iter in range(self.max_iter):
            rules_new = self.propose(rules_curr, y, r_matrix)
            cfmatrix, prob = self.compute_prob(r_matrix, y, rules_new)
            T = T0 ** (1 - ith_iter / self.max_iter)
            pt_new = sum(prob)
            # logger.debug("pt_new: %f, pt_curr: %f, T: %f, float(pt_new - pt_curr): %f" %
            #              (pt_new, pt_curr, T, float(pt_new - pt_curr)))
            if ith_iter > 0:
                # The original Wang et al. code did not have this check
                # and was resulting in RuntimeWarning because we were
                # passing np.exp() a very large number (-pt_curr = 1000000000 in 0-th iter).
                # We do not expect the algorithm performance to change with this check
                # and we can avoid the RuntimeWarning
                alpha = np.exp(float(pt_new - pt_curr) / T)
            if pt_new > sum(self.maps[0][-1][1]):
                if False:
                    logger.debug(
                        '\n** chain = {}, max at iter = {} ** \n accuracy = {}, TP = {},FP = {}, TN = {}, FN = {}\n '
                        'old is {}, pt_new is {}, prior_ChsRules={}, likelihood_1 = {}, likelihood_2 = {}\n '.format(
                            self.chains, ith_iter, (cfmatrix[0] + cfmatrix[2] + 0.0) / len(y), cfmatrix[0], cfmatrix[1],
                            cfmatrix[2], cfmatrix[3], sum(self.maps[0][-1][1]) + 0.1, sum(prob), prob[0], prob[1], prob[2]))
                # logger.debug("rules_new: %s" % str(rules_new))
                # logger.debug("const_denominator: %s" % str(self.const_denominator))
                self.Asize.append([np.floor(min(self.Asize[-1][l],
                                                (-pt_new + self.Lup + self.P0) / max(1., self.const_denominator[l])))
                                   for l in range(self.maxlen + 1)])
                self.const_denominator = [np.log(np.true_divide(max(1., self.pattern_space[l] + self.beta_l[l] - 1),
                                                                max(1., self.Asize[-1][l] + self.alpha_l[l] - 1)))
                                          for l in range(self.maxlen + 1)]
                self.maps[0].append([ith_iter, prob, rules_new,
                                     [self.rules[i] for i in rules_new],
                                     cfmatrix])
                new_supp = np.ceil(np.log(max([np.true_divide(self.pattern_space[l] - self.Asize[-1][l] + self.beta_l[l],
                                                              max(1., self.Asize[-1][l] - 1 + self.alpha_l[l]))
                                               for l in range(1, self.maxlen + 1, 1)])))
                self.C.append(new_supp)
                self.predicted_rules = rules_new
            if np_random() <= alpha:
                rules_curr, pt_curr = rules_new[:], pt_new

        return self.maps[0]

    def screen_rules(self, x, y):
        r_matrix = get_rule_satisfaction_matrix(x, y, self.rules)
        # logger.debug(r_matrix.shape)

        r_matrix_pos = r_matrix[np.where(y == 1)[0]]
        # logger.debug(r_matrix_pos.shape)

        TP = np.asarray(np.sum(r_matrix_pos, axis=0))
        # logger.debug(TP)

        # supp is threshold percentile of how TP a rule is
        supp_select = np.where(TP >= self.support * sum(y) / 100)[0]
        # logger.debug(supp_select)

        self.rules = [self.rules[ridx] for ridx in supp_select]
        sub_r_matrix = r_matrix[:, supp_select]
        # logger.debug(sub_r_matrix.shape)

        self.rules_len = [len(rule) for rule in self.rules]
        self.supp = np.sum(sub_r_matrix, axis=0)
        # logger.debug("rules_len: %s" % str(self.rules_len))
        # logger.debug("supp: %s" % str(self.supp))
        return sub_r_matrix

    def greedy_init(self, r_matrix):
        """ Selects an initial set of rules greedily such that they cover most instances """
        n = r_matrix.shape[0]
        greedy_rules = []
        stop_condition = max(int(n * self.greedy_threshold), self.n_min_support_stop)

        idx = np.arange(n)
        while True:
            TP = np.sum(r_matrix[idx], axis=0)
            # find index of the rule that has the highest support
            rule = sample(np.where(TP == TP.max())[0].tolist(), 1)
            # self.modify_rule(X, X_trans, y, rule[0])
            greedy_rules.extend(rule)
            Z = self.check_satisfies_at_least_one_rule(r_matrix, greedy_rules)
            idx = np.where(Z == False)
            if np.sum(r_matrix[idx], axis=0).max() < stop_condition:
                # if the max number of instances which do not satisfy
                # any of the selected rules falls below threshold,
                # then exit since most instances get selected by some
                # or the other rule.
                return greedy_rules

    def check_satisfies_at_least_one_rule(self, r_matrix, rule_indexes):
        """ Whether each instance satisfies at least one rule among rule_indexes

        :param r_matrix: np.ndarray
            full rule satisfaction matrix
        :param rule_indexes: list-like
            indexes of a subset of rules
        :return: np.array
            Indicator array where 1 indicates whether corresponding instance
            satisfies at least one rule, 0 otherwise

        Note: equivalent to find_rules_Z()
        """
        tmp = r_matrix[:, rule_indexes]
        count_sats = np.sum(tmp, axis=1)
        ret = np.zeros(r_matrix.shape[0], dtype=np.int32)
        ret[np.where(count_sats > 0)[0]] = 1
        return ret

    def fit(self, x, y, rules):
        """ Fit model with training data

        :param x: np.ndarray
        :param y: np.array
        :param rules: list
            list of ConjunctiveRule
        :return: None
        """
        self.set_parameters(x)
        self.precompute(y)
        self.rules = rules
        r_matrix = self.screen_rules(x, y)
        if self.greedy_initialization:
            init = self.greedy_init(r_matrix)
        else:
            init = []
        self.bayesian_pattern_based(y, r_matrix, init)


def sanity_check_bayesian_ruleset(x, y, rules, meta):
    br = BayesianRuleset(meta=meta, opts=None,
                         maxlen=get_max_len_in_rules(rules), max_iter=200)
    br.set_parameters(x)
    br.precompute(y)
    br.rules = rules

    r_matrix = br.screen_rules(x, y)
    logger.debug("r_matrix: %s" % str(r_matrix.shape))

    greedy_rules = br.greedy_init(r_matrix)
    logger.debug("greedy_rules: %s" % str(greedy_rules))

    cm, liks = br.compute_prob(r_matrix, y, greedy_rules)
    logger.debug("cm: %s" % str(cm))
    logger.debug("liks: %s" % str(liks))

    proposed = br.propose(rules_curr=np.arange(len(rules)), y=y, r_matrix=r_matrix)
    logger.debug("proposed: %s" % str(proposed))

    bpb = br.bayesian_pattern_based(y, r_matrix, init_rules=greedy_rules)
    logger.debug(str(bpb))


def test_bayesian_ruleset():
    from common.gen_samples import read_anomaly_dataset
    x, y = read_anomaly_dataset("toy2")
    y = np.asarray(y, dtype=np.int32)

    meta = get_feature_meta_default(x, y)
    # print(meta)

    compact_rules = [
        # These were precomputed; found after 30 feedback iterations with AAD.
        # Each rule corresponds to the anomaly class (label = 1)
        "(F1 > -0.431709) & (F1 <= 2.033541) & (F2 > 3.703597)",
        "(F1 > 4.752354) & (F1 <= 6.210754) & (F2 > 1.581015) & (F2 <= 3.592983)",
        "(F1 > 6.298735) & (F2 > -0.822048) & (F2 <= 3.740281)",
    ]

    rules = convert_strings_to_conjunctive_rules(compact_rules, meta)
    print("Candidate rules:")
    print("  %s" % "\n  ".join([str(rule) for rule in rules]))

    # sanity_check_bayesian_ruleset(x, y, rules, meta)

    br = BayesianRuleset(meta=meta, opts=None,
                         maxlen=get_max_len_in_rules(rules), max_iter=200,
                         n_min_support_stop=20)
    br.fit(x, y, rules)
    print("Selected rules:")
    for idx in br.predicted_rules:
        print("  rule %d: %s" % (idx, str(br.rules[idx])))


if __name__ == "__main__":
    args = get_command_args(debug=False, debug_args=None)
    configure_logger(args)

    random.seed(args.randseed)
    np.random.seed(args.randseed+1)

    test_bayesian_ruleset()
