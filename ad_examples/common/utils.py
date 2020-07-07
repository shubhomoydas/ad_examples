import pickle
import logging
import random
import os
import os.path
import errno
import pandas as pd
import numpy as np
import sys
from timeit import default_timer as timer
from datetime import timedelta
import traceback
import pkgutil
import io

from statsmodels.distributions.empirical_distribution import ECDF
import ranking
from ranking import Ranking

import scipy.sparse
from scipy.sparse import csr_matrix
import scipy.stats as stats
import scipy.optimize as opt

from sklearn.linear_model import LogisticRegression as LR
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF

from argparse import ArgumentParser


logger = logging.getLogger(__name__)


def get_option_list():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="airline", required=False,
                        help="Dataset name")
    parser.add_argument("--algo", type=str, default="", required=False,
                        help="Algorithm to apply")
    parser.add_argument("--explore_only", action="store_true", default=False,
                        help="Perform exploratory analysis only instead of more expensive model fitting.")
    parser.add_argument("--budget", type=int, default=1, required=False,
                        help="Budget for feedback")
    parser.add_argument("--n_epochs", type=int, default=200, required=False,
                        help="Max training epochs")
    parser.add_argument("--train_batch_size", type=int, default=25, required=False,
                        help="Batch size for stochastic gradient descent based training methods")
    parser.add_argument("--n_lags", type=int, default=12, required=False,
                        help="Number of time lags for timeseries models")
    parser.add_argument("--normalize_trend", action="store_true", default=False,
                        help="Whether to remove trend in timeseries by successive difference")
    parser.add_argument("--log_transform", action="store_true", default=False,
                        help="Whether to apply element-wise log transform to the timeseries")
    parser.add_argument("--n_anoms", type=int, default=10, required=False,
                        help="Number of top anomalies to report")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to enable output of debug statements")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Whether to plot figures")
    parser.add_argument("--log_file", type=str, default="", required=False,
                        help="File path to debug logs")
    parser.add_argument("--randseed", action="store", type=int, default=42,
                        help="Random seed so that results can be replicated")
    parser.add_argument("--results_dir", action="store", default="./temp",
                        help="Folder where the generated metrics will be stored")
    return parser


def get_command_args(debug=False, debug_args=None, parser=None):
    if parser is None:
        parser = get_option_list()

    if debug:
        unparsed_args = debug_args
    else:
        unparsed_args = sys.argv
        if len(unparsed_args) > 0:
            unparsed_args = unparsed_args[1:len(unparsed_args)]  # script name is first arg

    args = parser.parse_args(unparsed_args)
    return args


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed + 32767)


def matrix(d, nrow=None, ncol=None, byrow=False):
    """Returns the data as a 2-D matrix

    A copy of the same matrix will be returned if input data dimensions are
    same as output data dimensions. Else, a new matrix will be created
    and returned.

    Example:
        d = np.reshape(range(12), (6, 2))
        matrix(d[0:2, :], nrow=2, byrow=True)

    Args:
        d:
        nrow:
        ncol:
        byrow:

    Returns: np.ndarray
    """
    if byrow:
        # fill by row...in python 'C' fills by the last axis
        # therefore, data gets populated one-row at a time
        order = 'C'
    else:
        # fill by column...in python 'F' fills by the first axis
        # therefore, data gets populated one-column at a time
        order = 'F'
    if len(d.shape) == 2:
        d_rows, d_cols = d.shape
    elif len(d.shape) == 1:
        d_rows, d_cols = (1, d.shape[0])
    else:
        raise ValueError("Dimensions more than 2 are not supported")
    if nrow is not None and ncol is None:
        ncol = int(d_rows * d_cols / float(nrow))
    elif ncol is not None and nrow is None:
        nrow = int(d_rows * d_cols / float(ncol))
    if len(d.shape) == 2 and d_rows == nrow and d_cols == ncol:
        return d.copy()
    if not d_rows * d_cols == nrow * ncol:
        raise ValueError("input dimensions (%d, %d) not compatible with output dimensions (%d, %d)" %
                         (d_rows, d_cols, nrow, ncol))
    if isinstance(d, csr_matrix):
        return d.reshape((nrow, ncol), order=order)
    else:
        return np.reshape(d, (nrow, ncol), order=order)


# Ranks in decreasing order
def rank(x, ties_method="average"):
    ox = np.argsort(-x)
    sx = np.argsort(ox)
    if ties_method == "average":
        strategy = ranking.FRACTIONAL
    else:
        strategy = ranking.COMPETITION
    r = Ranking(x[ox], strategy=strategy, start=1)
    rnks = list(r.ranks())
    return np.array(rnks)[sx]


def nrow(x):
    if len(x.shape) == 2:
        return x.shape[0]
    return None


def ncol(x):
    if len(x.shape) == 2:
        return x.shape[1]
    return None


def rbind(m1, m2):
    if m1 is not None and m2 is not None and isinstance(m1, csr_matrix) and isinstance(m2, csr_matrix):
        return scipy.sparse.vstack([m1, m2])
    if m1 is None:
        return np.copy(m2)
    return np.append(m1, m2, axis=0)


def cbind(m1, m2):
    if len(m1.shape) == 1 and len(m2.shape) == 1:
        if len(m1) == len(m2):
            mat = np.empty(shape=(len(m1), 2))
            mat[:, 0] = m1
            mat[:, 1] = m2
            return mat
        else:
            raise ValueError("length of arrays differ: (%d, %d)" % (len(m1), len(m2)))
    return np.append(m1, m2, axis=1)


def sample(x, n):
    shuffle = np.array(x)
    np.random.shuffle(shuffle)
    return shuffle[0:n]


def get_sample_feature_ranges(x):
    min_vals = np.min(x, axis=0)
    max_vals = np.max(x, axis=0)
    return np.hstack([np.transpose([min_vals]), np.transpose([max_vals])])


def append(a1, a2):
    if isinstance(a1, np.ndarray) and len(a1.shape) == 1:
        return np.append(a1, a2)
    a = a1[:]
    if isinstance(a2, list):
        a.extend(a2)
    else:
        a.append(a2)
    return a


def rep(val, n, dtype=float):
    return np.ones(n, dtype=dtype) * val


def power(x, p):
    if isinstance(x, scipy.sparse.csr_matrix):
        return np.sqrt(x.power(p).sum(axis=1))
    else:
        return np.sqrt(np.power(x, p).sum(axis=1))


def quantile(x, q):
    return np.percentile(x, q)


def difftime(endtime, starttime, units="secs"):
    if units == "secs":
        t = timedelta(seconds=endtime-starttime)
    else:
        raise ValueError("units '%s' not supported!" % (units,))
    return t.seconds


def order(x, decreasing=False):
    if decreasing:
        return np.argsort(-x)
    else:
        return np.argsort(x)


def runif(n, min=0.0, max=1.0):
    return stats.uniform.rvs(loc=min, scale=min+max, size=n)


def rnorm(n, mean=0.0, sd=1.0):
    return stats.norm.rvs(loc=mean, scale=sd, size=n)


def pnorm(x, mean=0.0, sd=1.0):
    return stats.norm.cdf(x, loc=mean, scale=sd)


def ecdf(x):
    return ECDF(x)


def matrix_rank(x):
    return np.linalg.matrix_rank(x)


def normalize(w):
    # normalize ||w|| = 1
    w_norm = np.sqrt(w.dot(w))
    if w_norm > 0:
        w = w / w_norm
    return w


def get_random_item(samples, random_state):
    i = random_state.randint(0, samples.shape[0])
    return samples[i]


class SetList(list):
    """ A list class with support for rudimentary set operations
    This is a convenient class when set operations are required while
    preserving data ordering
    """
    def __init__(self, args):
        super(SetList, self).__init__(args)
    def __sub__(self, other):
        return self.__class__([item for item in self if item not in other])


class InstanceList(object):
    def __init__(self, x=None, y=None, ids=None, x_transformed=None):
        self.x = x
        self.y = y
        self.ids = ids
        # support for feature transform
        self.x_transformed = x_transformed
        if self.x is not None and self.x_transformed is not None:
            if self.x.shape[0] != self.x_transformed.shape[0]:
                raise ValueError("number of instances in x (%d) and x_transformed (%d) are not same" %
                                 (self.x.shape[0], self.x_transformed.shape[0]))

    def __len__(self):
        if self.x is not None:
            return self.x.shape[0]
        return 0

    def __repr__(self):
        return "instances(%s, %s, %s, %s)" % (
            "-" if self.x is None else str(self.x.shape),
            "-" if self.y is None else str(len(self.y)),
            "-" if self.x_transformed is None else str(self.x_transformed.shape),
            "-" if self.ids is None else str(len(self.ids))
        )

    def __str__(self):
        return repr(self)

    def add_instances(self, x, y, ids=None, x_transformed=None):
        if self.x is None:
            self.x = x
        else:
            self.x = rbind(self.x, x)

        if self.y is None:
            self.y = y
        elif y is not None:
            self.y = append(self.y, y)

        if self.ids is None:
            self.ids = ids
        elif ids is not None:
            self.ids = append(self.ids, ids)

        if self.x_transformed is None:
            self.x_transformed = x_transformed
        elif x_transformed is not None:
            self.x_transformed = rbind(self.x_transformed, x_transformed)

    def get_instances_at(self, indexes):
        insts_x = self.x[indexes, :]
        insts_y = None
        insts_id = None
        insts_transformed = None
        if self.y is not None:
            insts_y = self.y[indexes]
        if self.ids is not None:
            insts_id = self.ids[indexes]
        if self.x_transformed is not None:
            insts_transformed = self.x_transformed[indexes, :]
        return insts_x, insts_y, insts_id, insts_transformed

    def add_instance(self, x, y=None, id=None, x_transformed=None):
        if self.x is not None:
            self.x = rbind(self.x, x)
        else:
            self.x = x
        if y is not None:
            if self.y is not None:
                self.y = np.append(self.y, [y])
            else:
                self.y = np.array([y], dtype=int)
        if id is not None:
            if self.ids is not None:
                self.ids = np.append(self.ids, [id])
            else:
                self.ids = np.array([id], dtype=int)
        if x_transformed is not None:
            if self.x_transformed is not None:
                self.x_transformed = rbind(self.x_transformed, x_transformed)
            else:
                self.x_transformed = x_transformed

    def retain_with_mask(self, mask):
        self.x = self.x[mask]
        if self.y is not None:
            self.y = self.y[mask]
        if self.ids is not None:
            self.ids = self.ids[mask]
        if self.x_transformed is not None:
            self.x_transformed = self.x_transformed[mask]

    def remove_instance_at(self, index):
        mask = np.ones(self.x.shape[0], dtype=bool)
        mask[index] = False
        self.retain_with_mask(mask)


def append_instance_lists(list1, list2):
    """Merge two instance lists

    Args:
        list1: InstanceList
        list2: InstanceList
    """
    x = None
    if list1.x is not None and list2.x is not None:
        x = np.vstack([list1.x, list2.x])
    y = None
    if list1.y is not None and list2.y is not None:
        y = append(list1.y, list2.y)
    ids = None
    if list1.ids is not None and list2.ids is not None:
        ids = append(list1.ids, list2.ids)
    x_transformed = None
    if list1.x_transformed is not None and list2.x_transformed is not None:
        x_transformed = rbind(list1.x_transformed, list2.x_transformed)
    return InstanceList(x=x, y=y, ids=ids, x_transformed=x_transformed)


class SKLClassifier(object):
    def __init__(self):
        self.clf = None

    def predict(self, x, type="response"):
        if self.clf is None:
            raise ValueError("classifier not initialized/trained...")
        if type == "response":
            y = self.clf.predict_proba(x)
        else:
            y = self.clf.predict(x)
        return y

    def predict_prob_for_class(self, x, cls):
        if self.clf is None:
            raise ValueError("classifier not initialized/trained...")
        clsindex = np.where(self.clf.classes_ == cls)[0][0]
        # logger.debug("class index: %d" % (clsindex,))
        y = self.clf.predict_proba(x)[:, clsindex]
        return y


class DTClassifier(SKLClassifier):
    def __init__(self):
        SKLClassifier.__init__(self)
        self.max_depth = 5

    @staticmethod
    def fit(x, y, max_depth=5):
        classifier = DTClassifier()
        classifier.max_depth = max_depth

        classifier.clf = DT(max_depth=classifier.max_depth)

        classifier.clf.fit(x, y)
        return classifier


class RFClassifier(SKLClassifier):
    def __init__(self):
        SKLClassifier.__init__(self)

    @staticmethod
    def fit(x, y, n_estimators=10, max_depth=None):
        classifier = RFClassifier()
        classifier.clf = RF(n_estimators=n_estimators, max_depth=max_depth)
        classifier.clf.fit(x, y)
        return classifier


class SVMClassifier(SKLClassifier):
    def __init__(self):
        SKLClassifier.__init__(self)
        self.kernel = 'rbf'  # 'linear', 'poly', 'rbf', 'sigmoid'
        self.degree = 3
        self.C = 1.
        self.gamma = 'auto'

    @staticmethod
    def fit(x, y, C=1., kernel='rbf', degree=3, gamma='auto'):
        classifier = SVMClassifier()
        classifier.C = C
        classifier.kernel = kernel
        classifier.degree = degree
        classifier.gamma = gamma

        classifier.clf = svm.SVC(C=classifier.C, kernel=classifier.kernel,
                                 degree=classifier.degree,
                                 gamma=gamma, coef0=0.0, shrinking=True,
                                 probability=True, tol=0.001, cache_size=200,
                                 class_weight=None, verbose=False, max_iter=-1,
                                 random_state=None)

        classifier.clf.fit(x, y)
        return classifier


class LogisticRegressionClassifier(SKLClassifier):
    """
    see:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    def __init__(self):
        SKLClassifier.__init__(self)

    @staticmethod
    def fit(x, y, penalty='l2', C=1., fit_intercept=True):
        classifier = LogisticRegressionClassifier()
        classifier.clf = LR(penalty=penalty, dual=False, tol=0.0001, C=C,
                            fit_intercept=fit_intercept, intercept_scaling=1,
                            class_weight=None, random_state=None, solver='liblinear',
                            max_iter=100, multi_class='ovr', verbose=0)
        classifier.clf.fit(x, y)
        return classifier


def read_csv(file, header=None, sep=',', index_col=None, skiprows=None, usecols=None, encoding='utf8'):
    """Loads data from a CSV

    Returns:
        DataFrame
    """

    if header is not None and header:
        header = 0 # first row is header

    data_df = pd.read_csv(file, header=header, sep=sep, index_col=index_col, skiprows=skiprows, usecols=usecols, encoding=encoding)

    return data_df


def read_resource(resource_path, package_name):
    """ Reads resource files packaged with the python library """
    data = None
    # print("trying resource package: {}; {}".format(package_name, resource_path))
    try:
        data = pkgutil.get_data(package_name, resource_path)
        # print("data type: {}".format(type(data)))
    except:
        pass
    return data


def read_resource_csv(resource_path, package_name='ad_examples.datasets',
                      header=None, sep=',', skiprows=None, usecols=None, encoding='utf8'):
    """ First tries to load the resource from package; if fail, try path as file in 'datasets' folder in source """
    data = read_resource(resource_path=resource_path, package_name=package_name)
    if data is not None:
        # print("Found resource {}".format(resource_path))
        data_df = read_csv(io.BytesIO(data), header=header, sep=sep, skiprows=skiprows, index_col=None, usecols=usecols, encoding=encoding)
    else:
        # perhaps the code is being run from source 'python folder; offer legacy support
        file_resource_path = "ad_examples/datasets/" + resource_path
        print("Loading file (legacy support) '%s' instead of package resource '%s' ..." % (file_resource_path, resource_path))
        data_df = read_csv(file_resource_path, header=header, sep=sep, skiprows=skiprows, index_col=None, usecols=usecols, encoding=encoding)
    return data_df


def dataframe_to_matrix(df, labelindex=0, startcol=1):
    """ Converts a python dataframe in the expected anomaly dataset format to numpy arrays.

    The expected anomaly dataset format is a CSV with the label ('anomaly'/'nominal')
    as the first column. Other columns are numerical features.

    Note: Both 'labelindex' and 'startcol' are 0-indexed.
        This is different from the 'read_data_as_matrix()' method where
        the 'opts' parameter has same-named attributes but are 1-indexed.

    :param df: Pandas dataframe
    :param labelindex: 0-indexed column number that refers to the class label
    :param startcol: 0-indexed column number that refers to the first column in the dataframe
    :return: (np.ndarray, np.array)
    """
    cols = df.shape[1] - startcol
    x = np.zeros(shape=(df.shape[0], cols))
    for i in range(cols):
        x[:, i] = df.iloc[:, i + startcol]
    labels = np.array([1 if df.iloc[i, labelindex] == "anomaly" else 0 for i in range(df.shape[0])], dtype=int)
    return x, labels


def read_data_as_matrix(opts):
    """ Reads data from CSV file and returns numpy matrix.

    Important: Assumes that the first column has the label \in {anomaly, nominal}

    :param opts: AadOpts
        Supplies parameters like file name, whether first row contains header, etc...
    :return: numpy.ndarray
    """
    if opts.labelindex != 1:
        raise ValueError("Invalid label index parameter %d" % opts.labelindex)

    data = read_csv(opts.datafile, header=opts.header, sep=',')
    labelindex = opts.labelindex - 1
    startcol = opts.startcol - 1
    return dataframe_to_matrix(data, labelindex=labelindex, startcol=startcol)


def save(obj, filepath):
    filehandler = open(filepath, 'w')
    pickle.dump(obj, filehandler)
    return obj


def load(filepath):
    filehandler = open(filepath, 'r')
    obj = pickle.load(filehandler)
    return obj


def dir_create(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def exception_to_string(exc):
    exc_type, exc_value, exc_traceback = exc
    return (str(exc_type) + os.linesep + str(exc_value)
            + os.linesep + str(traceback.extract_tb(exc_traceback)))


def configure_logger(args):
    global logger
    logger_format = "%(levelname)s [%(asctime)s]: %(message)s"
    logger_level = logging.DEBUG if args.debug else logging.ERROR
    if args.log_file is not None and args.log_file != "":
        # print "configuring logger to file %s" % (args.log_file,)
        logging.basicConfig(filename=args.log_file,
                            level=logger_level, format=logger_format,
                            filemode='w') # use filemode='a' for APPEND
    else:
        logging.basicConfig(level=logger_level, format=logger_format)
    logger = logging.getLogger("default")


class Timer(object):
    def __init__(self):
        self.start_time = timer()
        self.end_time = None

    def start(self):
        self.start_time = timer()
        self.end_time = None

    def end(self):
        self.end_time = timer()

    def elapsed(self):
        etime = self.end_time
        if etime is None:
            etime = timer()
        return difftime(etime, self.start_time, units="secs")

    def message(self, msg):
        if self.end_time is None:
            self.end_time = timer()
        tdiff = self.elapsed()
        return "%s %f sec(s)" % (msg, tdiff)


def constr_optim(theta, f, grad=None, ui=None, ci=None, a=None, b=None,
                 hessian=None, bounds=None, method="BFGS",
                 outer_iterations=500, debug=False, args=None):
    """solve non-linear constraint optimization with scipy.optimize

    problems have the form:
        minimize f_0(x)
        s.t.
            ui * x >= ci             --> Note: this is opposite of cvxopt
            a * x = b                --> Supported
            #f_k(x) <= 0, k=1..m     --> Not supported

    :param theta: np.array
            initial values. Must be in the domain of f()
    :param f: function that is being minimized
            returns the function evaluation
    :param grad: function
            returns the first derivative
    :param ui: np.ndarray
    :param ci: np.array
    :param a: np.ndarray
    :param b: np.array
    :param mu:
    :param control:
    :param method:
    :param hessian:
    :param outer_iterations:
    :param outer_eps:
    :param debug:
    :param bounds:
    :param args:
    :return:
    """
    x0 = np.array(theta)
    # build the constraint set
    cons = ()
    if ui is not None:
        for i in range(nrow(ui)):
            # cons += ({'type': 'ineq', 'fun': lambda x: x.dot(u_) - c_},)
            def fcons_ineq(x, i=i):
                return x.dot(ui[i, :]) - ci[i]
            cons += ({'type': 'ineq', 'fun': fcons_ineq},)
    if a is not None:
        for i in range(nrow(a)):
            def fcons_eq(x, i=i):
                return x.dot(a[i, :]) - b[i]
            cons += ({'type': 'eq', 'fun': fcons_eq},)
    res = opt.minimize(f, x0,
                       args=() if args is None else args,
                       method=method, jac=grad,
                       hess=hessian, hessp=None, bounds=bounds,
                       constraints=cons, tol=1e-6, callback=None,
                       #options={'gtol': 1e-6, 'maxiter': outer_iterations, 'disp': True}
                       options={'maxiter': outer_iterations, 'disp': debug}
                       )
    if not res.success:
        logger.debug("Optimization Failure:\nStatus: %d; Msg: %s" % (res.status, res.message))
    return res.x, res.success

