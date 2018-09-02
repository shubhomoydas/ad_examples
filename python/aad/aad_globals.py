from argparse import ArgumentParser
from common.utils import *
from copy import copy

# ==============================
# Initialization Types
# ------------------------------
INIT_ZERO = 0
INIT_UNIF = 1
INIT_RAND = 2
initialization_types = ["zero", "uniform", "random"]

# ==============================
# Detector types
# ------------------------------
SIMPLE_UPD_TYPE = 1
SIMPLE_UPD_TYPE_R_OPTIM = 2
AAD_UPD_TYPE = 3
AAD_SLACK_CONSTR_UPD_TYPE = 4
BASELINE_UPD_TYPE = 5
AAD_ITERATIVE_GRAD_UPD_TYPE = 6
AAD_IFOREST = 7
SIMPLE_PAIRWISE = 8
IFOREST_ORIG = 9
ATGP_IFOREST = 10
AAD_HSTREES = 11
AAD_RSFOREST = 12
LODA = 13
PRECOMPUTED_SCORES = 14
AAD_MULTIVIEW_FOREST = 15

# Detector type names - first is blank string so these are 1-indexed
detector_types = ["", "simple_online", "online_optim", "aad",
                  "aad_slack", "baseline", "iter_grad", "iforest",
                  "simple_pairwise", "iforest_orig", "if_atgp", "hstrees", "rsfor",
                  "loda", "precomputed", "multiview_forest"]

# ==============================
# Tau Score Types
# ------------------------------
TAU_SCORE_NONE = 0
TAU_SCORE_VARIABLE = 1
TAU_SCORE_FIXED = 2
tau_score_types = ["notau", "variabletau", "fixedtau"]


# ==============================
# Forest Score Types
# ------------------------------
IFOR_SCORE_TYPE_INV_PATH_LEN = 0
IFOR_SCORE_TYPE_INV_PATH_LEN_EXP = 1
IFOR_SCORE_TYPE_NORM = 2
IFOR_SCORE_TYPE_CONST = 3
IFOR_SCORE_TYPE_NEG_PATH_LEN = 4
HST_LOG_SCORE_TYPE = 5
HST_SCORE_TYPE = 6
RSF_LOG_SCORE_TYPE = 7
RSF_SCORE_TYPE = 8
ORIG_TREE_SCORE_TYPE = 9

ENSEMBLE_SCORE_LINEAR = 0  # linear combination of scores
ENSEMBLE_SCORE_EXPONENTIAL = 1  # exp(linear combination)
ensemble_score_names = ["linear", "exp"]

# ------------------------------

# ==============================
# Prior Influence
# ------------------------------
# Prior influence is fixed irrespective of how many instances are labeled.
PRIOR_INFLUENCE_FIXED = 0

# Prior is divided by number of labeled instances such that as more instances
# are labeled, the prior's influence decreases.
PRIOR_INFLUENCE_ADAPTIVE = 1
# ------------------------------


# ==============================
# Constraint types when Detector Type is AAD_PAIRWISE_CONSTR_UPD_TYPE
# ------------------------------
AAD_CONSTRAINT_NONE = 0  # no constraints
AAD_CONSTRAINT_PAIRWISE = 1  # slack vars [0, Inf]; weights [-Inf, Inf]
AAD_CONSTRAINT_PAIRWISE_WEIGHTS_POSITIVE_SUM_1 = 2  # slack vars [0, Inf]; weights [0, Inf]
AAD_CONSTRAINT_WEIGHTS_POSITIVE_SUM_1 = 3  # no pairwise; weights [0, Inf], sum(weights)=1
AAD_CONSTRAINT_TAU_INSTANCE = 4  # tau-th quantile instance will be used in pairwise constraints

# Constraint type names - first is blank string so these are 1-indexed
constraint_types = ["no_constraints", "pairwise", "pairwise_pos_wts_sum1", "pos_wts_sum1", "tau_instance"]

# ==============================
# Baseline to use for simple weight inference
# ------------------------------
RELATIVE_MEAN = 1
RELATIVE_QUANTILE = 2

# first is blank to make the name list 1-indexed
RELATIVE_TO_NAMES = ["", "mean", "quantile"]
# ------------------------------


# ==============================
# Query types
# ------------------------------
QUERY_DETERMINISIC = 1
QUERY_TOP_RANDOM = 2
QUERY_QUANTILE = 3
QUERY_RANDOM = 4
QUERY_SEQUENTIAL = 5  # Some MDP/reinforcement learning based stuff for later
QUERY_GP = 6  # Gaussian Process support for later
QUERY_SCORE_VAR = 7  # Based on variance of scores assigned at the leaves
QUERY_CUSTOM_MODULE = 8  # Custom modules which will be dynamically instantiated
QUERY_EUCLIDEAN = 9  # Query such mean/min distance between query instances is maximized
QUERY_SUBSPACE_EUCLIDEAN = 10  # Query instances maximize min distance and minimal overlapping subspaces

# first blank string makes the other names 1-indexed
query_type_names = ["", "top", "toprandom", "quantile", "random", "sequential",
                    "gp", "scvar", "custom", "euclidean", "diverse_euc"]
# ------------------------------


# ==============================
# Euclidean distance-based diversity:
#   QUERY_EUCLIDEAN_DIST_MEAN - Selected instances in a query batch will
#       be diversified by maximizing average distance to other instances
#       in the same query batch.
#   QUERY_EUCLIDEAN_DIST_MIN - Selected instances in a query batch will
#       be diversified by maximizing the minimum distance to other instances
#       in the same query batch.
# ------------------------------
QUERY_EUCLIDEAN_DIST_MEAN = 0
QUERY_EUCLIDEAN_DIST_MIN = 1
query_euclidean_dist_names = ["mean", "min"]


# ==============================
# Stream Retention types determine
# which instances are retained in
# memory when a new window of data
# arrives.
# ------------------------------
STREAM_RETENTION_OVERWRITE = 0
STREAM_RETENTION_TOP_ANOMALOUS = 1
stream_retention_types = ["overwrite", "anomalous"]
# ------------------------------


# ==============================
# Optimization libraries
# ------------------------------
OPTIMLIB_SCIPY = 'scipy'
OPTIMLIB_CVXOPT = 'cvxopt'
# ------------------------------


def get_aad_option_list():
    parser = ArgumentParser()
    parser.add_argument("--filedir", action="store", default="",
                        help="Folder for input files")
    parser.add_argument("--cachedir", action="store", default="",
                        help="Folder where the generated models will be cached for efficiency")
    parser.add_argument("--plotsdir", action="store", default="",
                        help="Folder for output plots")
    parser.add_argument("--resultsdir", action="store", default="",
                        help="Folder where the generated metrics will be stored")
    parser.add_argument("--header", action="store_true", default=False,
                        help="Whether input file has header row")
    parser.add_argument("--startcol", action="store", type=int, default=2,
                        help="Starting column (1-indexed) for data in input CSV")
    parser.add_argument("--labelindex", action="store", type=int, default=1,
                        help="Index of the label column (1-indexed) in the input CSV. Labels should be anomaly/nominal")
    parser.add_argument("--dataset", action="store", default="", required=False,
                        help="Which dataset to use")
    parser.add_argument("--mink", action="store", type=int, default=100,
                        help="Minimum number of random projections for LODA")
    parser.add_argument("--maxk", action="store", type=int, default=200,
                        help="Maximum number of random projections for LODA")
    parser.add_argument("--original_dims", action="store_true", default=False,
                        help="Whether to use original feature space instead of random projections")
    parser.add_argument("--randseed", action="store", type=int, default=42,
                        help="Random seed so that results can be replicated")
    parser.add_argument("--querytype", action="store", type=int, default=QUERY_DETERMINISIC,
                        help="Query strategy to use. 1 - Top, 2 - Beta-active, 3 - Quantile, 4 - Random")
    parser.add_argument("--reps", action="store", type=int, default=0,
                        help="Number of independent dataset samples to use")
    parser.add_argument("--reruns", action="store", type=int, default=0,
                        help="Number of times each sample dataset should be rerun with randomization")
    parser.add_argument("--runtype", action="store", type=str, default="simple",
                        help="[simple|multi] Whether the there are multiple sub-samples for the input dataset")
    parser.add_argument("--budget", action="store", type=int, default=35,
                        help="Number of feedback iterations")
    parser.add_argument("--maxbudget", action="store", type=int, default=100,
                        help="Maximum number of feedback iterations")
    parser.add_argument("--topK", action="store", type=int, default=0,
                        help="Top rank within which anomalies should be present")
    parser.add_argument("--tau", action="store", type=float, default=0.03,
                        help="Top quantile within which anomalies should be present. "
                             "Relevant only when topK<=0")
    parser.add_argument("--tau_nominal", action="store", type=float, default=0.5,
                        help="Top quantile below which nominals should be present. "
                             "Relevant only when simple quantile inference is used")
    parser.add_argument("--withprior", action="store_true", default=False,
                        help="Whether to use weight priors")
    parser.add_argument("--unifprior", action="store_true", default=False,
                        help="Whether to use uniform priors for weights. "
                             "By default, weight from previous iteration "
                             "is used as prior when --withprior is specified.")
    parser.add_argument("--prior_influence", action="store", type=int, default=PRIOR_INFLUENCE_FIXED,
                        help="Whether to keep the prior's influence fixed or decrease it as more data is labeled.")
    parser.add_argument("--tau_score_type", action="store", type=int, default=TAU_SCORE_VARIABLE,
                        help="0 - No tau-score hinge loss, " +
                             "1 - Tau-score computed with each iteration, " +
                             "2 - Tau-score estimated once and then kept fixed")
    parser.add_argument("--init", action="store", type=int, default=INIT_RAND,
                        help="Parameter initialization type (0: uniform, 1: zeros, 2: random)")
    parser.add_argument("--sigma2", action="store", type=float, default=0.5,
                        help="If prior is used on weights, then the variance of prior")
    parser.add_argument("--Ca", action="store", type=float, default=1.,
                        help="Penalty for anomaly")
    parser.add_argument("--Cn", action="store", type=float, default=1.,
                        help="Penalty on nominals")
    parser.add_argument("--Cx", action="store", type=float, default=1.,
                        help="Penalty on constraints")
    parser.add_argument("--detector_type", action="store", type=int, default=AAD_UPD_TYPE,
                        help="Inference algorithm (simple_online(1) / online_optim(2) / aad_pairwise(3))")
    parser.add_argument("--constrainttype", action="store", type=int, default=AAD_CONSTRAINT_TAU_INSTANCE,
                        help="Inference algorithm (simple_online(1) / online_optim(2) / aad_pairwise(3))")
    parser.add_argument("--orderbyviolated", action="store_true", default=False,
                        help="Order by degree of violation when selecting subset of instances for constraints.")
    parser.add_argument("--ignoreAATPloss", action="store_true", default=False,
                        help="Ignore the AATP hinge loss in optimization function.")
    parser.add_argument("--random_instance_at_start", action="store_true", default=False,
                        help="[EXPERIMENTAL] Use random instance as tau-th instance in the first feedback.")
    parser.add_argument("--max_anomalies_in_constraint_set", type=int, default=1000, required=False,
                        help="Maximum number of labeled anomaly instances to use in building pair-wise constraints")
    parser.add_argument("--max_nominals_in_constraint_set", type=int, default=1000, required=False,
                        help="Maximum number of labeled nominal instances to use in building pair-wise constraints")
    parser.add_argument("--relativeto", action="store", type=int, default=RELATIVE_MEAN,
                        help="The relative baseline for simple online (1=mean, 2=quantile)")
    parser.add_argument("--query_search_candidates", action="store", type=int, default=1,
                        help="Number of search candidates to use in each search state (when query_type=5)")
    parser.add_argument("--query_search_depth", action="store", type=int, default=1,
                        help="Depth of search tree (when query_type=5)")
    parser.add_argument("--query_euclidean_dist_type", action="store", type=int, default=QUERY_EUCLIDEAN_DIST_MEAN,
                        help="Type of euclidean diversity to employ for a query batch when querytype %d is selected" %
                             QUERY_EUCLIDEAN)
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to enable output of debug statements")
    parser.add_argument("--log_file", type=str, default="", required=False,
                        help="File path to debug logs")
    parser.add_argument("--optimlib", type=str, default=OPTIMLIB_SCIPY, required=False,
                        help="optimization library to use")
    parser.add_argument("--op", type=str, default="nop", required=False,
                        help="name of operation")
    parser.add_argument("--cachetype", type=str, default="pydata", required=False,
                        help="type of cache (csv|pydata)")

    parser.add_argument("--norm_unit", action="store_true", default=False,
                        help="Whether to normalize tree-based features to unit length")

    parser.add_argument("--scoresdir", type=str, default="", required=False,
                        help="Folder where precomputed scores from ensemble of detectors are stored in CSV format. "
                        "Applies only when runtype=simple")

    parser.add_argument("--ensembletype", type=str, default="regular", required=False,
                        help="[regular|loda] - 'regular' if the file has precomputed scores from ensembles; "
                             "'loda' if LODA projections are to be used as ensemble members. Note: LODA is stochastic, "
                             "hence multiple runs might be required to get an average estimate of accuracy.")
    parser.add_argument("--datafile", type=str, default="", required=False,
                        help="Original data in CSV format. This is used when runtype is 'regular'")
    parser.add_argument("--scoresfile", type=str, default="", required=False,
                        help="Precomputed scores from ensemble of detectors in CSV format. One detector per column;"
                             "first column has label [anomaly|nominal]")

    parser.add_argument("--ifor_n_trees", action="store", type=int, default=100,
                        help="Number of trees for Isolation Forest")
    parser.add_argument("--ifor_n_samples", action="store", type=int, default=256,
                        help="Number of samples to build each tree in Isolation Forest")
    parser.add_argument("--ifor_score_type", action="store", type=int, default=IFOR_SCORE_TYPE_CONST,
                        help="Type of anomaly score computation for a node in Isolation Forest")
    parser.add_argument("--ifor_add_leaf_nodes_only", action="store_true", default=True,
                        help="Whether to include only leaf node regions only or intermediate node regions as well.")
    parser.add_argument("--tree_update_type", action="store", type=int, default=0,  # 0 - TREE_UPD_OVERWRITE
                        help="Type of update to Tree node counts (applies to HS Trees and RS Forest only). " +
                             "0 - overwrite with new counts, 1 - average of previous and current counts.")
    parser.add_argument("--modelfile", action="store", default="",
                        help="Model file path in case the model needs to be saved or loaded. " +
                             "Supported only for Isolation Forest.")
    parser.add_argument("--save_model", action="store_true", default=False,
                        help="Whether to save the trained model")
    parser.add_argument("--load_model", action="store_true", default=False,
                        help="Whether to load a pre-trained model")

    parser.add_argument("--plot2D", action="store_true", default=False,
                        help="Whether to plot the data, trees, and countours. " +
                             "Only supported for 2D data")

    parser.add_argument("--n_jobs", action="store", type=int, default=1,
                        help="Number of parallel threads (if supported)")

    parser.add_argument("--forest_n_trees", action="store", type=int, default=100,
                        help="Number of trees for Forest")
    parser.add_argument("--forest_n_samples", action="store", type=int, default=256,
                        help="Number of samples to build each tree in Forest")
    parser.add_argument("--forest_score_type", action="store", type=int, default=IFOR_SCORE_TYPE_CONST,
                        help="Type of anomaly score computation for a node in Forest")
    parser.add_argument("--ensemble_score", action="store", type=int, default=ENSEMBLE_SCORE_LINEAR,
                        help="How to combine scores from ensemble members")
    parser.add_argument("--forest_add_leaf_nodes_only", action="store_true", default=False,
                        help="Whether to include only leaf node regions only or intermediate node regions as well.")
    parser.add_argument("--forest_max_depth", action="store", type=int, default=15,
                        help="Number of samples to build each tree in Forest")
    parser.add_argument("--forest_replace_frac", action="store", type=float, default=0.2,
                        help="Number of trees in Forest which will be replaced "
                             "in streaming setting. This option applies only with --streaming.")

    parser.add_argument("--num_query_batch", action="store", type=int, default=5,
                        help="Applies only to querytype %d. " % QUERY_DETERMINISIC +
                             "Specifies how many top ranked items to query ")
    parser.add_argument("--n_explore", action="store", type=int, default=2,
                        help="Number of top ranked instances to evaluate during exploration. " +
                             "Applies to querytype(s) [toprandom(%d) | gp(%d) | scvar(%d))" %
                             (QUERY_TOP_RANDOM, QUERY_GP, QUERY_SCORE_VAR))

    parser.add_argument("--streaming", action="store_true", default=False,
                        help="Whether to run the algorithm in streaming setting")
    parser.add_argument("--stream_window", action="store", type=int, default=512,
                        help="Number of instances to hold in buffer before updating the model")
    parser.add_argument("--max_windows", action="store", type=int, default=30,
                        help="Maximum number of windows of streaming data to read")
    parser.add_argument("--min_feedback_per_window", action="store", type=int, default=2,
                        help="Min. number of instances to query per streaming window")
    parser.add_argument("--max_feedback_per_window", action="store", type=int, default=20,
                        help="Max. number of instances to query per streaming window")
    parser.add_argument("--labeled_to_window_ratio", action="store", type=float, default=None,
                        help="The ratio of number of labeled instances to window size. " +
                             "If number of labeled instances is large, " +
                             "a subset will be selected in each feedback iteration to " +
                             "minimize AAD loss and enforce constraints (applies to streaming mode).")
    parser.add_argument("--max_labeled_for_stream", action="store", type=int, default=None,
                        help="Max. number of labeled instances selected for minimizing " +
                             "AAD loss and enforcing constraints (applies to streaming mode).")
    parser.add_argument("--till_budget", action="store_true", default=False,
                        help="Whether to run the streaming algorithm till at least budget")
    parser.add_argument("--allow_stream_update", action="store_true", default=False,
                        help="Update the model when the window buffer is full in the streaming setting")
    parser.add_argument("--do_not_update_weights", action="store_true", default=False,
                        help="Whether to allow weights to be updated in the streaming setting. " +
                             "Set to False to test performance of vanilla streaming algorithm with uniform weights.")
    parser.add_argument("--retention_type", action="store", type=int, default=STREAM_RETENTION_OVERWRITE,
                        help="Determines which instances to retain im memory when a new window of data streams in.")
    parser.add_argument("--query_confident", action="store_true", default=False,
                        help="Whether to query only those top ranked instances for which we " +
                             "are confident the score is at least 1 std-dev higher than tau-th " +
                             "ranked instance' score")
    parser.add_argument("--kl_alpha", action="store", type=float, default=0.05,
                        help="The KL divergence threshold for updating model from stream data. "
                             "All trees whose KL divergence exceeds (1 - kl_alpha) significance will be replaced with each new window of data. "
                             "This only applies to forest-based models in streaming setting and only when the check_KL_divergence flag is True.")
    parser.add_argument("--check_KL_divergence", action="store_true", default=False,
                        help="Whether to check KL-divergence before model update in the streaming setting. "
                             "This only applies to forest-based models in streaming setting.")
    parser.add_argument("--n_weight_updates_after_stream_window", action="store", type=int, default=0,
                        help="Number of times the weight should be updated without feedback after the model "
                             "gets updated with each stream window")
    parser.add_argument("--describe_anomalies", action="store_true", default=False,
                        help="Whether to report compact descriptions for discovered anomalies " +
                             "(supported only for forest-based detectors)")
    parser.add_argument("--describe_n_top", action="store", type=int, default=5,
                        help="Number of top ranked subspaces to use for anomaly descriptions")
    parser.add_argument("--describe_volume_p", action="store", type=int, default=1,
                        help="Exponent for region volume while computing descriptions. " +
                             "Higher power encourages selection of smaller volumes")

    parser.add_argument("--query_module_name", action="store", type=str, default="aad.query_model_other",
                        help="Module/package name of the custom query model to use. " +
                             "Only applies when querytype=" + str(QUERY_CUSTOM_MODULE))
    parser.add_argument("--query_class_name", action="store", type=str, default="QueryTopDiverseSubspace",
                        help="Class name of the custom query model to use. " +
                             "Only applies when querytype=" + str(QUERY_CUSTOM_MODULE))

    parser.add_argument("--feature_partitions", action="store", type=str, default=None,
                        help="Feature partitions for multiview forest. " +
                             "Only applies when detector_type=" + str(AAD_MULTIVIEW_FOREST))

    parser.add_argument("--pretrain", action="store_true", default=False,
                        help="Whether to treat the first window of data in streaming setup as fully labeled.")
    parser.add_argument("--n_pretrain", action="store", type=int, default=10,
                        help="Number of times to run weight update on the first (labeled) window of data "
                             "if pretrain is enabled. Applies to streaming setup with pretrain only.")
    parser.add_argument("--n_pretrain_nominals", action="store", type=int, default=0,
                        help="Number of initial labeled nominal instances to retain after pretraining when "
                             "pretrain is enabled.")

    return parser


def get_aad_command_args(debug=False, debug_args=None):
    parser = get_aad_option_list()

    if debug:
        unparsed_args = debug_args
    else:
        unparsed_args = sys.argv
        if len(unparsed_args) > 0:
            unparsed_args = unparsed_args[1:len(unparsed_args)]  # script name is first arg

    args = parser.parse_args(unparsed_args)

    if args.startcol < 1:
        raise ValueError("startcol is 1-indexed and must be greater than 0")
    if args.labelindex < 1:
        raise ValueError("labelindex is 1-indexed and must be greater than 0")

    # LODA arguments
    args.keep = None
    args.exclude = None
    args.sparsity = np.nan
    args.explain = False
    #args.ntop = 30 # only for explanations
    args.marked = []

    return args


class AadOpts(object):
    def __init__(self, args):
        self.use_rel = False
        self.minfid = min(1, args.reps)
        self.maxfid = args.reps
        self.reruns = args.reruns
        self.runtype = args.runtype
        self.budget = args.budget
        self.maxbudget = args.maxbudget
        self.original_dims = args.original_dims
        self.qtype = args.querytype
        self.thres = 0.0  # used for feature weight in projection vector
        self.gam = 0.0  # used for correlation between projections
        self.nu = 1.0
        self.tau_score_type = args.tau_score_type
        self.Ca = args.Ca  # 100.0,
        self.Cn = args.Cn
        self.Cx = args.Cx  # penalization for slack in pairwise constraints
        self.topK = args.topK
        self.tau = args.tau
        self.detector_type = args.detector_type
        self.constrainttype = args.constrainttype
        self.ignoreAATPloss = args.ignoreAATPloss
        self.orderbyviolated = args.orderbyviolated
        self.withprior = args.withprior  # whether to include prior in loss
        self.unifprior = args.unifprior
        self.prior_influence = args.prior_influence
        self.priorsigma2 = args.sigma2  # 0.2, #0.5, #0.1,
        self.init = args.init
        self.single_inst_feedback = False
        self.random_instance_at_start = args.random_instance_at_start
        self.max_anomalies_in_constraint_set = args.max_anomalies_in_constraint_set
        self.max_nominals_in_constraint_set = args.max_nominals_in_constraint_set
        self.precision_k = [10, 20, 30]
        self.plot_hist = False
        self.relativeto = args.relativeto
        self.tau_nominal = args.tau_nominal
        self.num_query_batch = args.num_query_batch
        self.query_search_candidates = args.query_search_candidates
        self.query_search_depth = args.query_search_depth
        self.query_euclidean_dist_type = args.query_euclidean_dist_type
        self.optimlib = args.optimlib
        self.exclude = None
        self.keep = args.keep
        self.norm_unit = args.norm_unit
        self.randseed = args.randseed

        # LODA specific
        self.mink = args.mink
        self.maxk = max(self.mink, args.maxk)
        self.sparsity = args.sparsity

        # file related options
        self.dataset = args.dataset
        self.header = args.header
        self.startcol = args.startcol
        self.labelindex = args.labelindex
        self.filedir = args.filedir
        self.cachedir = args.cachedir
        self.resultsdir = args.resultsdir
        self.cachetype = args.cachetype
        self.fid = -1
        self.runidx = -1

        self.ensembletype = args.ensembletype
        self.datafile = args.datafile
        self.scoresdir = args.scoresdir
        self.scoresfile = args.scoresfile

        self.ifor_n_trees = args.ifor_n_trees
        self.ifor_n_samples = args.ifor_n_samples
        self.ifor_score_type = args.ifor_score_type
        self.ifor_add_leaf_nodes_only = args.ifor_add_leaf_nodes_only

        self.plot2D = args.plot2D
        self.n_jobs = args.n_jobs

        self.forest_n_trees = args.forest_n_trees
        self.forest_n_samples = args.forest_n_samples
        self.forest_score_type = args.forest_score_type
        self.forest_add_leaf_nodes_only = args.forest_add_leaf_nodes_only
        self.forest_max_depth = args.forest_max_depth
        self.forest_replace_frac = args.forest_replace_frac

        self.n_explore = args.n_explore

        self.ensemble_score = args.ensemble_score

        self.streaming = args.streaming
        self.stream_window = args.stream_window
        self.max_windows = args.max_windows
        self.min_feedback_per_window = args.min_feedback_per_window
        self.max_feedback_per_window = args.max_feedback_per_window
        self.labeled_to_window_ratio = args.labeled_to_window_ratio
        self.max_labeled_for_stream = args.max_labeled_for_stream
        self.till_budget = args.till_budget
        self.allow_stream_update = args.allow_stream_update
        self.retention_type = args.retention_type
        self.query_confident = args.query_confident
        self.tree_update_type = args.tree_update_type
        self.do_not_update_weights = args.do_not_update_weights
        self.kl_alpha = args.kl_alpha
        self.check_KL_divergence = args.check_KL_divergence
        self.n_weight_updates_after_stream_window = args.n_weight_updates_after_stream_window

        self.describe_anomalies = args.describe_anomalies
        self.describe_n_top = args.describe_n_top
        self.describe_volume_p = args.describe_volume_p

        self.query_module_name = args.query_module_name
        self.query_class_name = args.query_class_name

        self.pretrain = args.pretrain
        self.n_pretrain = args.n_pretrain
        self.n_pretrain_nominals = args.n_pretrain_nominals

        self.feature_partitions = None
        if args.feature_partitions is not None:
            str_features = args.feature_partitions.split(',')
            self.feature_partitions = [int(f) for f in str_features]

        self.modelfile = args.modelfile
        self.load_model = args.load_model
        self.save_model = args.save_model

    def is_simple_run(self):
        return self.runtype == "simple"

    def get_fids(self):
        if self.is_simple_run():
            return [0]
        else:
            return range(self.minfid, self.maxfid + 1)

    def get_runidxs(self):
        if self.is_simple_run():
            return [0]
        else:
            return range(1, self.reruns + 1)

    def set_multi_run_options(self, fid, runidx):
        self.fid = fid
        self.runidx = runidx

    def query_name_str(self):
        s = "%s%s" % (query_type_names[self.qtype], "" if not self.query_confident else "_conf")
        if self.num_query_batch > 1 and self.qtype in [QUERY_DETERMINISIC, QUERY_TOP_RANDOM, QUERY_SCORE_VAR]:
            s = "%sb%d" % (s, self.num_query_batch)
        if self.qtype == QUERY_SEQUENTIAL:
            s = "%s_nc%d_d%d" % (s, self.query_search_candidates, self.query_search_depth)
        return s

    def streaming_str(self):
        return "sw%d_asu%s%s%s_mw%df%d_%d_%s%s%s" % (self.stream_window, str(self.allow_stream_update),
                                                   "_KL%0.2f" % self.kl_alpha if self.check_KL_divergence else "",
                                                   "" if not self.do_not_update_weights else "_no_upd",
                                                   self.max_windows, self.min_feedback_per_window,
                                                   self.max_feedback_per_window,
                                                   stream_retention_types[self.retention_type],
                                                   "" if (self.check_KL_divergence or (
                                                              (self.detector_type == AAD_IFOREST or self.detector_type == AAD_MULTIVIEW_FOREST) and
                                                              self.forest_replace_frac == 0.2)
                                                          ) else "_f%0.2f" % self.forest_replace_frac,
                                                     "" if self.n_weight_updates_after_stream_window == 0 else "_u%d" % self.n_weight_updates_after_stream_window
                                                   )

    def detector_type_str(self):
        s = detector_types[self.detector_type]
        if self.detector_type == LODA:
            s = "%s_k%dt%d" % (s, self.mink, self.maxk)
        if self.detector_type == AAD_HSTREES or self.detector_type == AAD_RSFOREST:
            s = "%s%s" % (s, "_incr" if self.tree_update_type == 1 else "")
        if self.streaming:
            s = "%s%s%s" % (s, "_n%d" % self.max_labeled_for_stream if self.max_labeled_for_stream is not None else "",
                            "_r%0.1f" % self.labeled_to_window_ratio if self.labeled_to_window_ratio is not None else "")
        if self.detector_type == AAD_UPD_TYPE:
            return "%s_%s" % (s, constraint_types[self.constrainttype])
        elif (self.detector_type == AAD_IFOREST or self.detector_type == ATGP_IFOREST or
                self.detector_type == AAD_HSTREES or self.detector_type == AAD_RSFOREST or
                self.detector_type == AAD_MULTIVIEW_FOREST):
            return "%s_%s-trees%d_samples%d_nscore%d%s" % \
                   (s, constraint_types[self.constrainttype],
                    self.forest_n_trees, self.forest_n_samples, self.forest_score_type,
                    "_leaf" if self.forest_add_leaf_nodes_only else "")
        else:
            return s

    def do_not_upd_weights_str(self):
        if self.do_not_update_weights:
            return "_no_upd"
        else:
            return ""

    def till_budget_str(self):
        if self.streaming and self.till_budget:
            return "_tillbudget"
        else:
            return ""

    def model_file_prefix(self):
        return "%s_%d_r%d" % (self.dataset, self.fid, self.runidx)

    def get_metrics_path(self):
        prefix = self.get_alad_metrics_name_prefix()
        return os.path.join(self.resultsdir, prefix + "_alad_metrics.pydata")

    def get_metrics_summary_path(self):
        prefix = self.get_alad_metrics_name_prefix()
        return os.path.join(self.resultsdir, prefix + "_alad_summary.pydata")

    def prior_str(self):
        influence_sig = "" if self.prior_influence == PRIOR_INFLUENCE_FIXED else "_adapt"
        sig = (("-unifprior" if self.unifprior else "-prevprior") + influence_sig) if self.withprior else "-noprior"
        return sig

    def get_alad_metrics_name_prefix(self):
        if not self.is_simple_run():
            filesig = ("-fid%d" % (self.fid,)) + ("-runidx%d" % (self.runidx,))
        else:
            filesig = ""
        # optimsig = "-optim_%s" % (self.optimlib,)
        orderbyviolatedsig = "-by_violated" if self.orderbyviolated else ""
        ignoreAATPlosssig = "-noAATP" if self.ignoreAATPloss else ""
        randomInstanceAtStartSig = "-randFirst" if self.random_instance_at_start else ""
        streaming_sig = "-" + self.streaming_str() if self.streaming else ""
        norm_sig = "-norm" if self.norm_unit else ""
        tau_score_sig = "" if self.tau_score_type == TAU_SCORE_VARIABLE else "-%s" % tau_score_types[self.tau_score_type]
        nameprefix = (self.dataset +
                      ("-" + self.detector_type_str()) +
                      ("_" + RELATIVE_TO_NAMES[self.relativeto] if self.detector_type == SIMPLE_UPD_TYPE else "") +
                      randomInstanceAtStartSig +
                      # ("-single" if self.single_inst_feedback else "") +
                      ("-" + self.query_name_str()) +
                      ("-orig" if self.original_dims else "") +
                      self.prior_str() +
                      ("-init_%s" % (initialization_types[self.init],)) +
                      # ("-with_meanrel" if opts.withmeanrelativeloss else "-no_meanrel") +
                      ("-Ca%.0f" % (self.Ca,)) +
                      (("-Cn%0.0f" % (self.Cn,)) if self.Cn != 1 else "") +
                      ("-%d_%d" % (self.minfid, self.maxfid)) +
                      filesig +
                      ("-bd%d" % (self.budget,)) +
                      ("-tau%0.3f" % (self.tau,)) +
                      ("-tau_nominal" if self.detector_type == SIMPLE_UPD_TYPE
                                         and self.relativeto == RELATIVE_QUANTILE
                                         and self.tau_nominal != 0.5 else "") +
                      ("-topK%d" % (self.topK,)) +
                      # optimsig +
                      orderbyviolatedsig +
                      ignoreAATPlosssig +
                      self.do_not_upd_weights_str() +
                      norm_sig +
                      tau_score_sig +
                      streaming_sig + self.till_budget_str()
                      )
        return nameprefix.replace(".", "_")

    def cached_loda_projections_path(self):
        """pre-computed cached projections path"""
        return os.path.join(self.cachedir, 'loda_projs')

    def str_opts(self):
        orderbyviolatedsig = "-by_violated" if self.orderbyviolated else ""
        ignoreAATPlosssig = "-noAATP" if self.ignoreAATPloss else ""
        randomInstanceAtStartSig = "-randFirst" if self.random_instance_at_start else ""
        streaming_sig = "-" + self.streaming_str() if self.streaming else ""
        norm_sig = "-norm" if self.norm_unit else ""
        influence_sig = "" if self.prior_influence == PRIOR_INFLUENCE_FIXED else "_adapt"
        prior_sig = (("-unifprior" if self.unifprior else "-prevprior") +
               str(self.priorsigma2) + influence_sig) if self.withprior else "-noprior"
        tau_score_sig = "" if self.tau_score_type == TAU_SCORE_VARIABLE else "-%s" % tau_score_types[self.tau_score_type]
        srr = (("[" + self.dataset + "]") +
               ("-%s" % (self.detector_type_str(),)) +
               (("_%s" % (RELATIVE_TO_NAMES[self.relativeto],)) if self.detector_type == SIMPLE_UPD_TYPE else "") +
               randomInstanceAtStartSig +
               ("-single" if self.single_inst_feedback else "") +
               ("-query_" + self.query_name_str()) +
               ("-orig" if self.original_dims else "") +
               prior_sig +
               ("-init_%s" % (initialization_types[self.init],)) +
               ("-Ca" + str(self.Ca)) +
               (("-Cn" + str(self.Cn)) if self.Cn != 1 else "") +
               (("-Cx" + str(self.Cx)) if self.Cx != 1 else "") +
               ("-" + str(self.minfid) + "_" + str(self.maxfid)) +
               ("-reruns" + str(self.reruns)) +
               ("-bd" + str(self.budget)) +
               ("-tau" + str(self.tau)) +
               ("-tau_nominal" if self.detector_type == SIMPLE_UPD_TYPE
                                  and self.relativeto == RELATIVE_QUANTILE
                                  and self.tau_nominal != 0.5 else "") +
               ("-topK" + str(self.topK)) +
               ("-orgdim" if self.original_dims else "") +
               # ("sngl_fbk" if self.single_inst_feedback else "") +
               # ("-optimlib_%s" % (self.optimlib,)) +
               orderbyviolatedsig +
               ignoreAATPlosssig +
               self.do_not_upd_weights_str() +
               norm_sig +
               tau_score_sig +
               streaming_sig + self.till_budget_str()
               )
        return srr


def get_first_val_not_marked(vals, marked, start=1):
    for i in range(start, len(vals)):
        f = vals[i]
        if len(np.where(marked == f)[0]) == 0:
            return f
    return None


def get_first_vals_not_marked(vals, marked, n=1, start=1):
    unmarked = []
    for i in range(start, len(vals)):
        f = vals[i]
        if len(np.where(marked == f)[0]) == 0:
            unmarked.append(f)
        if len(unmarked) >= n:
            break
    return np.array(unmarked, dtype=int)


def get_anomalies_at_top(scores, lbls, K):
    ordered_idxs = order(scores)
    sorted_lbls = lbls[ordered_idxs]
    counts = np.zeros(len(K))
    for i in range(len(K)):
        counts[i] = np.sum(sorted_lbls[1:K[i]])
    return counts


class SampleData(object):
    def __init__(self, lbls, fmat, fid):
        self.lbls = lbls
        self.fmat = fmat
        self.fid = fid


def load_samples(filepath, opts, fid=-1):
    """Loads the data file.

    :param filepath: str
    :param opts: Opts
    :param fid: int
    :return: SampleData
    """
    fdata = read_csv(filepath, header=opts.header)
    fmat = np.ndarray(shape=(fdata.shape[0], fdata.shape[1] - opts.startcol + 1), dtype=float)
    fmat[:, :] = fdata.iloc[:, (opts.startcol - 1):fdata.shape[1]]
    lbls = np.array([1 if v == "anomaly" else 0 for v in fdata.iloc[:, 0]], dtype=int)
    return SampleData(lbls=lbls, fmat=fmat, fid=fid)


def load_all_samples(dataset, dirpath, fids, opts):
    """
    Args:
        dataset:
        dirpath:
        fids:
        opts:
            opts.startcol: 1-indexed column number
            opts.labelindex: 1-indexed column number
    Returns: list
    """
    alldata = []
    for fid in fids:
        filename = "%s_%d.csv" % (dataset, fid)
        filepath = os.path.join(dirpath, filename)
        alldata.append(load_samples(filepath, opts, fid=fid))
    return alldata


