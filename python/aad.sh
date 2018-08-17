#!/bin/bash

# To run:
# bash ./aad.sh <dataset> <budget> <reruns> <tau> <detector_type> <query_type> <query_confident[0|1]> <streaming[0|1]> <streaming_window> <retention_type[0|1]> <with_prior[0|1]> <init_type[0|1|2]>
#
# =========
# Examples:
# ---------
#
# Batch Mode Isolation Forest
# ---------------------------
# bash ./aad.sh toy2 35 1 0.03 7 1 0 0 512 0 1 1
#
# Streaming Mode Isolation Forest
# -------------------------------
# bash ./aad.sh toy2 35 1 0.03 11 1 0 1 512 0 1 1
#
# Compute angle between optimal hyperplane and uniform weight Isolation Forest
# -------------------------------
# bash ./aad.sh toy2 35 1 0.03 7 1 0 2 512 0 1 1

ARGC=$#
if [[ "$ARGC" -gt "0" ]]; then
    DATASET=$1
    BUDGET=$2
    RERUNS=$3
    TAU=$4

    # ==============================
    # Supported DETECTOR_TYPE:
    # ------------------------------
    #  7 - AAD_IFOREST
    # 11 - AAD_HSTREES
    # 12 - AAD_RSFOREST
    # 13 - LODA
    # 15 - AAD_MULTIVIEW_FOREST
    # ------------------------------
    DETECTOR_TYPE=$5

    # ==============================
    # Query types
    # ------------------------------
    # QUERY_DETERMINISIC = 1
    # QUERY_TOP_RANDOM = 2
    # QUERY_QUANTILE = 3
    # QUERY_RANDOM = 4
    # QUERY_SEQUENTIAL = 5
    # QUERY_GP = 6 (Gaussian Process)
    # QUERY_SCORE_VAR = 7 (Score Variance)
    # QUERY_CUSTOM_MODULE = 8
    # QUERY_EUCLIDEAN = 9
    #     NOTE: QUERY_CUSTOM_MODULE requires --query_module_name and --query_class_name
    #           to specify the correct query model class. By default:
    #           --query_module_name=aad.query_model_other
    #           --query_class_name=QueryTopDiverseSubspace
    #           The above two options imply that diverse query strategy will be employed
    #           when query_type=8 (i.e., --querytype=8).
    # ------------------------------
    QUERY_TYPE=$6
    
    # ==============================
    # Query Confident
    # ------------------------------
    # 0 - No confidence check
    # 1 - Query only instances having higher score 
    #     than tau-th score with 95% confidence
    # ------------------------------
    QUERY_CONFIDENT=$7
    
    # ==============================
    # Streaming Ind
    # ------------------------------
    # 0 - No streaming
    # 1 - Streaming
    # 2 - Compute angle between optimal hyperplane and 
    #     uniform weight vector. This option is technically
    #     not related to 'streaming', but just a hack.
    # ------------------------------
    STREAMING_IND=$8
    
    STREAM_WINDOW=$9  # 512
    
    # ===========================================
    # RETENTION_TYPE: Determines which instances 
    #   are retained in memory when a new window 
    #   of data arrives.
    # 0 - ignore all part data and overwrite with new
    # 1 - merge current unlabeled data with new window
    #     and then retain top-most anomalous instances.
    # -------------------------------------------
    RETENTION_TYPE=${10}

    # =====================================================
    # Following option determines whether we want to put 
    #   a prior on weights. The type of prior is determined 
    #   by option UNIF_PRIOR (defined later)
    # WITH_PRIOR="" - Puts no prior on weights
    # WITH_PRIOR="--withprior" - Adds prior to weights as 
    #   determined by UNIF_PRIOR
    # -----------------------------------------------------
    WITH_PRIOR_IND=${11}

    # ===========================================
    # INIT_TYPE: Determines how the weight vector
    #   should be initialized.
    # 0 - zeros
    # 1 - uniform (and normalized to unit length)
    # 2 -  random (and normalized to unit length)
    # -------------------------------------------
    INIT_TYPE=${12}

fi

# =====================================================
# The option UNIF_PRIOR is applicable only when WITH_PRIOR_IND=1
# When UNIF_PRIOR_IND = 0:
#   UNIF_PRIOR="" - Puts previous iteration weights as prior
# When UNIF_PRIOR_IND = 1:
#   UNIF_PRIOR="--unifprior" - Adds uniform prior on weights
# -----------------------------------------------------
UNIF_PRIOR_IND=1

# =====================================================
# PRIOR_INFLUENCE
#   0 - Keep the prior influence fixed
#   1 - Lower the prior influence as the number of
#       labeled instances increases
PRIOR_INFLUENCE=1
# -----------------------------------------------------
if [[ "$STREAMING_IND" == "1" ]]; then
    PRIOR_INFLUENCE=0 # to protect against noise in streaming setting
fi

REPS=1  # number of independent data samples (input files)

# =====================================================
# N_EXPLORE and N_BATCH apply to:
#   QUERY_DETERMINISTIC, QUERY_TOP_RANDOM, QUERY_GP, 
#   QUERY_SCORE_VAR, QUERY_CUSTOM_MODULE
N_EXPLORE=10  # number of unlabeled top ranked instances to explore

N_BATCH=1  # Number of queries per feedback iteration
N_BATCH_SIG=""
if [[ "${N_BATCH}" != "1" ]]; then
    N_BATCH_SIG="b${N_BATCH}"
fi


# =====================================================
# QUERY_EUCLIDEAN_DIST_TYPE: Distance type to use when
#    QUERY_TYPE==9.
#   0 - Selected instances in a query batch will be diversified
#       by maximizing average distance to other instances in the
#       same query batch.
#   1 - Selected instances in a query batch will be diversified
#       by maximizing the minimum distance to other instances
#       in the same query batch.
QUERY_EUCLIDEAN_DIST_TYPE=0

QUERY_SIG="q${QUERY_TYPE}${N_BATCH_SIG}"
if [[ "${QUERY_TYPE}" == "2" || "${QUERY_TYPE}" == "8" || "${QUERY_TYPE}" == "9" ]]; then
    QUERY_SIG="q${QUERY_TYPE}n${N_EXPLORE}b${N_BATCH}"
fi

# IMPORTANT: If the detector type is LODA, the data will not be normalized
NORM_UNIT_IND=1

# =====================================================
# DO_NOT_UPDATE_WEIGHTS_IND: Whether weights will be
#    updated with each feedback.
#   0 - Weights will be updated after each feedback
#   1 - Do not update weights. This mode is used to evaluate
#       performance of algorithms in default mode.
# NOTE: Applies only to both batch and streaming mode.
DO_NOT_UPDATE_WEIGHTS_IND=0
if [[ "$DO_NOT_UPDATE_WEIGHTS_IND" == "1" ]]; then
    DO_NOT_UPDATE_WEIGHTS_SIG="_no_upd"
    DO_NOT_UPDATE_WEIGHTS="--do_not_update_weights"
else
    DO_NOT_UPDATE_WEIGHTS_SIG=""
    DO_NOT_UPDATE_WEIGHTS=""
fi

# ==============================
# CONSTRAINT_TYPE:
# ------------------------------
# AAD_CONSTRAINT_NONE = 0 (no constraints)
# [unsupported] AAD_CONSTRAINT_PAIRWISE = 1 (slack vars [0, Inf]; weights [-Inf, Inf])
# [unsupported] AAD_CONSTRAINT_PAIRWISE_WEIGHTS_POSITIVE_SUM_1 = 2 (slack vars [0, Inf]; weights [0, Inf])
# [unsupported] AAD_CONSTRAINT_WEIGHTS_POSITIVE_SUM_1 = 3 (no pairwise; weights [0, Inf], sum(weights)=1)
# AAD_CONSTRAINT_TAU_INSTANCE = 4 (tau-th quantile instance will be used in pairwise constraints)
CONSTRAINT_TYPE=4
if [[ "$CONSTRAINT_TYPE" == "4" ]]; then
    TAU_SIG="_xtau"
else
    TAU_SIG=""
fi

CA=1  #100
CX=1
#CX=0.001
MAX_BUDGET=10000
TOPK=0

# The below two limits will be changed later.
# See further below for detector-specific settings.
MAX_ANOMALIES_CONSTRAINT=1000  # 50
MAX_NOMINALS_CONSTRAINT=1000  # 50

# LODA specific min/max number of projections
# Since we will be labeling many instances, we will need to increase
# the capacity of the model. Therefore, we make MIN_K high for LODA.
MIN_K=300
MAX_K=500

N_SAMPLES=256

N_JOBS=4  # Number of parallel threads

# ==============================
# FOREST_SCORE_TYPE:
# 0 - IFOR_SCORE_TYPE_INV_PATH_LEN
# 1 - IFOR_SCORE_TYPE_INV_PATH_LEN_EXP
# 3 - IFOR_SCORE_TYPE_CONST
# 4 - IFOR_SCORE_TYPE_NEG_PATH_LEN
# 5 - HST_LOG_SCORE_TYPE
# 6 - HST_SCORE_TYPE
# 7 - RSF_LOG_SCORE_TYPE
# 8 - RSF_SCORE_TYPE
# 9 - ORIG_TREE_SCORE_TYPE
# ------------------------------
INFERENCE_NAME="undefined"
FEATURE_PARTITIONS=
FOREST_SCORE_TYPE=3
N_TREES=100
MAX_DEPTH=100  # 9  #15  # 10
FOREST_LEAF_ONLY=1
if [[ "$DETECTOR_TYPE" == "7" ]]; then
    INFERENCE_NAME="if_aad"
    MAX_DEPTH=100
elif [[ "$DETECTOR_TYPE" == "11" ]]; then
    INFERENCE_NAME="hstrees"
    NORM_UNIT_IND=0  # DO NOT normalize for HSTrees
    FOREST_SCORE_TYPE=5  # 5-HSTrees Log Score # 6-HSTrees Score # 9-Original
    N_TREES=50  # 25  # 50  # 30  # 100
    MAX_DEPTH=8  # 10  # 15  # 9
    FOREST_LEAF_ONLY=1  # Allow only leaf nodes for HS Trees at this time...
    CA=1
    
    # HS Trees are usually deep. Therefore the computation is more expensive.
    # We set the max labeled nominal/anomalies to a reasonable number.
    MAX_ANOMALIES_CONSTRAINT=50
    MAX_NOMINALS_CONSTRAINT=50
    
    if [[ "$FOREST_SCORE_TYPE" == "9" ]]; then
        # These are settings in original published literature
        N_TREES=25
        MAX_DEPTH=15
    fi
elif [[ "$DETECTOR_TYPE" == "12" ]]; then
    INFERENCE_NAME="rsforest"
    NORM_UNIT_IND=0  # DO NOT normalize for RSForest
    FOREST_SCORE_TYPE=7  # 7-RSForest Log Score # 8-RSForest Score # 9-Original
    N_TREES=50  # 30
    MAX_DEPTH=8  # 15
    FOREST_LEAF_ONLY=1  # Allow only leaf nodes for RS Forest at this time...
    CA=1
    
    # RS Forest are usually deep. Therefore the computation is more expensive.
    # We set the max labeled nominal/anomalies to a reasonable number.
    MAX_ANOMALIES_CONSTRAINT=50
    MAX_NOMINALS_CONSTRAINT=50
    
    if [[ "$FOREST_SCORE_TYPE" == "9" ]]; then
        # These are settings in original published literature
        N_TREES=30
        MAX_DEPTH=15
    fi
    
elif [[ "$DETECTOR_TYPE" == "13" ]]; then
    INFERENCE_NAME="loda_k${MIN_K}t${MAX_K}"
    NORM_UNIT_IND=0  # DO NOT normalize for LODA
    PRIOR_INFLUENCE=0  # fixed prior for compatibility with earlier results
    
    # For compatibility with earlier published results
    CA=100
    CX=0.001
    
    # Keep the below limits at reasonable values so that
    # execution can be speeded up with minimal difference
    # in accuracy from published results. Note: Published
    # results took max 100 feedback.
    MAX_ANOMALIES_CONSTRAINT=100
    MAX_NOMINALS_CONSTRAINT=100
elif [[ "$DETECTOR_TYPE" == "15" ]]; then
    INFERENCE_NAME="multiview"
    MAX_DEPTH=100
    FEATURE_PARTITIONS="--feature_partitions=120,100"
fi

if [[ "$DETECTOR_TYPE" == "7" || "$DETECTOR_TYPE" == "11" || "$DETECTOR_TYPE" == "12" || "$DETECTOR_TYPE" == "15" ]]; then
    if [[ "$FOREST_LEAF_ONLY" == "1" ]]; then
        FOREST_LEAF_ONLY="--forest_add_leaf_nodes_only"
        FOREST_LEAF_ONLY_SIG="_leaf"
        if [[ "$DETECTOR_TYPE" == "7" ]]; then
            # IFOR_SCORE_TYPE_NEG_PATH_LEN supported only for isolation forest leaf-only
            FOREST_SCORE_TYPE=4
        elif [[ "$DETECTOR_TYPE" == "12" && "$FOREST_SCORE_TYPE" != 9 ]]; then
            # NOTE: scoretype 7 is geometric mean. scoretype 6 is arithmetic mean.
            # When used with AAD where gradients are required for optimization with 
            # SGD, then using geometric means might help because we can work with
            # logarithmic leaf scores.
            FOREST_SCORE_TYPE=7  #7 #8
        elif [[ "$DETECTOR_TYPE" == "15" ]]; then
            # 3 - IFOR_SCORE_TYPE_CONST
            # 4 - IFOR_SCORE_TYPE_NEG_PATH_LEN (isolation forest leaf-only)
            FOREST_SCORE_TYPE=4
        fi
    else
        FOREST_LEAF_ONLY=""
        FOREST_LEAF_ONLY_SIG=""
    fi
else
    FOREST_LEAF_ONLY=""
    FOREST_LEAF_ONLY_SIG=""
fi

if [[ "$NORM_UNIT_IND" == "1" ]]; then
    NORM_UNIT_SIG="_norm"
    NORM_UNIT="--norm_unit"
else
    NORM_UNIT_SIG=""
    NORM_UNIT=""
fi

RAND_SEED=42
OPERATION="aad"

# =========================================
# Input CSV file properties.
# the startcol and labelindex are 1-indexed
# -----------------------------------------
STARTCOL=2
LABELINDEX=1

# SIGMA2 determines the weight on prior.
SIGMA2=0.5

# ===================================================
# TAU_SCORE_TYPE: Determines whether the
#   tau-th score should be computed after
#   each feedback or whether it should be
#   estimated once using the unlabeled training
#   data at the start and then kept fixed. From
#   the analysis point of view, this might have
#   some implications which are yet open for
#   research.
# 0 - Do not use the tau-th score in the hinge loss
# 1 - Determine the tau-th score after each feedback
# 2 - Estimate tau-th score once and keep fixed.
# ---------------------------------------------------
TAU_SCORE_TYPE=1

PRIOR_INFLUENCE_SIG=
if [[ "$PRIOR_INFLUENCE" == "1" ]]; then
    PRIOR_INFLUENCE_SIG="_adapt"
fi

if [[ "$WITH_PRIOR_IND" == "1" ]]; then
    WITH_PRIOR="--withprior"
    WITH_PRIOR_SIG="_s${SIGMA2}${PRIOR_INFLUENCE_SIG}"
else
    WITH_PRIOR=""
    WITH_PRIOR_SIG="_noprior"
fi

if [[ "$UNIF_PRIOR_IND" == "1" ]]; then
    UNIF_PRIOR="--unifprior"
else
    UNIF_PRIOR=""
fi

if [[ "$TAU_SCORE_TYPE" == "0" ]]; then
    FIXED_TAU_SCORE_SIG="_notau"
elif [[ "$TAU_SCORE_TYPE" == "2" ]]; then
    FIXED_TAU_SCORE_SIG="_fixedtau"
else
    # $TAU_SCORE_TYPE" == "1"
    FIXED_TAU_SCORE_SIG=""
fi

MIN_FEEDBACK_PER_WINDOW=2
MAX_FEEDBACK_PER_WINDOW=20
MAX_WINDOWS=30

TILL_BUDGET_IND=1
ALLOW_STREAM_UPDATE_IND=1

# =====================================================
# TREE_UPDATE_TYPE_VAL - applies only to HS Trees and RD Forest
#   0 - Replace the old sample node counts with new
#   1 - Replace old sample counts with the average of the old
#       sample counts and new sample counts.
TREE_UPDATE_TYPE_VAL=0
TREE_UPDATE_TYPE_SIG=""
TREE_UPDATE_TYPE="--tree_update_type=${TREE_UPDATE_TYPE_VAL}"
if [[ "$DETECTOR_TYPE" == "11" || "$DETECTOR_TYPE" == "12" ]]; then
    if [[ "$TREE_UPDATE_TYPE_VAL" == "1" ]]; then
        TREE_UPDATE_TYPE_SIG="_incr"
    fi
fi

# =====================================================
# RESTRICT_LABELED_SET - Applies only to stream setting
#   0 - Use all labeled instances for minimizing AAD loss
#       and enforcing constraints. The problem with this
#       setting is that the proportion of labeled instances
#       will grow relative to the fixed-sized stream window
#       and eventually, optimization will become extremely
#       biased towards historical [labeled] data.
#   1 - Use a random subset of labeled instances while
#       minimizing AAD loss and enforcing constraints.
#       The size of the random subset is determined by the options:
#       --labeled_to_window_ratio : Ratio of labeled instances
#           to stream window size, and
#       --max_labeled_for_stream : The upper bound on the
#           subset size irrespective of labeled_to_window_ratio
# See also: MAX_LABELED_FOR_STREAM, LABELED_TO_WINDOW_RATIO
RESTRICT_LABELED_SET=0

LABELED_TO_WINDOW_RATIO=""
MAX_LABELED_FOR_STREAM=""
if [[ "$RESTRICT_LABELED_SET" == "1" ]]; then
    LABELED_TO_WINDOW_RATIO="--labeled_to_window_ratio=0.2"
    MAX_LABELED_FOR_STREAM="--max_labeled_for_stream=100"
fi


# ===================================================
# N_WEIGHT_UPDATES_AFTER_STREAM: The number of times
# the weights will be updated *without* feedback
# immediately after the model is updated due to a new
# window of data. This is helpful when (say) many trees 
# are dropped we need to run multiple iterations of learning
# ensemble weights as well as estimating the tau-quantile
# score. This option applies to streaming setting only.
N_WEIGHT_UPDATES_AFTER_STREAM=10  # 5
# ---------------------------------------------------

# ===================================================
# FOREST_REPLACE_FRAC: The fraction of trees which will
#   be replaced (*for forests which support this update*)
#   in a streaming setting.
#   This option applies only when ALLOW_STREAM_UPDATE_IND=1
#   and currently only Isolation Forest supports this.
FOREST_REPLACE_FRAC=0.2
# FOREST_REPLACE_FRAC=1.0
# FOREST_REPLACE_FRAC=0.8
# ---------------------------------------------------

if [[ "$TILL_BUDGET_IND" == "1" ]]; then
    TILL_BUDGET="--till_budget"
    TILL_BUDGET_SIG="_tillbudget"
else
    TILL_BUDGET=
    TILL_BUDGET_SIG=
fi

if [[ "$QUERY_CONFIDENT" == "1" ]]; then
    QUERY_CONFIDENT="--query_confident"
    QUERY_CONFIDENT_SIG="_conf"
    MAX_WINDOWS=30
else
    QUERY_CONFIDENT=
    QUERY_CONFIDENT_SIG=
fi

ALLOW_STREAM_UPDATE=
ALLOW_STREAM_UPDATE_SIG=
if [[ "$ALLOW_STREAM_UPDATE_IND" == "1" ]]; then
    ALLOW_STREAM_UPDATE="--allow_stream_update"
    ALLOW_STREAM_UPDATE_SIG="asu"
fi

# ==============================
# CHECK_KL: Whether to check KL-divergence before updating model 
#   in streaming mode. If true, then trees which exceed a threshold
#   based on ${KL_ALPHA} will be discarded and replaced with new ones.
# NOTE: Applies only to streaming mode for forest-based algorithms 
#   when --allow_stream_update flag is set.
# ------------------------------
CHECK_KL_IND=1
CHECK_KL_SIG=""
CHECK_KL=""
KL_ALPHA=0.05
if [[ "$CHECK_KL_IND" == "1" ]]; then
    CHECK_KL_SIG="_KL${KL_ALPHA}"
    CHECK_KL="--check_KL_divergence"
fi

# ==============================
# PRETRAIN_IND:
#   0: Treats the first window of data in streaming setup as fully *unlabeled*.
#   1: Treats the first window of data in streaming setup as fully *labeled*.
# N_PRETRAIN: Number of times to run weight update on the first (labeled)
#   window of data if pretrain is enabled. Applies to streaming setup
#   with pretrain only.
# N_PRETRAIN_NOMINALS: Number of initial labeled nominal instances to retain
#   after pretraining when pretrain is enabled. We should avoid including too
#   many nominals since it would result in class imbalance. Internally, if
#   N_PRETRAIN_NOMINALS > 0, then the algorithm will retain N_PRETRAIN_NOMINALS
#   number of nominal instances which are most distinct from the *initial* labeled
#   anomalies from the initial labeled set.
# NOTE: Applies only to streaming mode for forest-based algorithms.
# ------------------------------
PRETRAIN_IND=0
N_PRETRAIN=50
N_PRETRAIN_NOMINALS=10
PRETRAIN_SIG=""
PRETRAIN=""
if [[ "$PRETRAIN_IND" == "1" ]]; then
    PRETRAIN_SIG="_p${N_PRETRAIN}"
    if [[ "$N_PRETRAIN_NOMINALS" -gt "0" ]]; then
        PRETRAIN_SIG="${PRETRAIN_SIG}n${N_PRETRAIN_NOMINALS}"
    fi
    PRETRAIN="--pretrain"
fi

DATASET_FOLDER=datasets
#if [[ "$DATASET" == "covtype" || "$DATASET" == "kddcup" ]]; then
if [[ "$DATASET" == "covtype" ]]; then
    DATASET_FOLDER=datasets #${STREAMING_SIG}
    MAX_WINDOWS=1000
fi

if [[ "$STREAMING_IND" == "1" ]]; then
    STREAMING="--streaming"
    STREAMING_SIG="_stream"
    STREAMING_FLAGS="${STREAM_WINDOW}${ALLOW_STREAM_UPDATE_SIG}${CHECK_KL_SIG}${PRETRAIN_SIG}_mw${MAX_WINDOWS}f${MIN_FEEDBACK_PER_WINDOW}_${MAX_FEEDBACK_PER_WINDOW}_ret${RETENTION_TYPE}${TILL_BUDGET_SIG}"
    if [[ "${DETECTOR_TYPE}" == "7" && "${ALLOW_STREAM_UPDATE_IND}" == "1" && "${FOREST_REPLACE_FRAC}" != "0.2" && "$CHECK_KL_IND" != "1" ]]; then
        STREAMING_FLAGS="${STREAMING_FLAGS}_f${FOREST_REPLACE_FRAC}"
    fi
    if [[ "${N_WEIGHT_UPDATES_AFTER_STREAM}" != "0" ]]; then
        STREAMING_FLAGS="${STREAMING_FLAGS}_u${N_WEIGHT_UPDATES_AFTER_STREAM}"
    fi
    PYSCRIPT=aad_stream.py
    PYMODULE=aad.aad_stream
elif [[ "$STREAMING_IND" == "0" ]]; then
    STREAMING=""
    STREAMING_SIG=
    STREAMING_FLAGS=
    PYSCRIPT=aad_batch.py
    PYMODULE=aad.aad_batch
else
    OPERATION="angles"
    STREAMING=""
    STREAMING_SIG=""
    STREAMING_FLAGS="_angle"
    PYSCRIPT=test_hyperplane_angles.py
    PYMODULE=aad.test_hyperplane_angles
fi

# ===================================================================
# --runtype=[simple|multi]:
#    Whether the there are multiple sub-samples for the input dataset
# -------------------------------------------------------------------
#RUN_TYPE=simple
RUN_TYPE=multi

NAME_PREFIX="undefined"
if [[ "$DETECTOR_TYPE" == "7" || "$DETECTOR_TYPE" == "11" || "$DETECTOR_TYPE" == "12" || "$DETECTOR_TYPE" == "15" ]]; then
    NAME_PREFIX="${INFERENCE_NAME}${TREE_UPDATE_TYPE_SIG}_trees${N_TREES}_samples${N_SAMPLES}_i${DETECTOR_TYPE}_${QUERY_SIG}${QUERY_CONFIDENT_SIG}_bd${BUDGET}_nscore${FOREST_SCORE_TYPE}${FOREST_LEAF_ONLY_SIG}_tau${TAU}${TAU_SIG}${WITH_PRIOR_SIG}_init${INIT_TYPE}_ca${CA}_cx${CX}_ma${MAX_ANOMALIES_CONSTRAINT}_mn${MAX_NOMINALS_CONSTRAINT}_d${MAX_DEPTH}${STREAMING_SIG}${STREAMING_FLAGS}${DO_NOT_UPDATE_WEIGHTS_SIG}${NORM_UNIT_SIG}${FIXED_TAU_SCORE_SIG}"
elif [[ "$DETECTOR_TYPE" == "9" ]]; then
    NAME_PREFIX="${INFERENCE_NAME}_trees${N_TREES}_samples${N_SAMPLES}"
elif [[ "$DETECTOR_TYPE" == "13" ]]; then
    NAME_PREFIX="${INFERENCE_NAME}_i${DETECTOR_TYPE}_${QUERY_SIG}${QUERY_CONFIDENT_SIG}_bd${BUDGET}_tau${TAU}${TAU_SIG}${WITH_PRIOR_SIG}_init${INIT_TYPE}_ca${CA}_cx${CX}_ma${MAX_ANOMALIES_CONSTRAINT}_mn${MAX_NOMINALS_CONSTRAINT}${STREAMING_SIG}${STREAMING_FLAGS}${DO_NOT_UPDATE_WEIGHTS_SIG}${NORM_UNIT_SIG}"
fi

SCRIPT_PATH=./aad/${PYSCRIPT}
BASE_DIR=
if [ -d "/Users/moy" ]; then
    # personal laptop
    BASE_DIR=../${DATASET_FOLDER}
    LOG_PATH=./temp/${OPERATION}
    PYTHON_CMD=pythonw
    RESULTS_PATH="temp/${OPERATION}/$DATASET/${NAME_PREFIX}"
elif [ -d "/home/sdas/codebase/bb_python/ad_examples" ]; then
    # cluster environment
    BASE_DIR=/data/doppa/users/sdas/${DATASET_FOLDER}
    LOG_PATH=/data/doppa/users/sdas/temp/aad${STREAMING_SIG}
    PYTHON_CMD="python -m"
    RESULTS_PATH="${BASE_DIR}/results-aad${STREAMING_SIG}/$DATASET/${NAME_PREFIX}"
    source /home/sdas/py_venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:/home/sdas/codebase/bb_python/ad_examples/python
    SCRIPT_PATH=${PYMODULE}
else
    # default setting...
    echo "Using default file paths..."
    BASE_DIR=../${DATASET_FOLDER}
    LOG_PATH=./temp/${OPERATION}
    PYTHON_CMD=pythonw
    RESULTS_PATH="temp/${OPERATION}/$DATASET/${NAME_PREFIX}"
fi

DATASET_DIR="${BASE_DIR}/anomaly/$DATASET"

LOG_FILE=$LOG_PATH/${NAME_PREFIX}_${DATASET}.log
echo ${LOG_FILE}

ORIG_FEATURES_PATH=${DATASET_DIR}/fullsamples
DATA_FILE=${ORIG_FEATURES_PATH}/${DATASET}_1.csv

mkdir -p "${LOG_PATH}"
mkdir -p "${RESULTS_PATH}"

MODEL_FILE=${LOG_PATH}/${NAME_PREFIX}.mdl
LOAD_MODEL=  # "--load_model"
SAVE_MODEL=  # "--save_model"

PLOT2D=
PLOT2D="--plot2D"

${PYTHON_CMD} ${SCRIPT_PATH} --startcol=$STARTCOL --labelindex=$LABELINDEX --header \
    --filedir=$ORIG_FEATURES_PATH --datafile=$DATA_FILE \
    --resultsdir=$RESULTS_PATH \
    --randseed=$RAND_SEED --dataset=$DATASET --querytype=$QUERY_TYPE \
    --detector_type=$DETECTOR_TYPE --constrainttype=$CONSTRAINT_TYPE \
    --sigma2=$SIGMA2 --runtype=$RUN_TYPE --reps=$REPS --reruns=$RERUNS \
    --budget=$BUDGET --maxbudget=$MAX_BUDGET --topK=$TOPK --init=${INIT_TYPE} \
    --tau=$TAU --forest_n_trees=$N_TREES --forest_n_samples=$N_SAMPLES \
    --forest_score_type=${FOREST_SCORE_TYPE} ${FOREST_LEAF_ONLY} \
    --forest_max_depth=${MAX_DEPTH} --tau_score_type=${TAU_SCORE_TYPE} \
    --Ca=$CA --Cn=1 --Cx=$CX $WITH_PRIOR $UNIF_PRIOR $NORM_UNIT \
    --mink=${MIN_K} --maxk=${MAX_K} --prior_influence=${PRIOR_INFLUENCE} \
    --max_anomalies_in_constraint_set=$MAX_ANOMALIES_CONSTRAINT \
    --max_nominals_in_constraint_set=$MAX_NOMINALS_CONSTRAINT \
    --n_explore=${N_EXPLORE} --num_query_batch=${N_BATCH} \
    --log_file=$LOG_FILE --cachedir=$MODEL_PATH \
    --modelfile=${MODEL_FILE} ${LOAD_MODEL} ${SAVE_MODEL} \
    ${DO_NOT_UPDATE_WEIGHTS} ${TREE_UPDATE_TYPE} \
    ${MAX_LABELED_FOR_STREAM} ${LABELED_TO_WINDOW_RATIO} \
    ${QUERY_CONFIDENT} --max_windows=${MAX_WINDOWS} \
    --query_euclidean_dist_type=${QUERY_EUCLIDEAN_DIST_TYPE} \
    --min_feedback_per_window=${MIN_FEEDBACK_PER_WINDOW} \
    --max_feedback_per_window=${MAX_FEEDBACK_PER_WINDOW} \
    ${STREAMING} ${ALLOW_STREAM_UPDATE} --stream_window=${STREAM_WINDOW} \
    --retention_type=${RETENTION_TYPE} ${TILL_BUDGET} \
    --forest_replace_frac=${FOREST_REPLACE_FRAC} ${FEATURE_PARTITIONS} \
    ${CHECK_KL} --kl_alpha=${KL_ALPHA} \
    ${PRETRAIN} --n_pretrain=${N_PRETRAIN} --n_pretrain_nominals=${N_PRETRAIN_NOMINALS} \
    --n_weight_updates_after_stream_window=${N_WEIGHT_UPDATES_AFTER_STREAM} \
    ${PLOT2D} --debug
