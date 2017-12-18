#!/bin/bash

# To run:
# bash ./loda_aad.sh <dataset> <budget> <reruns> <tau> <detector_type> <query_type> <query_confident[0|1]> <streaming[0|1]> <streaming_window> <retention_type[0|1]>
#
# =========
# Examples:
# ---------
#
# Batch Mode LODA
# ---------------------------
# bash ./loda_aad.sh toy2 1 1 0.03 13 1 0 0 512 0
#
# Streaming Mode LODA
# -------------------------------
# bash ./loda_aad.sh toy2 35 1 0.03 13 1 0 1 512 0
#
# Compute angle between optimal hyperplane and uniform weight LODA
# -------------------------------
# bash ./loda_aad.sh toy2 35 1 0.03 13 1 0 2 512 0

ARGC=$#
if [[ "$ARGC" -gt "0" ]]; then
    DATASET=$1
    BUDGET=$2
    RERUNS=$3
    TAU=$4

    # ==============================
    # Supported DETECTOR_TYPE:
    # ------------------------------
    # 13 - LODA
    # ------------------------------
    DETECTOR_TYPE=$5

    # ==============================
    # Query types
    # ------------------------------
    # QUERY_DETERMINISIC = 1
    # QUERY_BETA_ACTIVE = 2
    # QUERY_QUANTILE = 3
    # QUERY_RANDOM = 4
    # QUERY_SEQUENTIAL = 5
    # QUERY_GP = 6 (Gaussian Process)
    # QUERY_SCORE_VAR = 7
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

fi

REPS=1  # number of independent data samples (input files)

# LODA configurations
MIN_K=100
MAX_K=200

NORM_UNIT_IND=1
if [[ "$NORM_UNIT_IND" == "1" ]]; then
    NORM_UNIT_SIG="_norm"
    NORM_UNIT="--norm_unit"
else
    NORM_UNIT_SIG=""
    NORM_UNIT=""
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
MAX_BUDGET=300
TOPK=0

MAX_ANOMALIES_CONSTRAINT=1000  # 50
MAX_NOMINALS_CONSTRAINT=1000  # 50

N_SAMPLES=256

N_JOBS=4  # Number of parallel threads

INFERENCE_NAME=loda

RAND_SEED=42

# =========================================
# Input CSV file properties.
# the startcol and labelindex are 1-indexed
# -----------------------------------------
STARTCOL=2
LABELINDEX=1

# SIGMA2 determines the weight on prior.
SIGMA2=0.5

# =====================================================
# Following option determines whether we want to put 
#   a prior on weights. The type of prior is determined 
#   by option UNIF_PRIOR (defined later)
# WITH_PRIOR="" - Puts no prior on weights
# WITH_PRIOR="--withprior" - Adds prior to weights as 
#   determined by UNIF_PRIOR
# -----------------------------------------------------
WITH_PRIOR_IND=1
if [[ "$WITH_PRIOR_IND" == "1" ]]; then
    WITH_PRIOR="--withprior"
    WITH_PRIOR_SIG="_s${SIGMA2}"
else
    WITH_PRIOR=""
    WITH_PRIOR_SIG="_noprior"
fi

UNIF_PRIOR_IND=1
if [[ "$UNIF_PRIOR_IND" == "1" ]]; then
    UNIF_PRIOR="--unifprior"
else
    UNIF_PRIOR=""
fi

# ===========================================
# INIT_TYPE: Determines how the weight vector
#   should be initialized.
# 0 - zeros
# 1 - uniform (and normalized to unit length)
# 2 -  random (and normalized to unit length)
# -------------------------------------------
INIT_TYPE=1

MIN_FEEDBACK_PER_WINDOW=2
MAX_FEEDBACK_PER_WINDOW=20
MAX_WINDOWS=30

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
ALLOW_STREAM_UPDATE_IND=1
if [[ "$ALLOW_STREAM_UPDATE_IND" == "1" ]]; then
    ALLOW_STREAM_UPDATE="--allow_stream_update"
    ALLOW_STREAM_UPDATE_SIG="asu"
fi

if [[ "$STREAMING_IND" == "1" ]]; then
    STREAMING="--streaming"
    STREAMING_SIG="_stream"
    STREAMING_FLAGS="${STREAM_WINDOW}${ALLOW_STREAM_UPDATE_SIG}_mw${MAX_WINDOWS}f${MIN_FEEDBACK_PER_WINDOW}_${MAX_FEEDBACK_PER_WINDOW}_ret${RETENTION_TYPE}"
    PYSCRIPT=loda_aad_stream.py
    PYMODULE=aad.forest_aad_stream
elif [[ "$STREAMING_IND" == "0" ]]; then
    STREAMING=""
    STREAMING_SIG=
    STREAMING_FLAGS=
    PYSCRIPT=loda_aad.py
    PYMODULE=aad.loda_aad
else
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

NAME_PREFIX="${INFERENCE_NAME}_i${DETECTOR_TYPE}_q${QUERY_TYPE}${QUERY_CONFIDENT_SIG}_bd${BUDGET}_tau${TAU}${TAU_SIG}${WITH_PRIOR_SIG}_init${INIT_TYPE}_ca${CA}_cx${CX}_ma${MAX_ANOMALIES_CONSTRAINT}_mn${MAX_NOMINALS_CONSTRAINT}${STREAMING_SIG}${STREAMING_FLAGS}${NORM_UNIT_SIG}"
if [[ "$DETECTOR_TYPE" == "9" ]]; then
    NAME_PREFIX="${INFERENCE_NAME}_trees${N_TREES}_samples${N_SAMPLES}"
fi

DATASET_FOLDER=datasets
if [[ "$DATASET" == "covtype" || "$DATASET" == "kddcup" ]]; then
    DATASET_FOLDER=datasets${STREAMING_SIG}
fi

SCRIPT_PATH=./aad/${PYSCRIPT}
BASE_DIR=
if [ -d "/Users/moy" ]; then
    # personal laptop
    BASE_DIR=/Users/moy/work/git/pyaad/${DATASET_FOLDER}
    # BASE_DIR=./${DATASET_FOLDER}
    LOG_PATH=./temp/aad
    PYTHON_CMD=pythonw
    RESULTS_PATH="temp/aad/$DATASET/${NAME_PREFIX}"
elif [ -d "/home/sdas/codebase/bb_python/ad_examples" ]; then
    # cluster environment
    BASE_DIR=/data/doppa/users/sdas/${DATASET_FOLDER}
    LOG_PATH=/data/doppa/users/sdas/temp/aad${STREAMING_SIG}
    PYTHON_CMD="python -m"
    RESULTS_PATH="${BASE_DIR}/results-aad${STREAMING_SIG}/$DATASET/${NAME_PREFIX}"
    source /home/sdas/py_venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:/home/sdas/codebase/bb_python/ad_examples/python
    #SCRIPT_PATH=/home/sdas/codebase/pyalad/pyalad/${PYSCRIPT}
    #SCRIPT_PATH=/home/sdas/codebase/bb_python/ad_examples/python/aad/${PYSCRIPT}
    SCRIPT_PATH=${PYMODULE}
fi

DATASET_DIR="${BASE_DIR}/anomaly/$DATASET"

LOG_FILE=$LOG_PATH/${NAME_PREFIX}_${DATASET}.log

ORIG_FEATURES_PATH=${DATASET_DIR}/fullsamples
DATA_FILE=${ORIG_FEATURES_PATH}/${DATASET}_1.csv

mkdir -p "${LOG_PATH}"
mkdir -p "${RESULTS_PATH}"

MODEL_FILE=${LOG_PATH}/${NAME_PREFIX}.mdl
LOAD_MODEL=  # "--load_model"
SAVE_MODEL=  # "--save_model"

PLOT2D=
# PLOT2D="--plot2D"

${PYTHON_CMD} ${SCRIPT_PATH} --startcol=$STARTCOL --labelindex=$LABELINDEX --header \
    --filedir=$ORIG_FEATURES_PATH --datafile=$DATA_FILE \
    --resultsdir=$RESULTS_PATH \
    --randseed=$RAND_SEED --dataset=$DATASET --querytype=$QUERY_TYPE \
    --detector_type=$DETECTOR_TYPE --constrainttype=$CONSTRAINT_TYPE \
    --sigma2=$SIGMA2 --runtype=$RUN_TYPE --reps=$REPS --reruns=$RERUNS \
    --budget=$BUDGET --maxbudget=$MAX_BUDGET --topK=$TOPK --init=${INIT_TYPE} \
    --tau=$TAU --mink=${MIN_K} --maxk=${MAX_K} \
    --Ca=$CA --Cn=1 --Cx=$CX $WITH_PRIOR $UNIF_PRIOR $NORM_UNIT \
    --max_anomalies_in_constraint_set=$MAX_ANOMALIES_CONSTRAINT \
    --max_nominals_in_constraint_set=$MAX_NOMINALS_CONSTRAINT \
    --log_file=$LOG_FILE --cachedir=$MODEL_PATH \
    --modelfile=${MODEL_FILE} ${LOAD_MODEL} ${SAVE_MODEL} \
    ${QUERY_CONFIDENT} --max_windows=${MAX_WINDOWS} \
    --min_feedback_per_window=${MIN_FEEDBACK_PER_WINDOW} \
    --max_feedback_per_window=${MAX_FEEDBACK_PER_WINDOW} \
    ${STREAMING} ${ALLOW_STREAM_UPDATE} --stream_window=${STREAM_WINDOW} \
    --retention_type=${RETENTION_TYPE} \
    ${PLOT2D} --debug
