#!/bin/bash

# The script that runs:
#   GLAD: *GL*ocalized *A*nomaly *D*etection via Active Feature Space Suppression
# 
# To run:
# bash ./glad.sh <dataset> <budget> <reruns> <tau> <ensemble_type>
#
# =========
# Examples:
# ---------
#
# Batch Mode GLAD with LODA ensembles
# ---------------------------
# bash ./glad.sh toy2 35 1 0.03 loda
#
# bash ./glad.sh toy2 150 10 0.03 loda

ARGC=$#
if [[ "$ARGC" -gt "0" ]]; then
    DATASET=$1
    BUDGET=$2
    RERUNS=$3
    TAU=$4

    # ==============================
    # Supported ENSEMBLE_TYPE:
    # ------------------------------
    # loda - LODA
    # ------------------------------
    ENSEMBLE_TYPE=$5

fi

# ==============================
# Various AFSS defaults
# Note(s):
#   1. If AFSS_LAMBDA_PRIOR=0, then no feature space prior 
#      will be applied. This is NOT recommended. 
#      Recommended: AFSS_LAMBDA_PRIOR=1.0
#   2. AFSS_MAX_LABELED_REPS is the number of times the labeled instances
#      will be over-sampled during training. Value > 1 helps overcome class-imbalance.
#   3. If AFSS_NODES=0, GLAD will use max(50, <num_ensemble_members> * 3) nodes 
#      in the hidden layer. Else, the number of nodes specified will be used.
# ------------------------------
AFSS_C_TAU=1.0
AFSS_LAMBDA_PRIOR=1.0
AFSS_BIAS_PROB=0.50
AFSS_NODES=0
AFSS_MAX_LABELED_REPS=5
N_EPOCHS=200

# ==============================
# AFSS_NO_PRIME_IND: Whether to skip priming the suppression network
# 0: (default, recommended) Prime the network before feedback loop
# 1: Do not prime the network - pity the analyst
# ------------------------------
AFSS_NO_PRIME_IND=0
if [[ "$AFSS_NO_PRIME_IND" == "1" ]]; then
    AFSS_NO_PRIME="--afss_no_prime"
fi

# ==============================
# TRAIN_BATCH_SIZE determines the number of instances in
# each training mini-batch for optimization.
# For the larger datasets, we increase the batch size for speed.
# ------------------------------
TRAIN_BATCH_SIZE=25
if [[ "$DATASET" == "covtype" || "$DATASET" == "kddcup" ]]; then
    TRAIN_BATCH_SIZE=200
elif [[ "$DATASET" == "mammography" || "$DATASET" == "shuttle_1v23567" ]]; then
    TRAIN_BATCH_SIZE=100
fi

# ==============================
# ENSEMBLE_ONLY_IND: Whether to run AFSS on the generated ensemble.
#   0: Run AFSS after training the baseline unsupervised ensemble
#   1: DO NOT run AFSS after training the baseline unsupervised ensemble
# NOTE(s):
#   1. In either case, if COMPARE_AAD_IND==1, then AAD will be run on the ensemble.
#   2. This is useful of debugging where we only want to test the base ensemble and/or AAD
# ------------------------------
ENSEMBLE_ONLY_IND=0
ENSEMBLE_ONLY=""
if [[ "$ENSEMBLE_ONLY_IND" == "1" ]]; then
    ENSEMBLE_ONLY="--ensemble_only"
fi

# ==============================
# COMPARE_AAD_IND: Whether to run AAD on the same ensembles used with AFSS for comparison.
#   0: DO NOT run AAD
#   1: Run AAD
# NOTE(s):
#   1. This option is relevant only if PYSCRIPT (below) is glad_vs_aad.
#   2. This option is *not* relevant if PYSCRIPT is glad_batch.
# ------------------------------
COMPARE_AAD_IND=1
COMPARE_AAD=""
if [[ "$COMPARE_AAD_IND" == "1" ]]; then
    COMPARE_AAD="--compare_aad"
fi

RAND_SEED=42
OPERATION="glad"

# LODA specific min/max number of projections
# We keep the number of projections low here since the objective of
# AFSS is to be able to incorporate feedback effectively when only a
# few detectors are available.
MIN_K=2
MAX_K=15

# PYSCRIPT=glad_batch.py
# PYMODULE=glad.glad_batch

PYSCRIPT=glad_vs_aad.py
PYMODULE=glad.glad_vs_aad

NAME_PREFIX="${OPERATION}-${ENSEMBLE_TYPE}_${MIN_K}_${MAX_K}-nodes${AFSS_NODES}-bd${BUDGET}-tau${TAU}-bias${AFSS_BIAS_PROB}-c${AFSS_C_TAU}-amr${AFSS_MAX_LABELED_REPS}-r${RERUNS}"
NAME_PREFIX="${NAME_PREFIX//./_}"  # replace '.' with '_'

echo "NAME: ${NAME_PREFIX}"

DATASET_FOLDER=datasets
SCRIPT_PATH=./glad/${PYSCRIPT}
BASE_DIR=
if [ -d "/Users/moy" ]; then
    # personal laptop
    BASE_DIR=../${DATASET_FOLDER}
    LOG_PATH=./temp/${OPERATION}
    PYTHON_CMD="python -m"
    RESULTS_PATH="./temp/${OPERATION}/$DATASET/${NAME_PREFIX}"
    SCRIPT_PATH=${PYMODULE}
elif [ -d "/home/sdas/codebase/bb_python/ad_examples" ]; then
    # cluster environment
    BASE_DIR=/data/doppa/users/sdas/${DATASET_FOLDER}
    LOG_PATH=/data/doppa/users/sdas/temp/${OPERATION}
    PYTHON_CMD="python -m"
    RESULTS_PATH="${BASE_DIR}/results-${OPERATION}/$DATASET/${NAME_PREFIX}"
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

echo "RESULTS: ${RESULTS_PATH}"

LOG_FILE=${LOG_PATH}/${NAME_PREFIX}_${DATASET}.log
echo "LOGS: ${LOG_FILE}"

DATASET_DIR="${BASE_DIR}/anomaly/$DATASET"
ORIG_FEATURES_PATH=${DATASET_DIR}/fullsamples
DATA_FILE=${ORIG_FEATURES_PATH}/${DATASET}_1.csv

mkdir -p "${LOG_PATH}"
mkdir -p "${RESULTS_PATH}"

${PYTHON_CMD} ${SCRIPT_PATH} --dataset=$DATASET --datafile=${DATA_FILE} \
    --results_dir=${RESULTS_PATH} --budget=${BUDGET} --reruns=${RERUNS} \
    --afss_c_tau=${AFSS_C_TAU} --afss_lambda_prior=${AFSS_LAMBDA_PRIOR} \
    --afss_bias_prob=${AFSS_BIAS_PROB} --afss_nodes=${AFSS_NODES} \
    --afss_max_labeled_reps=${AFSS_MAX_LABELED_REPS} --n_epochs=${N_EPOCHS} \
    --loda_mink=${MIN_K} --loda_maxk=${MAX_K} ${AFSS_NO_PRIME} \
    --train_batch_size=${TRAIN_BATCH_SIZE} --randseed=${RAND_SEED} \
    --ensemble_type=${ENSEMBLE_TYPE} ${ENSEMBLE_ONLY} ${COMPARE_AAD} \
    --log_file=${LOG_FILE} --debug

