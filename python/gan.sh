#!/bin/bash

# To run:
# bash ./gan.sh <dataset> <algo> <ano_gan_ind[0|1]> <n_epochs>
#
# =========
# Supported Datasets:
#   2, 3, 4, donut, face, toy2
# ---------
#
# =========
# Examples:
# ---------
#
# bash ./gan.sh 2 cond 0 1000
# bash ./gan.sh 3 cond 0 1000
# bash ./gan.sh 4 cond 0 1000
# bash ./gan.sh toy2 gan 0 200
# bash ./gan.sh toy2 gan 0 1000
# bash ./gan.sh toy2 cond 0 1000

ARGC=$#
if [[ "$ARGC" -gt "0" ]]; then
    DATASET=$1
    ALGO=$2
    ANO_GAN_IND=$3
    N_EPOCHS=$4
fi

ALGO_NAME="gan"

LABEL_SMOOTHING_IND=1
LABEL_SMOOTHING=""
LABEL_SMOOTHING_SIG=""
SMOOTHING_PROB=0.9
if [[ "$LABEL_SMOOTHING_IND" == "1" ]]; then
    LABEL_SMOOTHING="--label_smoothing"
    LABEL_SMOOTHING_SIG="_ls"
fi

ANO_GAN_LAMBDA=0.5
ANO_GAN=""
N_ANO_GAN_TEST=1
if [[ "$ANO_GAN_IND" == "1" ]]; then
    # ALGO_NAME="ano_${ALGO_NAME}"
    ANO_GAN="--ano_gan"
fi

INDV_LOSS_IND=0
DIST_LOSS_IND=0
INDV_LOSS=""
DIST_LOSS=""
if [[ "$INDV_LOSS_IND" == "1" ]]; then
    INDV_LOSS="--ano_gan_individual"
fi
if [[ "$DIST_LOSS_IND" == "1" ]]; then
    DIST_LOSS="--ano_gan_use_dist"
fi

COND_GAN=""
if [[ "$ALGO" == "cond" ]]; then
    ALGO_NAME="cond_${ALGO_NAME}"
    COND_GAN="--conditional"
fi

INFO_GAN=""
INFO_GAN_LAMBDA=1.0
if [[ "$ALGO" == "info" ]]; then
    ALGO_NAME="info_${ALGO_NAME}"
    INFO_GAN="--info_gan"
fi

LOG_DIR="./temp/gan"
RESULTS_NAME="${DATASET}_${ALGO_NAME}${LABEL_SMOOTHING_SIG}_${N_EPOCHS}"
RESULTS_DIR="./temp/gan/${RESULTS_NAME}"

mkdir -p ${LOG_DIR}
mkdir -p ${RESULTS_DIR}

python -m dnn.test_gan --dataset=${DATASET} ${COND_GAN} ${ANO_GAN} \
    --n_ano_gan_test=${N_ANO_GAN_TEST} ${INFO_GAN} --info_gan_lambda=${INFO_GAN_LAMBDA} \
    ${LABEL_SMOOTHING} --smoothing_prob=${SMOOTHING_PROB} \
    --ano_gan_lambda=${ANO_GAN_LAMBDA} ${INDV_LOSS} ${DIST_LOSS} \
    --n_epochs=${N_EPOCHS} --results_dir=${RESULTS_DIR} \
    --log_file="${LOG_DIR}/${RESULTS_NAME}_a${ANO_GAN_IND}.log" --debug --plot
