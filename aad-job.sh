#!/bin/bash
#
# Example:
# chmod +x ./aad-job.sh
#
# To run:
# bash ./aad-job.sh <dataset> <budget> <reruns> <tau> <inference_type> <query_type> <query_confident[0|1]> <streaming[0|1]> <streaming_window> <retention_type[0|1]> <with_prior[0|1]> <init_type[0|1|2]>
#
# Examples:
# bash ./aad-job.sh toy2 10 1 0 0.03 7 1 0 0 512 0 1 1

DATASET=$1
BUDGET=$2
RERUNS=$3
TAU=$4
DETECTOR_TYPE=$5
QUERY_TYPE=$6
QUERY_CONFIDENT=$7
STREAMING_IND=$8
STREAM_WINDOW=$9
RETENTION_TYPE=${10}
WITH_PRIOR_IND=${11}
INIT_TYPE=${12}

QUEUE="doppa"

if [[ "$STREAMING_IND" == "1" ]]; then
    STREAMING_SIG="_stream${STREAM_WINDOW}c${QUERY_CONFIDENT}r${RETENTION_TYPE}"
else
    STREAMING_SIG=
fi

JOBNAME="ad_${DATASET}_b${BUDGET}n${RERUNS}tau${TAU}i${DETECTOR_TYPE}q${QUERY_TYPE}p${WITH_PRIOR_IND}${INIT_TYPE}${STREAMING_SIG}"

BASE_PATH="/data/doppa/users/sdas/temp/aad"
mkdir -p $BASE_PATH

OUTFILE="$BASE_PATH/${JOBNAME}.txt"

# OPERATION="/home/sdas/codebase/pyalad/tree_aad.sh"
OPERATION="/home/sdas/codebase/bb_python/ad_examples/python/aad.sh"

echo "log: $OUTFILE"
echo "cmd: $OPERATION"

rm -f $OUTFILE

qsub -N "$JOBNAME" -l walltime=36:00:00,mem=4gb,nodes=1:doppa:ppn=4 -o $OUTFILE -j oe -k oe -M shubhomoy.das@wsu.edu -m ae -v DATASET=${DATASET},BUDGET=${BUDGET},RERUNS=${RERUNS},TAU=${TAU},DETECTOR_TYPE=${DETECTOR_TYPE},QUERY_TYPE=${QUERY_TYPE},QUERY_CONFIDENT=${QUERY_CONFIDENT},STREAMING_IND=${STREAMING_IND},STREAM_WINDOW=${STREAM_WINDOW},RETENTION_TYPE=${RETENTION_TYPE},WITH_PRIOR_IND=${WITH_PRIOR_IND},INIT_TYPE=${INIT_TYPE} -q ${QUEUE} $OPERATION
