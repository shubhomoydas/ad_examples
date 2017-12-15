#!/bin/bash
#
# Example:
# chmod +x ./tree_aad-job.sh
#
# To run:
# bash ./tree_aad-job.sh <dataset> <budget> <reruns> <tau> <inference_type> <query_type> <query_confident[0|1]> <streaming[0|1]> <streaming_window> <retention_type[0|1]>
#
# Examples:
# bash ./tree_aad-job.sh toy2 10 1 0 0.03 7 1 0 0 512 0

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

QUEUE="doppa"

if [[ "$STREAMING_IND" == "1" ]]; then
    STREAMING_SIG="_stream${STREAM_WINDOW}c${QUERY_CONFIDENT}r${RETENTION_TYPE}"
else
    STREAMING_SIG=
fi

JOBNAME="aad_${DATASET}_${BUDGET}_${RERUNS}_tau${TAU}_i${DETECTOR_TYPE}_q${QUERY_TYPE}${STREAMING_SIG}"

BASE_PATH="/data/doppa/users/sdas/temp/tree_aad"
mkdir -p $BASE_PATH

OUTFILE="$BASE_PATH/${JOBNAME}.txt"

# OPERATION="/home/sdas/codebase/pyalad/tree_aad.sh"
OPERATION="/home/sdas/codebase/bb_python/ad_examples/python/tree_aad.sh"

echo "log: $OUTFILE"
echo "cmd: $OPERATION"

rm -f $OUTFILE

qsub -N "$JOBNAME" -l walltime=148:00:00,mem=4gb,nodes=1:doppa:ppn=6 -o $OUTFILE -j oe -k oe -M shubhomoy.das@wsu.edu -m ae -v DATASET=${DATASET},BUDGET=${BUDGET},RERUNS=${RERUNS},TAU=${TAU},DETECTOR_TYPE=${DETECTOR_TYPE},QUERY_TYPE=${QUERY_TYPE},QUERY_CONFIDENT=${QUERY_CONFIDENT},STREAMING_IND=${STREAMING_IND},STREAM_WINDOW=${STREAM_WINDOW},RETENTION_TYPE=${RETENTION_TYPE} -q ${QUEUE} $OPERATION
