#!/bin/bash
# usage: bash chain_bsdfpose.sh <gpu> <wait-marker> <ds> <seq...>
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260921; GPU=$1; MARK=$2; ds=$3; shift 3
until grep -q "$MARK" $L/progress; do sleep 30; done
for s in "$@"; do GPU=$GPU bash $L/run_bsdfpose.sh $ds $s || { [ $? -eq 1 ] && GPU=$GPU bash $L/run_bsdfpose.sh $ds $s; }; done
echo "[$(date '+%F %T')] CHAIN BSDFPOSE-$ds DONE" >> $L/progress
