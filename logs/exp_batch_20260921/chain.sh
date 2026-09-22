#!/bin/bash
# usage: bash chain.sh <gpu> <jobfile> <chain-name>   (jobfile lines: "<cfg> <ds> <seq>"; failed runs are retried once)
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260921; GPU=$1; JOBS=$2; NAME=$3
while read -r cfg ds s; do
  [ -z "$cfg" ] && continue
  GPU=$GPU bash $L/run_one.sh $cfg $ds $s; rc=$?
  if [ $rc -eq 1 ]; then echo "[$(date '+%F %T')] RETRY $cfg $ds $s" >> $L/progress; GPU=$GPU bash $L/run_one.sh $cfg $ds $s || echo "[$(date '+%F %T')] SKIPPED-AFTER-RETRY $cfg $ds $s" >> $L/progress; fi
done < $JOBS
echo "[$(date '+%F %T')] CHAIN $NAME DONE" >> $L/progress
