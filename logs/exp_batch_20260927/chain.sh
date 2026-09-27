#!/bin/bash
# usage: bash chain.sh <gpu> <jobfile> <chain-name>   (jobfile lines: "<script> <args...>"; failed runs (rc 1) are retried once)
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260927; GPU=$1; JOBS=$2; NAME=$3
while read -r script a1 a2 a3 a4; do
  [ -z "$script" ] && continue
  GPU=$GPU bash $L/$script $a1 $a2 $a3 $a4; rc=$?
  if [ $rc -eq 1 ]; then echo "[$(date '+%F %T')] RETRY $script $a1 $a2 $a3 $a4" >> $L/progress; GPU=$GPU bash $L/$script $a1 $a2 $a3 $a4 || echo "[$(date '+%F %T')] SKIPPED-AFTER-RETRY $script $a1 $a2 $a3 $a4" >> $L/progress; fi
done < $JOBS
echo "[$(date '+%F %T')] CHAIN $NAME DONE" >> $L/progress
