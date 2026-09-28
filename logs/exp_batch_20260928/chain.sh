#!/bin/bash
# usage (inside a container): CONTAINER=<name> bash chain.sh <gpu> <jobfile> <chain-name>
# jobfile lines: "<ARM> <ds> <seq> <mode|Ks> [K] [steps] [rep]"; failed runs (rc 1) are retried once; rc 3 (disk) stops the chain.
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260928; GPU=$1; JOBS=$2; NAME=$3; export TZ=Asia/Seoul
echo "[$(date '+%F %T')] CHAIN $NAME START gpu=$GPU container=${CONTAINER:-?} jobs=$JOBS" >> $L/progress
while read -r a1 a2 a3 a4 a5 a6 a7; do
  [ -z "$a1" ] && continue; case "$a1" in \#*) continue;; esac
  GPU=$GPU bash $L/run_job.sh $a1 $a2 $a3 $a4 $a5 $a6 $a7; rc=$?
  if [ $rc -eq 3 ]; then echo "[$(date '+%F %T')] CHAIN $NAME STOPPED (disk)" >> $L/progress; exit 3; fi
  if [ $rc -eq 1 ]; then echo "[$(date '+%F %T')] RETRY $a1 $a2 $a3 $a4 $a5 $a6 $a7" >> $L/progress; GPU=$GPU bash $L/run_job.sh $a1 $a2 $a3 $a4 $a5 $a6 $a7 || echo "[$(date '+%F %T')] SKIPPED-AFTER-RETRY $a1 $a2 $a3 $a4 $a5 $a6 $a7" >> $L/progress; fi
done < $JOBS
echo "[$(date '+%F %T')] CHAIN $NAME DONE" >> $L/progress
