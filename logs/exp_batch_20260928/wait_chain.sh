#!/bin/bash
# usage (inside a container): CONTAINER=<name> bash wait_chain.sh "<chain names to wait for>" <gpu> <jobfile> <chain-name> [pre-command]
# waits until every listed chain logged DONE or STOPPED, runs the optional pre-command (e.g. writes the job file), then runs chain.sh.
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260928; export TZ=Asia/Seoul
for c in $1; do until grep -q "CHAIN $c DONE\|CHAIN $c STOPPED" $L/progress 2>/dev/null; do sleep 60; done; done
if [ -n "$5" ]; then echo "[$(date '+%F %T')] PRE $4: $5" >> $L/progress; eval "$5" >> $L/progress 2>&1; fi
bash $L/chain.sh $2 $3 $4
