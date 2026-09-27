#!/bin/bash
# usage: bash wait_chain.sh "<progress pattern to wait for>" <gpu> <jobfile> <name>
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260927
while ! grep -q "$1" $L/progress 2>/dev/null; do sleep 60; done
bash $L/chain.sh $2 $3 $4
