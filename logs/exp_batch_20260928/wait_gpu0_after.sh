#!/bin/bash
# GPU0 follow-up (inside the gsplat container): waits for the N chain on GPU0 to finish and for gpu1_after to have started
# (= pick_best_g.py has written jobs/gpu0_after.txt), then runs the low-priority G jobs on GPU0.
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260928; export TZ=Asia/Seoul
until grep -q "CHAIN gpu0_N DONE\|CHAIN gpu0_N STOPPED" $L/progress && grep -q "CHAIN gpu1_after START" $L/progress; do sleep 60; done
bash $L/chain.sh 0 $L/jobs/gpu0_after.txt gpu0_after
