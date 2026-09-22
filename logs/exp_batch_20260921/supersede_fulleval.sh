#!/bin/bash
# 4.4 on the 22 fulleval_20260912 global checkpoints (CPU, nice).  Waits for the 3.2 rescoring to finish first.
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260921; O=$L/supersede; mkdir -p $O
until grep -q ALLDONE $L/cd_common/progress 2>/dev/null; do sleep 60; done
for ds in ho3d ycb; do for d in outputs/fulleval_20260912/$ds/*/; do s=$(basename $d)
  [ -f $O/fulleval_$s/summary.json ] && continue
  if [ $ds = ycb ]; then vd=datasets/YCBInEOAT/$s; KF="--keyframes-yml $L/common_seen/$s.yml"; else vd=datasets/HO3D_v3/evaluation/$s; KF=""; fi
  nice -n 10 /usr/bin/python3 experiments/exp_extract_supersede.py --dataset $ds --video-dir $vd --run-dir outputs/fulleval_20260912/$ds/$s --checkpoint outputs/fulleval_20260912/$ds/$s/final/gs/checkpoint_global.pt --out-dir $O/fulleval_$s $KF > $O/fulleval_$s.log 2>&1 || echo "FAIL $s" >> $O/progress
  echo "done $s" >> $O/progress
done; done; echo ALLDONE >> $O/progress
