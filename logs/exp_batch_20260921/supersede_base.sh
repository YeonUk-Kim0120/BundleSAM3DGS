#!/bin/bash
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260921; O=$L/supersede
for spec in "ycb mustard0" "ycb cracker_box_yalehand0" "ho3d AP12" "ho3d MPM12"; do set -- $spec; ds=$1; s=$2
  if [ $ds = ycb ]; then vd=datasets/YCBInEOAT/$s; KF="--keyframes-yml $L/common_seen/$s.yml"; else vd=datasets/HO3D_v3/evaluation/$s; KF=""; fi
  rd=outputs/exp_batch_20260921/ctrl/$ds/$s
  nice -n 10 /usr/bin/python3 experiments/online_variants/launch.py experiments/exp_extract_supersede.py --dataset $ds --video-dir $vd --run-dir $rd --checkpoint $rd/final/gs/checkpoint_global.pt --out-dir $O/base_$s $KF > $O/base_$s.log 2>&1 || echo "FAIL base $s" >> $O/progress
  echo "done base $s" >> $O/progress
done; echo BASEDONE >> $O/progress
