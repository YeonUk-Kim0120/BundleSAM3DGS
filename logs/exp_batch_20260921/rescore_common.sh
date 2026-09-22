#!/bin/bash
# 3.2: YCB meshes rescored with the common seen labels (every 5th GT frame); existing JSONs untouched.
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260921; O=$L/cd_common; mkdir -p $O
B=/home/kist/Desktop/BundleSDF_baseline_outputs/full_eval_sam2_ycb/ycb
for s in bleach0 bleach_hard_00_03_chaitanya cracker_box_reorient cracker_box_yalehand0 mustard0 mustard_easy_00_02 sugar_box1 sugar_box_yalehand0 tomato_soup_can_yalehand0; do
  vd=datasets/YCBInEOAT/$s
  m=$(/usr/bin/python3 -c "import json; print(json.load(open('logs/fulleval_20260912/cd_ref_ycb_$s.json'))['mesh'])")
  nice -n 10 /usr/bin/python3 experiments/eval_mesh_cd.py --dataset ycb --video-dir $vd --run-dir $B/$s --mesh $m --keyframes-yml $L/common_seen/$s.yml --out-json $O/cd_ref_ycb_$s.json > $O/cd_ref_ycb_$s.log 2>&1 || echo "FAIL ref $s" >> $O/progress
  nice -n 10 /usr/bin/python3 experiments/eval_mesh_cd.py --dataset ycb --video-dir $vd --run-dir outputs/fulleval_20260912/ycb/$s --mesh outputs/fulleval_20260912/ycb/$s/final/gs/mesh_real_world.obj --keyframes-yml $L/common_seen/$s.yml --out-json $O/cd_ours_ycb_$s.json > $O/cd_ours_ycb_$s.log 2>&1 || echo "FAIL ours $s" >> $O/progress
  echo "done $s" >> $O/progress
done
echo ALLDONE >> $O/progress
