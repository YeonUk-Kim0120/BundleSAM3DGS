#!/bin/bash
# 4.6: usage: GPU=<n> bash run_bsdfpose.sh <ds> <seq>   (scores with --run-dir = the BundleSDF run = alignment gauge)
cd /home/kist/Desktop/BundleSAM3DGS; L=logs/exp_batch_20260921; ds=$1; s=$2; GPU=${GPU:-1}; export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
B=/home/kist/Desktop/BundleSDF_baseline_outputs/full_eval_sam2_$ds/$ds/$s; rd=outputs/exp_batch_20260921/bsdfpose/$ds/$s; tag=bsdfpose_${ds}_$s
if [ $ds = ycb ]; then vd=datasets/YCBInEOAT/$s; KF="--keyframes-yml $L/common_seen/$s.yml"; else vd=datasets/HO3D_v3/evaluation/$s; KF=""; fi
[ -f $rd/final/gs/mesh_real_world.obj ] && exit 0
free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free" -lt 150 ] && { echo "[$(date '+%F %T')] DISKSTOP $tag" >> $L/progress; exit 3; }
[ -d $rd ] && { mkdir -p outputs/exp_batch_20260921/_failed; cp -al $rd outputs/exp_batch_20260921/_failed/${tag}_$(date +%s); /usr/bin/python3 -c "import shutil; shutil.rmtree('$rd')"; }
echo "[$(date '+%F %T')] START bsdfpose $ds $s gpu=$GPU" >> $L/progress
/usr/bin/python3 experiments/exp_replay_bsdf_poses.py --dataset $ds --seq $s --bsdf-run $B --out-dir $rd > $L/run_$tag.log 2>&1; rc=$?
echo "[$(date '+%F %T')] END   bsdfpose $ds $s rc=$rc" >> $L/progress
[ $rc -ne 0 ] && { echo "[$(date '+%F %T')] FAIL  bsdfpose $ds $s rc=$rc" >> $L/progress; exit 1; }
for k in global:gs online0:gs_online0; do n=${k%%:*}; d=${k#*:}
  /usr/bin/python3 experiments/eval_mesh_cd.py --dataset $ds --video-dir $vd --run-dir $B --mesh $rd/final/$d/mesh_real_world.obj $KF --out-json $L/cd_${tag}_$n.json > $L/cd_${tag}_$n.log 2>&1 || echo "[$(date '+%F %T')] CDFAIL $tag $n" >> $L/progress
done
echo "[$(date '+%F %T')] DONE  bsdfpose $ds $s $(/usr/bin/python3 -c "
import json
for k in ('global','online0'):
    c=json.load(open('$L/cd_${tag}_%s.json' % k)); p=c['P3_regions']; print('%s P1 %.3f P2 %.3f seen %.3f unseen %.3f (%.0f%%)' % (k, c['P1_original']['chamfer_cm'], c['P2_full_model']['chamfer_cm'], p['gt_to_pred_seen_cm'], p['gt_to_pred_unseen_cm'] or float('nan'), 100*(p['unseen_within_5mm_frac'] or 0)), end=' | ')
" 2>&1 | tail -1)" >> $L/progress
