#!/bin/bash
# EXP_BATCH_20260927 noop online run (template: logs/exp_batch_20260921/run_one.sh).  MAIN CODE entry points (run_ho3d.py / run_custom.py),
# --gs_feedback noop, 500/500, default runner config (config_gs_2dgs_1mm_lifecycle.yml) = batch-1 ctrl settings.  Runs INSIDE the container.
# usage: GPU=<n> bash run_noop.sh <ycb|ho3d> <seq>
cd /home/kist/Desktop/BundleSAM3DGS; E=exp_batch_20260927; L=logs/$E; cfg=noop; ds=$1; s=$2; GPU=${GPU:-0}
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INIT=500; UPD=500; FB=noop
P=/home/kist/Desktop/sam-3d-objects/output_YCBInEOAT_sam2mask_mesh
if [ "$ds" = ycb ]; then vd=datasets/YCBInEOAT/$s; else vd=datasets/HO3D_v3/evaluation/$s; fi
rd=outputs/$E/$cfg/$ds/$s; tag=${cfg}_${ds}_$s
if [ -f "$rd/final/gs/mesh_real_world.obj" ]; then echo "[$(date '+%F %T')] SKIP  $cfg $ds $s (exists)" >> $L/progress; exit 0; fi
free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
if [ "$free" -lt 150 ]; then echo "[$(date '+%F %T')] DISKSTOP $cfg $ds $s free=${free}G" >> $L/progress; exit 3; fi
if [ -d "$rd" ]; then mv_to=outputs/$E/_failed/${tag}_$(date +%s); mkdir -p outputs/$E/_failed; mv "$rd" "$mv_to"; echo "[$(date '+%F %T')] NOTE previous partial $rd moved to $mv_to" >> $L/progress; fi
if [ "$ds" = ycb ]; then
  CMD="/usr/bin/python3 run_custom.py --mode run_video --video_dir $vd --out_folder $rd --mask_dir masks_sam2 --use_segmenter 0 --use_gui 0 --debug_level 2 --backend gaussian --gs_feedback $FB --gs_initial_steps $INIT --gs_update_steps $UPD --prior_mesh_npz $P/${s}_mesh_depth.npz --prior_pose_json $P/${s}_mesh_depth.json --prior_gaussian_ply $P/${s}_splat_depth.ply"
else
  CMD="/usr/bin/python3 run_ho3d.py --video_dirs $vd --out_dir outputs/$E/$cfg/ho3d --mask_dir masks_SAM2 --backend gaussian --gs_feedback $FB --gs_initial_steps $INIT --gs_update_steps $UPD"
fi
T0=$(date '+%F %T')
echo "[$T0] START $cfg $ds $s gpu=$GPU fb=$FB init=$INIT upd=$UPD free=${free}G" >> $L/progress
CUDA_VISIBLE_DEVICES=$GPU $CMD > $L/run_$tag.log 2>&1
rc=$?; T1=$(date '+%F %T'); echo "[$T1] END   $cfg $ds $s rc=$rc" >> $L/progress
/usr/bin/python3 - <<PY
import json, subprocess
m = {"cfg": "$cfg", "dataset": "$ds", "seq": "$s", "gpu": "$GPU", "command": """$CMD""", "env": {"CUDA_VISIBLE_DEVICES": "$GPU", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
     "feedback": "$FB", "initial_steps": $INIT, "update_steps": $UPD, "runner_config": "config_gs_2dgs_1mm_lifecycle.yml (default)",
     "runner_config_text": open("config_gs_2dgs_1mm_lifecycle.yml").read(), "start": "$T0", "end": "$T1", "rc": $rc,
     "git_head": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
     "git_status_main_code": subprocess.run(["git","status","--short","--","gaussian_runner.py","prior_lifecycle.py","bundlesdf.py","run_ho3d.py","run_custom.py","gaussian_global.py","sam3d_prior.py"],capture_output=True,text=True).stdout,
     "note": "pose source for EXP_BATCH_20260927 (batch-1 ctrl outputs lack color/depth/mask/per-frame poses_before_gs after cleanup)"}
json.dump(m, open("$L/manifests/$tag.json", "w"), indent=1)
PY
if [ $rc -ne 0 ] || [ ! -f "$rd/final/gs/mesh_real_world.obj" ]; then echo "[$(date '+%F %T')] FAIL  $cfg $ds $s rc=$rc" >> $L/progress; exit 1; fi
/usr/bin/python3 experiments/eval_add_ycbineoat.py --dataset $ds --video-dir $vd --run-dirs $rd --out-json $L/add_$tag.json > $L/add_$tag.log 2>&1
echo "[$(date '+%F %T')] DONE  $cfg $ds $s $(/usr/bin/python3 -c "
import json
try:
    a=json.load(open('$L/add_$tag.json')); a=a[0] if isinstance(a,list) else a; print('ADD %.3f ADD-S %.3f' % (a['ADD_err_cm'], a['ADDS_err_cm']))
except Exception as e: print('ADD n/a', e)")" >> $L/progress
