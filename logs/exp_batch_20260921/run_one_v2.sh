#!/bin/bash
# EXP_BATCH_20260921 runner (template: logs/exp_fusion_20260915/run_one.sh).  Official entry points through
# experiments/online_variants/launch.py (experiment copies injected; main code untouched).  Runs INSIDE the container.
# usage: GPU=<n> bash run_one.sh <cfg> <ycb|ho3d> <seq>
cd /home/kist/Desktop/BundleSAM3DGS; E=exp_batch_20260921; L=logs/$E; cfg=$1; ds=$2; s=$3; GPU=${GPU:-1}
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=$(cat $L/BASE_VARIANTS 2>/dev/null)          # "" or "normalinit" (decided after 4.1)
INIT=500; UPD=500; FB=noop; RUNNER_CFG=""; V="$BASE"
case $cfg in
  ctrl|ctrl_r2)             V="";;
  sA_r2)      RUNNER_CFG=$L/config_o05.yml;;
  sC_r2)      RUNNER_CFG=$L/config_o05.yml; INIT=0; UPD=0;;
  scaleclamp_r2) V="${BASE:+$BASE,}scaleclamp";;
  fb_ctrl_r2) FB=on;;
  fb_poselr001_r2) FB=on; RUNNER_CFG=$L/config_poselr001.yml;;
  normalinit|normalinit_r2) V="normalinit";;
  sh0)        RUNNER_CFG=$L/config_sh0.yml;;
  base_x)     ;;                                   # base re-run where needed (e.g. sequences outside the 4.1 set)
  sA)         RUNNER_CFG=$L/config_o05.yml;;
  sB)         RUNNER_CFG=$L/config_o05.yml; UPD=100;;
  sC)         RUNNER_CFG=$L/config_o05.yml; INIT=0; UPD=0;;
  scaleclamp) V="${BASE:+$BASE,}scaleclamp";;
  fb_ctrl)    FB=on;;
  fb_poselr001) FB=on; RUNNER_CFG=$L/config_poselr001.yml;;
  fb_anchor)  FB=on; V="${BASE:+$BASE,}anchor";;
  fb_anchor_poselr001) FB=on; V="${BASE:+$BASE,}anchor"; RUNNER_CFG=$L/config_poselr001.yml;;
  *) echo "unknown cfg $cfg"; exit 2;;
esac
P=/home/kist/Desktop/sam-3d-objects/output_YCBInEOAT_sam2mask_mesh
if [ "$ds" = ycb ]; then vd=datasets/YCBInEOAT/$s; KF="--keyframes-yml $L/common_seen/$s.yml"; else vd=datasets/HO3D_v3/evaluation/$s; KF=""; fi
rd=outputs/$E/$cfg/$ds/$s; tag=${cfg}_${ds}_$s
if [ -f "$rd/final/gs/mesh_real_world.obj" ]; then echo "[$(date '+%F %T')] SKIP  $cfg $ds $s (exists)" >> $L/progress; exit 0; fi
free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
if [ "$free" -lt 150 ]; then echo "[$(date '+%F %T')] DISKSTOP $cfg $ds $s free=${free}G" >> $L/progress; exit 3; fi
if [ -d "$rd" ]; then mv_to=outputs/$E/_failed/${tag}_$(date +%s); mkdir -p outputs/$E/_failed; cp -al "$rd" "$mv_to" 2>/dev/null; echo "[$(date '+%F %T')] NOTE previous partial $rd hard-linked to $mv_to" >> $L/progress; fi
if [ "$ds" = ycb ]; then
  CMD="/usr/bin/python3 experiments/online_variants/launch.py run_custom.py --mode run_video --video_dir $vd --out_folder $rd --mask_dir masks_sam2 --use_segmenter 0 --use_gui 0 --debug_level 2 --backend gaussian --gs_feedback $FB --gs_initial_steps $INIT --gs_update_steps $UPD ${RUNNER_CFG:+--gs_runner_config $RUNNER_CFG} --prior_mesh_npz $P/${s}_mesh_depth.npz --prior_pose_json $P/${s}_mesh_depth.json --prior_gaussian_ply $P/${s}_splat_depth.ply"
else
  CMD="/usr/bin/python3 experiments/online_variants/launch.py run_ho3d.py --video_dirs $vd --out_dir outputs/$E/$cfg/ho3d --mask_dir masks_SAM2 --backend gaussian --gs_feedback $FB --gs_initial_steps $INIT --gs_update_steps $UPD ${RUNNER_CFG:+--gs_runner_config $RUNNER_CFG}"
fi
T0=$(date '+%F %T')
echo "[$T0] START $cfg $ds $s gpu=$GPU variants='$V' fb=$FB init=$INIT upd=$UPD cfg=${RUNNER_CFG:-default} free=${free}G" >> $L/progress
GS_ONLINE_VARIANTS="$V" CUDA_VISIBLE_DEVICES=$GPU $CMD > $L/run_$tag.log 2>&1
rc=$?; T1=$(date '+%F %T'); echo "[$T1] END   $cfg $ds $s rc=$rc" >> $L/progress
/usr/bin/python3 - <<PY
import json, subprocess, hashlib
m = {"cfg": "$cfg", "dataset": "$ds", "seq": "$s", "gpu": "$GPU", "command": """$CMD""", "env": {"GS_ONLINE_VARIANTS": "$V", "CUDA_VISIBLE_DEVICES": "$GPU", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
     "feedback": "$FB", "initial_steps": $INIT, "update_steps": $UPD, "runner_config": "${RUNNER_CFG:-config_gs_2dgs_1mm_lifecycle.yml}",
     "runner_config_text": open("${RUNNER_CFG:-config_gs_2dgs_1mm_lifecycle.yml}").read(), "start": "$T0", "end": "$T1", "rc": $rc,
     "git_head": open(".git/HEAD").read().strip(), "copy_runner_sha1": hashlib.sha1(open("experiments/online_variants/gaussian_runner.py","rb").read()).hexdigest(),
     "copy_lifecycle_sha1": hashlib.sha1(open("experiments/online_variants/prior_lifecycle.py","rb").read()).hexdigest()}
try: m["git_commit"] = open(".git/" + m["git_head"].split(": ")[1]).read().strip()
except Exception: pass
json.dump(m, open("$L/manifests/$tag.json", "w"), indent=1)
PY
cp experiments/online_variants/gaussian_runner.py $L/manifests/${tag}_gaussian_runner_copy.py 2>/dev/null
if [ $rc -ne 0 ] || [ ! -f "$rd/final/gs/mesh_real_world.obj" ]; then echo "[$(date '+%F %T')] FAIL  $cfg $ds $s rc=$rc" >> $L/progress; exit 1; fi
export CUDA_VISIBLE_DEVICES=$GPU
GS_ONLINE_VARIANTS="$V" /usr/bin/python3 experiments/online_variants/launch.py experiments/exp_extract_online_mesh.py --run-dir $rd > $L/online0_$tag.log 2>&1 || echo "[$(date '+%F %T')] ONLINE0FAIL $cfg $ds $s" >> $L/progress
/usr/bin/python3 experiments/eval_add_ycbineoat.py --dataset $ds --video-dir $vd --run-dirs $rd --out-json $L/add_$tag.json > $L/add_$tag.log 2>&1
/usr/bin/python3 experiments/eval_mesh_cd.py --dataset $ds --video-dir $vd --run-dir $rd --mesh $rd/final/gs/mesh_real_world.obj $KF --out-json $L/cd_${tag}_global.json > $L/cd_${tag}_global.log 2>&1 || echo "[$(date '+%F %T')] CDFAIL $cfg $ds $s global" >> $L/progress
/usr/bin/python3 experiments/eval_mesh_cd.py --dataset $ds --video-dir $vd --run-dir $rd --mesh $rd/final/gs_online0/mesh_real_world.obj $KF --out-json $L/cd_${tag}_online0.json > $L/cd_${tag}_online0.log 2>&1 || echo "[$(date '+%F %T')] CDFAIL $cfg $ds $s online0" >> $L/progress
for ck in global:final/gs/checkpoint_global.pt online:gs_online/checkpoint_final.pt; do n=${ck%%:*}; f=${ck#*:}
  /usr/bin/python3 experiments/online_variants/launch.py experiments/eval_map_quality.py --dataset $ds --video-dir $vd --run-dir $rd --checkpoint $rd/$f --out-json $L/map_${tag}_$n.json --device cuda:0 > $L/map_${tag}_$n.log 2>&1 || echo "[$(date '+%F %T')] MAPFAIL $cfg $ds $s $n" >> $L/progress
  /usr/bin/python3 experiments/online_variants/launch.py experiments/analyze_observed_normals.py --dataset $ds --video-dir $vd --run-dir $rd --checkpoint $rd/$f --out-json $L/normals/${tag}_$n.json > $L/normals_${tag}_$n.log 2>&1 || echo "[$(date '+%F %T')] NORMALSFAIL $cfg $ds $s $n" >> $L/progress
done
/usr/bin/python3 experiments/online_variants/launch.py experiments/analyze_map_layers.py --dataset $ds --video-dir $vd --run-dir $rd --out-dir $L/viz/$tag --device cuda:0 > $L/layers_$tag.log 2>&1 || echo "[$(date '+%F %T')] LAYERFAIL $cfg $ds $s" >> $L/progress
echo "[$(date '+%F %T')] DONE  $cfg $ds $s $(/usr/bin/python3 - <<PY 2>&1 | tail -1
import json
L="$L"; t="$tag"
def j(p):
    try: return json.load(open(p))
    except Exception: return None
a=j(f"{L}/add_{t}.json"); a=a[0] if isinstance(a,list) else a
line="ADD %.3f" % a["ADD_err_cm"] if a else "ADD n/a"
for k in ("global","online0"):
    c=j(f"{L}/cd_{t}_{k}.json")
    if c: p=c["P3_regions"]; line+=" | %s P1 %.3f P2 %.3f seen %.3f unseen %.3f (%.0f%%)" % (k, c["P1_original"]["chamfer_cm"], c["P2_full_model"]["chamfer_cm"], p["gt_to_pred_seen_cm"], p["gt_to_pred_unseen_cm"] or float("nan"), 100*(p["unseen_within_5mm_frac"] or 0))
m=j(f"{L}/map_{t}_global.json")
if m: o=m["opacity"]["observed"]; line+=" | N %d obs<0.1 %.0f%% big %d far %d render %.2f centres->GT %.2f" % (m["gaussians"], 100*(o["lt0.1"] or 0), m["outliers"]["radius_gt_10mm"], m["outliers"]["dist_gt_1.25"], m["render_depth_residual_mm"]["median"], m["vs_gt"]["centres_to_gt_mm"]["median"])
print(line)
PY
)" >> $L/progress
