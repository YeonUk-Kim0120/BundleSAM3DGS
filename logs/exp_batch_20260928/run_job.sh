#!/bin/bash
# EXP_BATCH_20260928 one cell.  Runs INSIDE a container (N: bundlesdf, G*/MAPS: BundleSAM3DGS-gsplat-smoke).
# usage: GPU=<n> bash run_job.sh <ARM: N|G0|G0B|G1|G3> <ycb|ho3d> <seq> <S|I> <K: 20|40|80|last|-> <steps> [rep-tag]
#        GPU=<n> bash run_job.sh MAPS <ycb|ho3d> <seq> <Ks, comma-separated e.g. 40,80,last>
# rc: 0 done/skip, 1 failed (chain retries once), 3 disk stop, 4 missing input
cd /home/kist/Desktop/BundleSAM3DGS; E=exp_batch_20260928; L=logs/$E; arm=$1; ds=$2; s=$3; GPU=${GPU:-0}
export TZ=Asia/Seoul PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
inp=$L/inputs/${ds}_$s.json; mapdir=outputs/$E/g0maps/$ds/$s
if [ "$arm" = "MAPS" ]; then
  Ks=$(echo $4 | tr ',' ' '); od=$mapdir; tag=MAPS_${ds}_$s; done_file=$od/maps.json
  CMD="/usr/bin/python3 experiments/exp_gs_joint_refit.py --input $inp --mode maps --map-poses noop --Ks $Ks --out $od"
else
  mode=$4; K=$5; steps=$6; rep=$7
  if [ "$mode" = "S" ]; then cell=S_K${K}_s$steps${rep:+_$rep}; else cell=I_s$steps${rep:+_$rep}; fi
  od=outputs/$E/$arm/$ds/$s/$cell; tag=${arm}_${cell}_${ds}_$s; done_file=$od/result.json
  if [ "$arm" = "N" ]; then CMD="/usr/bin/python3 experiments/exp_nerf_round.py --input $inp --mode $mode --K $K --n-step $steps --out $od"
  else CMD="/usr/bin/python3 experiments/exp_gs_joint_refit.py --input $inp --arm $arm --mode $mode --K $K --steps $steps --maps-dir $mapdir --out $od"; fi
  if [ "$arm" = "G0" ] || [ "$arm" = "G0B" ]; then
    if [ ! -f "$mapdir/maps.json" ]; then echo "[$(date '+%F %T')] NOINPUT $tag (no $mapdir/maps.json)" >> $L/progress; exit 4; fi
  fi
fi
if [ -f "$done_file" ]; then echo "[$(date '+%F %T')] SKIP  $tag (done)" >> $L/progress; exit 0; fi
if [ ! -f "$inp" ]; then echo "[$(date '+%F %T')] NOINPUT $tag ($inp)" >> $L/progress; exit 4; fi
free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
if [ "$free" -lt 150 ]; then echo "[$(date '+%F %T')] DISKSTOP $tag free=${free}G" >> $L/progress; exit 3; fi
if [ -d "$od" ]; then mkdir -p outputs/$E/_failed; mv "$od" outputs/$E/_failed/${tag}_$(date +%s); echo "[$(date '+%F %T')] NOTE partial $od moved to _failed" >> $L/progress; fi
mkdir -p $(dirname $od)
T0=$(date '+%F %T'); echo "[$T0] START $tag gpu=$GPU container=${CONTAINER:-?}" >> $L/progress
CUDA_VISIBLE_DEVICES=$GPU $CMD > $L/runs/$tag.log 2>&1; rc=$?; T1=$(date '+%F %T')
/usr/bin/python3 - <<PY
import json, subprocess
g = lambda *a: subprocess.run(["git", "-c", "safe.directory=*", *a], capture_output=True, text=True).stdout
m = {"tag": "$tag", "arm": "$arm", "dataset": "$ds", "seq": "$s", "gpu": "$GPU", "container": "${CONTAINER:-?}", "command": """$CMD""",
     "env": {"CUDA_VISIBLE_DEVICES": "$GPU", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONDONTWRITEBYTECODE": "1", "TZ": "Asia/Seoul"},
     "input": "$inp", "output": "$od", "start": "$T0", "end": "$T1", "rc": $rc, "git_head": g("rev-parse", "HEAD").strip(),
     "git_status_main_code": g("status", "--short", "--", "gaussian_runner.py", "prior_lifecycle.py", "bundlesdf.py", "gaussian_global.py", "sam3d_prior.py", "nerf_runner.py"),
     "scripts_sha1": {p: subprocess.run(["sha1sum", p], capture_output=True, text=True).stdout.split()[0] for p in
                      ["experiments/exp_nerf_round.py", "experiments/exp_gs_joint_refit.py", "experiments/exp_round_inputs.py", "experiments/joint_refit_eval.py"]}}
json.dump(m, open("$L/manifests/$tag.json", "w"), indent=1)
PY
if [ $rc -ne 0 ] || [ ! -f "$done_file" ]; then echo "[$T1] FAIL  $tag rc=$rc" >> $L/progress; exit 1; fi
echo "[$T1] DONE  $tag $(grep -h '^\[' $L/runs/$tag.log | grep -v 'I round' | tail -3 | tr '\n' ' ')" >> $L/progress
