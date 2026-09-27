#!/bin/bash
# EXP_BATCH_20260927 stage-1 probe (experiments/exp_reference_optimizer_probe.py) on a noop run.  Runs INSIDE the container.
# usage: GPU=<n> bash run_probe.sh <REF: R0|A1|A2|A3|RP|RG> <ycb|ho3d> <seq>
cd /home/kist/Desktop/BundleSAM3DGS; E=exp_batch_20260927; L=logs/$E; ref=$1; ds=$2; s=$3; GPU=${GPU:-0}
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rd=outputs/$E/noop/$ds/$s; od=outputs/$E/probe_$ref/$ds/$s; tag=probe_${ref}_${ds}_$s
if [ -f "$od/done.json" ]; then echo "[$(date '+%F %T')] SKIP  $tag (done)" >> $L/progress; exit 0; fi
if [ ! -f "$rd/final/gs/mesh_real_world.obj" ]; then echo "[$(date '+%F %T')] NOINPUT $tag (noop run $rd not finished)" >> $L/progress; exit 1; fi
free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
if [ "$free" -lt 150 ]; then echo "[$(date '+%F %T')] DISKSTOP $tag free=${free}G" >> $L/progress; exit 3; fi
if [ -d "$od" ]; then mkdir -p outputs/$E/_failed; mv "$od" outputs/$E/_failed/${tag}_$(date +%s); echo "[$(date '+%F %T')] NOTE partial $od moved to _failed" >> $L/progress; fi
mkdir -p $(dirname $od)
CMD="/usr/bin/python3 experiments/exp_reference_optimizer_probe.py --dataset $ds --seq $s --run-dir $rd --reference $ref --output-dir $od"
T0=$(date '+%F %T'); echo "[$T0] START $tag gpu=$GPU" >> $L/progress
CUDA_VISIBLE_DEVICES=$GPU $CMD > $L/$tag.log 2>&1; rc=$?; T1=$(date '+%F %T')
/usr/bin/python3 - <<PY
import json, subprocess
m = {"tag": "$tag", "reference": "$ref", "dataset": "$ds", "seq": "$s", "gpu": "$GPU", "command": """$CMD""", "env": {"CUDA_VISIBLE_DEVICES": "$GPU", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
     "noop_run": "$rd", "start": "$T0", "end": "$T1", "rc": $rc, "git_head": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
     "git_status_main_code": subprocess.run(["git","status","--short","--","gaussian_runner.py","prior_lifecycle.py","bundlesdf.py","gaussian_global.py","sam3d_prior.py"],capture_output=True,text=True).stdout,
     "scripts_sha1": {p: subprocess.run(["sha1sum",p],capture_output=True,text=True).stdout.split()[0] for p in ["experiments/exp_reference_optimizer_probe.py","experiments/hygiene_runner.py","experiments/exp_feedback_gradient_probe_gtmap.py"]}}
json.dump(m, open("$L/manifests/$tag.json", "w"), indent=1)
PY
if [ $rc -ne 0 ] || [ ! -f "$od/done.json" ]; then echo "[$T1] FAIL  $tag rc=$rc" >> $L/progress; exit 1; fi
echo "[$T1] DONE  $tag $(/usr/bin/python3 -c "import json; d=json.load(open('$od/done.json')); print('probes', d['probes'], 'cycles', d['probe_cycles'], '%.0fs' % d['seconds'])")" >> $L/progress
