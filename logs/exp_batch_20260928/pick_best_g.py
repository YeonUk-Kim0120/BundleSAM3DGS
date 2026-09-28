"""Pre-registered (before any production result) choice of the mode-I G arm (brief 4: best G from mode S; G0/G0B excluded).
Rule: over the 11 mode-S cells at 500 steps (all K, 4 sequences), mean of the per-cell d-median reduction fraction
(median d before - median d after) / median d before; higher wins; |difference| < 0.01 -> lower mean worsened rate wins.
Writes logs/exp_batch_20260928/best_g.json, jobs/gpu1_after.txt and jobs/gpu0_after.txt (priority order of brief 6)."""
import json, sys
from pathlib import Path
import numpy as np
R = Path("/home/kist/Desktop/BundleSAM3DGS"); E = "exp_batch_20260928"; L = R / "logs" / E
sys.path.insert(0, str(R / "experiments"))
from joint_refit_eval import Evaluator, summary
SEQ = [("ho3d", "AP12", ["40", "80", "last"]), ("ycb", "mustard0", ["20", "last"]), ("ho3d", "SM1", ["40", "80", "last"]), ("ho3d", "MPM12", ["40", "80", "last"])]
score = {}
for arm in ("G1", "G3"):
    red, wor, cells = [], [], {}
    for ds, s, Ks in SEQ:
        ev = Evaluator(str(L / "inputs" / f"{ds}_{s}.json"))
        for K in Ks:
            f = R / "outputs" / E / arm / ds / s / f"S_K{K}_s500" / "result.json"
            if not f.exists(): cells[f"{s}/{K}"] = None; continue
            r = json.load(open(f))["rounds"][0]; sm = summary(ev.compare(r["kf_ids"], r["poses_in"], r["poses_out"]))
            red.append(sm["d_reduction_frac"]); wor.append(sm["wor"]); cells[f"{s}/{K}"] = sm
    score[arm] = dict(mean_reduction=float(np.mean(red)) if red else float("-inf"), mean_worsened=float(np.mean(wor)) if wor else 1.0, n_cells=len(red), cells=cells)
a, b = score["G1"], score["G3"]
if abs(a["mean_reduction"] - b["mean_reduction"]) >= 0.01: best = "G1" if a["mean_reduction"] > b["mean_reduction"] else "G3"
else: best = "G1" if a["mean_worsened"] <= b["mean_worsened"] else "G3"
second = "G3" if best == "G1" else "G1"
json.dump(dict(best=best, second=second, rule=__doc__, score=score), open(L / "best_g.json", "w"), indent=1)
S = [("ho3d", "SM1"), ("ho3d", "AP12"), ("ho3d", "MPM12"), ("ycb", "mustard0")]
cell = lambda arm, steps: [f"{arm} {ds} {s} S {K} {steps}" for ds, s, Ks in SEQ for K in Ks]
# GPU1: must-do + higher priority; GPU0 (after the N chain, gsplat container): the two lowest-priority items (brief 6 skip order), no overlap.
lines1 = [f"{best} {ds} {s} I - 500" for ds, s in S[:2]] + cell("G1", 2000) + [f"{best} ho3d MPM12 I - 500", "G1 ho3d AP12 S last 500 rep1", f"{best} ycb mustard0 I - 500"]
lines0 = cell("G3", 2000) + [f"{second} {ds} {s} I - 500" for ds, s in S]
open(L / "jobs" / "gpu1_after.txt", "w").write("\n".join(lines1) + "\n"); open(L / "jobs" / "gpu0_after.txt", "w").write("\n".join(lines0) + "\n")
print(f"best G = {best} (mean d-median reduction G1 {a['mean_reduction']:+.3f} over {a['n_cells']} cells, G3 {b['mean_reduction']:+.3f} over {b['n_cells']}; worsened G1 {a['mean_worsened']:.2f}, G3 {b['mean_worsened']:.2f})")
