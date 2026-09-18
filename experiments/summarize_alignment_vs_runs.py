"""(a) Does the initial prior-alignment error predict the run's outcome?  (b) Why does the loss-selected 'best' pose miss?
(2026-09-16, user request).  Inputs: exp_prior_align_multiframe outputs (1-frame runs in --align-dirs, traces in
--trace-dir) and the adopted-run scores (ADD / CD / map JSONs in logs/exp_fusion_20260915, cfg 'adopt').
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

L = Path("logs/exp_fusion_20260915")
HO = {"AP10", "AP11", "AP12", "AP13", "AP14", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "SM1"}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--trace-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for d in args.align_dirs:
        for f in sorted(glob.glob(str(d / "*.json"))):
            a = json.load(open(f)); s = a["seq"]; ds = "ho3d" if s in HO else "ycb"
            r1 = a["runs"].get("1")
            if not r1: continue
            cd = load(L / f"cd_adopt_{ds}_{s}_real_world.json"); ad = load(L / f"add_adopt_{ds}_{s}.json"); m = load(L / f"map_adopt_{ds}_{s}.json")
            if not (cd and ad): continue
            p3 = cd["P3_regions"]; r = ad[0] if isinstance(ad, list) else ad
            rows.append({"seq": s, "dataset": ds, "align_init_mm": a["init_error"]["mesh_disp_mm"], "align_best_mm": r1["best"]["mesh_disp_mm"],
                         "align_final_mm": r1["curve"][-1]["mesh_disp_mm"], "align_rot_deg": r1["best"]["rot_deg"], "align_scale": r1["best"]["scale_ratio"],
                         "ADD": r["ADD_err_cm"], "P1": cd["P1_original"]["chamfer_cm"], "P2": cd["P2_full_model"]["chamfer_cm"], "unseen_cov": 100 * p3["unseen_within_5mm_frac"],
                         "centres_to_gt_mm": m["vs_gt"]["centres_to_gt_mm"]["median"] if m else np.nan})
    rows.sort(key=lambda x: x["align_best_mm"])
    lines = ["| seq | dataset | align init mm | align best mm | rot deg | scale | ADD | P1 | P2 | unseen % | centres→GT mm |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x['seq']} | {x['dataset']} | {x['align_init_mm']:.1f} | {x['align_best_mm']:.1f} | {x['align_rot_deg']:.1f} | {x['align_scale']:.2f} | {x['ADD']:.3f} | {x['P1']:.3f} | {x['P2']:.3f} | {x['unseen_cov']:.0f} | {x['centres_to_gt_mm']:.2f} |")
    def corr(sub, key):
        a = np.array([x["align_best_mm"] for x in sub]); b = np.array([x[key] for x in sub]); return float(spearmanr(a, b).correlation)
    lines.append("")
    for name, sub in (("all", rows), ("ho3d", [x for x in rows if x["dataset"] == "ho3d"]), ("ycb", [x for x in rows if x["dataset"] == "ycb"])):
        if len(sub) > 3:
            lines.append(f"Spearman(align best mm, ·) over {name} (n={len(sub)}): ADD {corr(sub,'ADD'):.2f}, P1 {corr(sub,'P1'):.2f}, P2 {corr(sub,'P2'):.2f}, unseen cov {corr(sub,'unseen_cov'):.2f}, centres→GT {corr(sub,'centres_to_gt_mm'):.2f}")
    # traces: which loss part makes the early 'best'
    if args.trace_dir and args.trace_dir.exists():
        lines.append("\n### best-selection traces (per step: total loss parts vs mesh displacement)\n")
        for f in sorted(glob.glob(str(args.trace_dir / "*.json"))):
            a = json.load(open(f)); r1 = a["runs"]["1"]; c = r1["curve"]
            if not c or "parts" not in c[0]: continue
            keys = [k for k in c[0]["parts"] if k not in ("visible_ratio", "depth_valid_ratio")]
            best_i = int(np.argmin([x["loss"] for x in c])); fin = c[-1]; b = c[best_i]
            errs = np.array([x["mesh_disp_mm"] for x in c]); i_min_err = int(np.argmin(errs))
            lines.append(f"**{a['seq']}**: loss-min at step {c[best_i]['step']} (disp {b['mesh_disp_mm']:.1f} mm) vs final step {fin['step']} (disp {fin['mesh_disp_mm']:.1f}) vs error-min at step {c[i_min_err]['step']} (disp {errs[i_min_err]:.1f}); "
                         + "loss parts at loss-min / final: " + ", ".join(f"{k} {b['parts'].get(k, float('nan')):.4f} / {fin['parts'].get(k, float('nan')):.4f}" for k in keys)
                         + f"; visible_ratio {b['parts'].get('visible_ratio', float('nan')):.2f} / {fin['parts'].get('visible_ratio', float('nan')):.2f}")
            # correlation of each part with the error along the trace
            lines.append("   Spearman(part, disp) along the trace: " + ", ".join(f"{k} {spearmanr([x['parts'].get(k, np.nan) for x in c], errs).correlation:.2f}" for k in keys))
    open(args.out, "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
