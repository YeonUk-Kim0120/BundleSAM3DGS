"""EXP_BATCH_20260921 4.4 (H3, double layer) — extraction-time test, no training: drop prior-lineage VERIFIED surfels
that are superseded by live observed Gaussians (>= k observed-lineage Gaussians with opacity >= 0.1, radius <= 10 mm,
normalized distance <= 1.25 within r), then the same Poisson + largest component as gaussian_global.extract_meshes.
The filter of gaussian_global.gaussian_surfels is reproduced here (main code untouched); variant 'control' re-extracts
the current mesh with this script (reproduction check).
usage: exp_extract_supersede.py --dataset ho3d --video-dir ... --run-dir ... --checkpoint ... --out-dir ... [--keyframes-yml ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(REPO))
from gaussian_global import DEFAULT_GLOBAL_CONFIG, load_online_checkpoint, poisson_mesh, quat_wxyz_to_matrix  # noqa: E402
from prior_lifecycle import STATE_CONTRADICTED, STATE_VERIFIED  # noqa: E402

VARIANTS = {"control": None, "r3k3": (0.003, 3), "r5k3": (0.005, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True); ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--keyframes-yml", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT_GLOBAL_CONFIG
    r = load_online_checkpoint(a.checkpoint, device=a.device)
    with torch.no_grad():
        means_n = r.splats["means"].detach().cpu().numpy()
        means = r.normalization.metric_points(means_n).astype(np.float64)
        opac = torch.sigmoid(r.splats["opacities"].detach()).cpu().numpy()
        normals = quat_wxyz_to_matrix(r.splats["quats"].detach().cpu()).numpy()[:, :, 2].astype(np.float64)
        colors = np.clip(r.splats["sh0"].detach().cpu().numpy()[:, 0, :] * 0.28209479177387814 + 0.5, 0, 1).astype(np.float64)
        radius_mm = (torch.exp(r.splats["scales"][:, :2]).max(1).values / r.normalization.scale * 1000.0).cpu().numpy()
    radius_norm = np.linalg.norm(means_n, axis=1)
    state = r.lifecycle_fields.state.cpu().numpy(); lineage = r.lifecycle_fields.lineage.cpu().numpy()
    keep0 = (opac > cfg["opacity_min"]) & (radius_mm <= cfg["max_scale_mm"]) & (radius_norm <= cfg["max_radius_norm"]) & (state != STATE_CONTRADICTED)
    live_obs = (~lineage) & (opac >= 0.1) & (radius_mm <= 10.0) & (radius_norm <= 1.25) & (state != STATE_CONTRADICTED)
    prior_ver = lineage & (state == STATE_VERIFIED)
    centre = -r.normalization.translation.astype(np.float64)
    tree = cKDTree(means[live_obs]) if live_obs.any() else None
    summary = {"checkpoint": str(a.checkpoint), "gaussians": int(len(means)), "poisson_input_control": int(keep0.sum()),
               "prior_verified_in_input": int((keep0 & prior_ver).sum()), "live_observed": int(live_obs.sum()), "variants": {}}
    for name, rk in VARIANTS.items():
        t0 = time.time(); keep = keep0.copy(); removed = 0
        if rk is not None and tree is not None:
            rad, k = rk
            cand = np.where(keep0 & prior_ver)[0]
            counts = np.array([len(x) for x in tree.query_ball_point(means[cand], r=rad, workers=-1)])
            drop = cand[counts >= k]; keep[drop] = False; removed = int(len(drop))
        m, n, c = means[keep], normals[keep].copy(), colors[keep]
        flip = ((m - centre) * n).sum(1) < 0; n[flip] *= -1.0; n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        mesh = poisson_mesh(m, n, c, cfg["poisson_depth"], cfg["poisson_density_quantile"])
        mesh.remove_infinite_values(); mesh.remove_unreferenced_vertices()
        parts = mesh.split(only_watertight=False); n_parts = len(parts)
        big_parts = int(sum(len(p.vertices) >= 100 for p in parts))
        mesh = max(parts, key=lambda p: len(p.vertices)) if n_parts else mesh
        d = a.out_dir / name; d.mkdir(exist_ok=True); mesh.export(d / "mesh_real_world.obj")
        cmd = [sys.executable, str(REPO / "experiments/eval_mesh_cd.py"), "--dataset", a.dataset, "--video-dir", str(a.video_dir), "--run-dir", str(a.run_dir),
               "--mesh", str(d / "mesh_real_world.obj"), "--out-json", str(d / "cd.json")] + (["--keyframes-yml", str(a.keyframes_yml)] if a.keyframes_yml else [])
        rc = subprocess.run(cmd, capture_output=True, text=True)
        cd = json.load(open(d / "cd.json")) if (d / "cd.json").exists() else None
        row = {"removed_prior_surfels": removed, "poisson_points": int(keep.sum()), "components": n_parts, "components_ge_100v": big_parts, "vertices": int(len(mesh.vertices)), "seconds": time.time() - t0}
        if cd:
            p = cd["P3_regions"]
            row.update({"P1": cd["P1_original"]["chamfer_cm"], "P2": cd["P2_full_model"]["chamfer_cm"], "pred_to_gt": cd["P2_full_model"]["pred_to_gt_cm"],
                        "gt_to_pred_seen": p["gt_to_pred_seen_cm"], "gt_to_pred_unseen": p["gt_to_pred_unseen_cm"], "unseen_within_5mm": p["unseen_within_5mm_frac"], "seen_within_5mm": p["seen_within_5mm_frac"]})
        else:
            row["cd_error"] = rc.stderr[-400:]
        summary["variants"][name] = row
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    json.dump(summary, open(a.out_dir / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
