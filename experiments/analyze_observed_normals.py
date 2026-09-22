"""H1 check (EXP_BATCH_20260921 3.1): are the observed-lineage 2DGS discs still facing the object-frame z axis they were
born with (`_new_splat_values` sets quats = identity), or did training turn them onto the surface?

Normal n = third column of quat->R (same as GaussianRunner._splat_normals); discs are two-sided so angles use |a.b|.
Reference normal n_gt: GT mesh moved with the benchmark alignment (first online pose + ICP 2 cm, as in
analyze_map_layers.py / eval_map_quality.py); nearest of 400k GT surface samples gives the face normal; only Gaussians
within 5 mm of the GT surface are used.
  e0 = angle(z_hat, n_gt)  error the disc would have if never trained      e1 = angle(n, n_gt)  current error
  a_z = angle(n, z_hat)
usage: analyze_observed_normals.py --dataset ho3d --video-dir ... --run-dir ... --checkpoint ... --out-json ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
from eval_mesh_cd import ICP_THRES, VOXEL, load_gt_mesh, load_gt_poses, to_o3d  # noqa: E402
from gaussian_global import load_online_checkpoint, quat_wxyz_to_matrix  # noqa: E402
from prior_lifecycle import STATE_VERIFIED  # noqa: E402

BINS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 90)]


def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else None


def table(e0, e1, low):
    rows = []
    for lo, hi in BINS:
        m = (e0 >= lo) & (e0 < hi + (1e-6 if hi == 90 else 0))
        rows.append({"e0_bin": f"{lo}-{hi}", "n": int(m.sum()), "e1_median": q(e1[m], 50), "e1_p25": q(e1[m], 25), "e1_p75": q(e1[m], 75),
                     "share_opacity_lt_0.1": float(low[m].mean()) if m.any() else None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    r = load_online_checkpoint(args.checkpoint, device=args.device)
    with torch.no_grad():
        means_n = r.splats["means"].detach().cpu().numpy()
        means = r.normalization.metric_points(means_n).astype(np.float64)
        opac = torch.sigmoid(r.splats["opacities"].detach()).cpu().numpy()
        Rm = quat_wxyz_to_matrix(r.splats["quats"].detach().cpu()).numpy().astype(np.float64)
        s01 = (torch.exp(r.splats["scales"][:, :2]) / r.normalization.scale).detach().cpu().numpy()
    normals = Rm[:, :, 2]; radius_mm = s01.max(1) * 1000.0; dist_norm = np.linalg.norm(means_n, axis=1)
    lf = r.lifecycle_fields; lineage = lf.lineage.cpu().numpy(); state = lf.state.cpu().numpy()
    obs = ~lineage; prior_ver = lineage & (state == STATE_VERIFIED)
    ok_geom = (radius_mm <= 10.0) & (dist_norm <= 1.25)
    # benchmark alignment
    ids, gt_poses, K_seq, hw = load_gt_poses(args.dataset, args.video_dir)
    gt_mesh = load_gt_mesh(args.dataset, args.video_dir.name)
    first = sorted(os.listdir(args.run_dir / "ob_in_cam"))[0]
    T0 = np.linalg.inv(gt_poses[0]) @ np.loadtxt(args.run_dir / "ob_in_cam" / first).reshape(4, 4)
    if args.dataset == "ho3d":
        p1_gt = np.asarray(o3d.io.read_point_cloud(str(args.video_dir / "visible_mesh.ply")).voxel_down_sample(VOXEL).points).copy()
    else:
        p1_gt = np.asarray(trimesh.sample.sample_surface(gt_mesh, 99999, seed=0)[0], dtype=np.float64)
    good = ok_geom & ((obs & (opac >= 0.1)) | prior_ver)
    pts = means[good] @ T0[:3, :3].T + T0[:3, 3]
    pts = pts[((pts <= p1_gt.max(0) + 0.3) & (pts >= p1_gt.min(0) - 0.3)).all(1)]
    reg = o3d.pipelines.registration.registration_icp(to_o3d(pts).voxel_down_sample(VOXEL), to_o3d(p1_gt), ICP_THRES, np.eye(4),
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint())
    T = np.asarray(reg.transformation) @ T0
    samples, fidx = trimesh.sample.sample_surface(gt_mesh, 400000, seed=1)
    gt_n = np.asarray(gt_mesh.face_normals, dtype=np.float64)[fidx]
    d, nn = cKDTree(np.asarray(samples)).query(means @ T[:3, :3].T + T[:3, 3])
    n_gt = gt_n[nn]; n_g = normals @ T[:3, :3].T; z_g = np.tile(T[:3, :3] @ np.array([0.0, 0.0, 1.0]), (len(means), 1))
    near = d < 0.005
    e0 = ang(z_g, n_gt); e1 = ang(n_g, n_gt); a_z = ang(n_g, z_g); low = opac < 0.1
    out = {"checkpoint": str(args.checkpoint), "gaussians": int(len(means)), "icp_fitness": float(reg.fitness), "groups": {}}
    for name, m in (("observed_all", obs & ok_geom & near), ("observed_opacity_ge_0.1", obs & ok_geom & near & ~low),
                    ("observed_opacity_lt_0.1", obs & ok_geom & near & low), ("prior_verified", prior_ver & ok_geom & near)):
        hist, _ = np.histogram(a_z[m], bins=[0, 5, 10, 15, 30, 45, 60, 90.001])
        out["groups"][name] = {"n": int(m.sum()), "e0_median": q(e0[m], 50), "e1_median": q(e1[m], 50), "a_z_median": q(a_z[m], 50),
                               "share_a_z_lt_5deg": float((a_z[m] < 5).mean()) if m.any() else None,
                               "a_z_hist_[0,5,10,15,30,45,60,90]": hist.tolist(), "by_e0_bin": table(e0[m], e1[m], low[m])}
    big = [b for b in out["groups"]["observed_all"]["by_e0_bin"] if b["e0_bin"] in ("45-60", "60-90")]
    m = obs & ok_geom & near & (e0 > 45)
    out["e0_gt_45"] = {"n": int(m.sum()), "e1_median": q(e1[m], 50), "share_opacity_lt_0.1": float(low[m].mean()) if m.any() else None}
    args.out_json.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(args.out_json, "w"), indent=1)
    g = out["groups"]["observed_all"]
    print(f"{args.checkpoint}: observed near GT {g['n']}, a_z median {g['a_z_median']:.1f} deg (share <5 deg {g['share_a_z_lt_5deg']:.2f}), e0 median {g['e0_median']:.1f}, e1 median {g['e1_median']:.1f}; "
          f"e0>45: n {out['e0_gt_45']['n']}, e1 median {out['e0_gt_45']['e1_median']}, low-opacity share {out['e0_gt_45']['share_opacity_lt_0.1']}; prior verified e1 median {out['groups']['prior_verified']['e1_median']}")
    for b in g["by_e0_bin"]:
        print("   ", b)


if __name__ == "__main__":
    main()
