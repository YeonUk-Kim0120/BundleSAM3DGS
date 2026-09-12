"""EXPERIMENT (2026-09-12): SuGaR-style mesh extraction for the GS map — Poisson on points sampled from RENDERED
depth/normal maps instead of Gaussian centres.  Reads a finished global checkpoint; no training; main code untouched.

Point sources (each variant -> its own Poisson mesh):
  train         : 2DGS median depth + rendered normals at the TRAINING views (inside alpha & object mask) — the
                  SuGaR recipe (visible level-set points; the 2DGS median depth is already the 0.5-transmittance point,
                  so no 3-sigma refinement is needed).  Expected: good seen surface, no completion.
  train+virtual : the same plus points from VIRTUAL cameras on a sphere, kept only where no training-view point exists
                  within ``--gap-mm`` (fills the never-observed region from the prior).  Over-sized Gaussians hidden
                  during the virtual renders as in gaussian_global.
  train+virtual_nohide : as above but nothing hidden (pure visibility filtering) — fairness check for the hand filters.
Normals: rendered normal map (camera frame -> world with R_c2w), oriented towards the camera; the frame convention is
self-checked against the surfel normals of the nearest Gaussian and reported in the manifest.
All point sets are voxel-downsampled (``--voxel-mm``) before Poisson (depth 9, density cut 5 %), largest component kept.

  python3 experiments/exp_mesh_render_poisson.py --run-dir outputs/fulleval_20260912/ho3d/AP12 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gaussian_global import (HideBigGaussians, fibonacci_sphere, largest_component, load_online_checkpoint,  # noqa: E402
                             look_at_c2w, poisson_mesh)
from gaussian_runner import GaussianRunner  # noqa: E402
from sam3d_prior import quat_wxyz_to_matrix  # noqa: E402


def render_full(runner: GaussianRunner, K: np.ndarray, c2w_metric: np.ndarray, width: int, height: int):
    """rgb (H,W,3), alpha (H,W), metric median depth (H,W), rendered normals (H,W,3, rasterizer frame)."""
    device = runner.device
    c2w_norm = torch.from_numpy(runner.normalization.normalize_c2w(c2w_metric)).float().to(device)[None]
    sh_degree = min(runner.total_steps // int(runner.config["sh_degree_interval"]), int(runner.config["sh_degree"]))
    _, _, rasterization_2dgs, _ = runner._require_gsplat()
    with torch.no_grad():
        colors, alpha, normals, _, _, median, _ = rasterization_2dgs(
            means=runner.splats["means"], quats=runner.splats["quats"], scales=torch.exp(runner.splats["scales"]),
            opacities=torch.sigmoid(runner.splats["opacities"]),
            colors=torch.cat((runner.splats["sh0"], runner.splats["shN"]), dim=1),
            viewmats=torch.linalg.inv(c2w_norm), Ks=torch.from_numpy(K.astype(np.float32)).to(device)[None],
            width=int(width), height=int(height), sh_degree=sh_degree, packed=bool(runner.config["packed"]),
            sparse_grad=False, render_mode="RGB+ED", absgrad=False, distloss=False, depth_mode="median")
        return (colors[0, ..., :3].clamp(0, 1).cpu().numpy(), alpha[0, ..., 0].cpu().numpy(),
                (median[0, ..., 0] / runner.normalization.scale).cpu().numpy(), normals[0].cpu().numpy())


def backproject(depth: np.ndarray, valid: np.ndarray, K: np.ndarray, c2w: np.ndarray, normals_r: np.ndarray, rgb: np.ndarray,
                normal_frame: str):
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs].astype(np.float64)
    pc = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
    R, t = c2w[:3, :3].astype(np.float64), c2w[:3, 3].astype(np.float64)
    pw = pc @ R.T + t
    n = normals_r[ys, xs].astype(np.float64)
    if normal_frame == "camera":
        n = n @ R.T
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    to_cam = t[None, :] - pw
    flip = (n * to_cam).sum(1) < 0            # visible surface: normal faces the camera
    n[flip] *= -1.0
    return pw, n, rgb[ys, xs].astype(np.float64)


def surfel_normals(runner: GaussianRunner):
    with torch.no_grad():
        means = runner.normalization.metric_points(runner.splats["means"].detach().cpu().numpy()).astype(np.float64)
        R = quat_wxyz_to_matrix(runner.splats["quats"].detach().cpu()).numpy()
    return means, R[:, :, 2].astype(np.float64)


def check_normal_frame(runner: GaussianRunner, view, c2w: np.ndarray, alpha_thr: float):
    """Which frame are the rasterizer's normals in?  Compare against the nearest surfel's normal (|cos|)."""
    rgb, alpha, depth, nr = render_full(runner, view.K.numpy(), c2w, view.width, view.height)
    valid = (alpha > alpha_thr) & view.mask.numpy()
    means, sn = surfel_normals(runner)
    tree = cKDTree(means)
    scores = {}
    for frame in ("camera", "world"):
        pw, n, _ = backproject(depth, valid, view.K.numpy(), c2w, nr, rgb, frame)
        if len(pw) > 5000:
            sel = np.random.default_rng(0).choice(len(pw), 5000, replace=False); pw, n = pw[sel], n[sel]
        d, idx = tree.query(pw)
        ok = d < 0.005
        scores[frame] = float(np.mean(np.abs((n[ok] * sn[idx[ok]]).sum(1)))) if ok.any() else 0.0
    best = max(scores, key=scores.get)
    return best, scores


def downsample(pts, normals, colors, voxel):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts); pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    pcd = pcd.voxel_down_sample(voxel)
    n = np.asarray(pcd.normals); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return np.asarray(pcd.points), n, np.asarray(pcd.colors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None, help="default <run-dir>/final/gs/checkpoint_global.pt")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <run-dir>/final/gs_renderpoisson")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha-threshold", type=float, default=None)
    ap.add_argument("--voxel-mm", type=float, default=2.0)
    ap.add_argument("--gap-mm", type=float, default=4.0, help="virtual-view points kept only this far from any training-view point")
    ap.add_argument("--virtual-views", type=int, default=120)
    ap.add_argument("--virtual-radius-factor", type=float, default=2.5)
    ap.add_argument("--max-scale-mm", type=float, default=10.0)
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--poisson-density-quantile", type=float, default=0.05)
    args = ap.parse_args()
    run_dir = args.run_dir
    ckpt = args.checkpoint or run_dir / "final" / "gs" / "checkpoint_global.pt"
    out_dir = args.out_dir or run_dir / "final" / "gs_renderpoisson"
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    runner = load_online_checkpoint(ckpt, device=args.device)
    alpha_thr = float(args.alpha_threshold if args.alpha_threshold is not None else runner.config["depth_alpha_threshold"])
    manifest = {"checkpoint": str(ckpt), "views": len(runner.views), "gaussians": runner.num_gaussians,
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}

    # normal-frame self check on the view whose rotation is furthest from the identity (view 0 is the identity, which
    # cannot distinguish the two conventions)
    rots = [runner.normalization.metric_c2w(v.c2w_normalized.numpy()).astype(np.float64) for v in runner.views]
    i0 = int(np.argmax([np.degrees(np.arccos(np.clip((np.trace(c[:3, :3]) - 1) / 2, -1, 1))) for c in rots]))
    v0, c2w0 = runner.views[i0], rots[i0]
    normal_frame, scores = check_normal_frame(runner, v0, c2w0, alpha_thr)
    manifest["normal_frame"] = {"chosen": normal_frame, "abs_cos_vs_surfel": scores}
    print(f"[rp] rendered-normal frame: {normal_frame} (|cos| vs surfel normals {scores})", flush=True)

    # 1. training-view points
    P, N, C = [], [], []
    for view in runner.views:
        c2w = runner.normalization.metric_c2w(view.c2w_normalized.numpy()).astype(np.float64)
        rgb, alpha, depth, nr = render_full(runner, view.K.numpy(), c2w, view.width, view.height)
        valid = (alpha > alpha_thr) & view.mask.numpy() & (depth > 0.05)
        pw, n, c = backproject(depth, valid, view.K.numpy(), c2w, nr, rgb, normal_frame)
        P.append(pw); N.append(n); C.append(c)
    P, N, C = np.concatenate(P), np.concatenate(N), np.concatenate(C)
    manifest["train_points_raw"] = int(len(P))
    Pt, Nt, Ct = downsample(P, N, C, args.voxel_mm / 1000.0)
    manifest["train_points"] = int(len(Pt))
    print(f"[rp] training views: {len(P)} raw -> {len(Pt)} points ({time.time() - t0:.0f}s)", flush=True)

    # 2. virtual-view points (gap fill), with and without hiding over-sized Gaussians
    centre = -runner.normalization.translation.astype(np.float64)
    radius_obj = 1.0 / (runner.normalization.scale * 1.2)
    dist = args.virtual_radius_factor * radius_obj
    K_seq = np.loadtxt(run_dir / "cam_K.txt").reshape(3, 3)
    W, H = 640, 480
    tree_t = cKDTree(Pt)

    def virtual_points():
        Pv, Nv, Cv = [], [], []
        for d in fibonacci_sphere(args.virtual_views):
            c2w = look_at_c2w(centre + dist * d, centre)
            rgb, alpha, depth, nr = render_full(runner, K_seq, c2w, W, H)
            valid = (alpha > alpha_thr) & (depth > 0.05)
            pw, n, c = backproject(depth, valid, K_seq, c2w, nr, rgb, normal_frame)
            Pv.append(pw); Nv.append(n); Cv.append(c)
        Pv, Nv, Cv = np.concatenate(Pv), np.concatenate(Nv), np.concatenate(Cv)
        Pv, Nv, Cv = downsample(Pv, Nv, Cv, args.voxel_mm / 1000.0)
        gap = tree_t.query(Pv)[0] > args.gap_mm / 1000.0
        return Pv[gap], Nv[gap], Cv[gap], int(len(Pv))

    variants = {}
    with HideBigGaussians(runner, args.max_scale_mm) as hidden:
        Pv, Nv, Cv, n_all = virtual_points()
    manifest["hidden_big_gaussians"] = hidden.count
    manifest["virtual_points"] = {"all": n_all, "gap_fill": int(len(Pv))}
    variants["train"] = (Pt, Nt, Ct)
    variants["train+virtual"] = (np.concatenate([Pt, Pv]), np.concatenate([Nt, Nv]), np.concatenate([Ct, Cv]))
    Pv2, Nv2, Cv2, n_all2 = virtual_points()
    manifest["virtual_points_nohide"] = {"all": n_all2, "gap_fill": int(len(Pv2))}
    variants["train+virtual_nohide"] = (np.concatenate([Pt, Pv2]), np.concatenate([Nt, Nv2]), np.concatenate([Ct, Cv2]))
    print(f"[rp] virtual gap-fill points: hide {len(Pv)} / nohide {len(Pv2)} ({time.time() - t0:.0f}s)", flush=True)

    for name, (p, n, c) in variants.items():
        t1 = time.time()
        trimesh.PointCloud(p, colors=(np.clip(c, 0, 1) * 255).astype(np.uint8)).export(out_dir / f"points_{name}.ply")
        mesh = poisson_mesh(p, n, c, args.poisson_depth, args.poisson_density_quantile)
        raw = len(mesh.vertices)
        mesh = largest_component(mesh)
        mesh.export(out_dir / f"mesh_{name}.obj")
        manifest[name] = {"points": int(len(p)), "vertices_raw": raw, "vertices": len(mesh.vertices), "seconds": time.time() - t1}
        print(f"[rp] {name}: {manifest[name]}", flush=True)
    manifest["seconds_total"] = time.time() - t0
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=1)
    print(f"[rp] done in {manifest['seconds_total']:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
