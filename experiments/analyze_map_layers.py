"""Map-layer analysis and outlier visualisation of a Gaussian checkpoint (B-track discussion, 2026-09-14).

For one run it writes, in ``--out-dir``:
  layers as coloured OBJ (``v x y z r g b``, MeshLab reads the colours) and PLY point clouds in the metric map frame:
    prior_verified.obj (green)  prior_unseen.obj (dark green)  prior_suspect.obj (yellow)
    observed_normal.obj (grey)  observed_low_opacity.obj (blue, opacity < 0.1)  far.obj (magenta, normalized
    distance > 1.25)  big_radius_discs.obj (red ellipses, in-plane radius > 10 mm, any lineage)
    observed_single_view.obj (orange, observed lineage supported in <= 1 keyframe incl. its birth view)
    all_gaussians_colored.obj (one file, precedence far > big > single-view > low-opacity > normal / prior)
    gt_aligned.obj (GT model moved into the map frame with the benchmark alignment: first online pose + ICP 2 cm)
  analysis.json / README.txt:
    offsets of observed Gaussians from the nearest prior surfel split into the component along the prior normal
    (front/back layer) and the tangential component (sideways); support-count histogram of the observed lineage;
    distance to the GT surface per category; rendering with and without the opacity < 0.1 Gaussians.
usage: analyze_map_layers.py --dataset ho3d --video-dir ... --run-dir ... --out-dir ... [--checkpoint ...]
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
from gaussian_global import load_online_checkpoint, quat_wxyz_to_matrix, render_depth  # noqa: E402
from prior_lifecycle import STATE_CONTRADICTED, STATE_SUSPECT, STATE_UNSEEN, STATE_VERIFIED  # noqa: E402

COLORS = {"prior_verified": (0, 170, 0), "prior_unseen": (0, 90, 0), "prior_suspect": (220, 200, 0),
          "observed_normal": (150, 150, 150), "observed_low_opacity": (40, 90, 255), "far": (255, 0, 255),
          "big_radius": (255, 30, 30), "observed_single_view": (255, 140, 0)}


def write_obj_points(path: Path, pts: np.ndarray, rgb) -> None:
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    if rgb.ndim == 1:
        rgb = np.repeat(rgb[None, :], len(pts), 0)
    with open(path, "w") as f:
        f.write(f"# {len(pts)} points, vertex colours\n")
        for p, c in zip(pts, rgb):
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
    if len(pts):
        trimesh.PointCloud(pts, colors=np.c_[(rgb * 255).astype(np.uint8), np.full(len(pts), 255, np.uint8)]).export(path.with_suffix(".ply"))


def write_obj_discs(path: Path, centres: np.ndarray, R: np.ndarray, s01: np.ndarray, rgb, segments: int = 12) -> None:
    """Ellipses of the 2DGS in-plane axes (axis 0/1 of R scaled by the metric scales)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    with open(path, "w") as f:
        f.write(f"# {len(centres)} discs\n")
        for i in range(len(centres)):
            ring = centres[i][None, :] + np.cos(ang)[:, None] * (R[i][:, 0] * s01[i, 0])[None, :] + np.sin(ang)[:, None] * (R[i][:, 1] * s01[i, 1])[None, :]
            f.write(f"v {centres[i][0]:.6f} {centres[i][1]:.6f} {centres[i][2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
            for p in ring:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
            base = i * (segments + 1) + 1
            for k in range(segments):
                f.write(f"f {base} {base + 1 + k} {base + 1 + (k + 1) % segments}\n")


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "mean": float(x.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--views", type=int, default=12)
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint or args.run_dir / "final" / "gs" / "checkpoint_global.pt"
    r = load_online_checkpoint(ckpt, device=args.device)
    with torch.no_grad():
        means_n = r.splats["means"].detach().cpu().numpy()
        means = r.normalization.metric_points(means_n).astype(np.float64)
        opac = torch.sigmoid(r.splats["opacities"].detach()).cpu().numpy()
        R = quat_wxyz_to_matrix(r.splats["quats"].detach().cpu()).numpy().astype(np.float64)
        s01 = (torch.exp(r.splats["scales"][:, :2]) / r.normalization.scale).detach().cpu().numpy().astype(np.float64)
    radius_mm = s01.max(1) * 1000.0
    dist_norm = np.linalg.norm(means_n, axis=1)
    lf = r.lifecycle_fields
    lineage = lf.lineage.cpu().numpy(); state = lf.state.cpu().numpy(); support_n = lf.residual_count.cpu().numpy()
    prior, obs = lineage, ~lineage
    low, big, far = opac < 0.1, radius_mm > 10.0, dist_norm > 1.25
    single = obs & (support_n <= 1)
    normal = obs & ~low & ~big & ~far
    cats = {"prior_verified": prior & (state == STATE_VERIFIED), "prior_unseen": prior & (state == STATE_UNSEEN),
            "prior_suspect": prior & (state == STATE_SUSPECT), "observed_normal": normal, "observed_low_opacity": obs & low,
            "far": far, "big_radius": big, "observed_single_view": single}
    res = {"checkpoint": str(ckpt), "gaussians": int(len(means)), "counts": {k: int(v.sum()) for k, v in cats.items()},
           "big_radius_prior_lineage": int((big & prior).sum()), "far_observed_lineage": int((far & obs).sum()),
           "overlaps": {"far&low": int((far & low).sum()), "far&single": int((far & single).sum()), "single&low": int((single & low).sum()),
                        "big&low": int((big & low).sum())},
           "support_count_observed": {f"{k}": int(((support_n == k) & obs).sum()) for k in range(0, 6)} | {">=6": int(((support_n >= 6) & obs).sum())}}

    # 1. offsets of observed Gaussians (not far) from the nearest prior surfel: along the prior normal vs tangential
    P = means[prior]; nP = R[prior][:, :, 2]
    sel = obs & ~far
    d_idx = cKDTree(P).query(means[sel])
    dvec = means[sel] - P[d_idx[1]]; n = nP[d_idx[1]]
    comp_n = np.abs((dvec * n).sum(1)); comp_t = np.linalg.norm(dvec - (dvec * n).sum(1)[:, None] * n, axis=1)
    inside_disc = comp_t <= s01[prior][d_idx[1]].max(1) * 2.0  # within ~2 sigma of the surfel footprint
    res["offset_observed_to_prior_mm"] = {"total": stats(d_idx[0] * 1000), "along_normal": stats(comp_n * 1000), "tangential": stats(comp_t * 1000),
                                          "share_layer(normal>2mm & tangential<=2mm)": float(((comp_n > 0.002) & (comp_t <= 0.002)).mean()),
                                          "share_normal_dominant": float((comp_n > comp_t).mean()),
                                          "share_inside_prior_disc_footprint": float(inside_disc.mean()),
                                          "along_normal_if_inside_footprint": stats(comp_n[inside_disc] * 1000)}

    # 2. GT alignment (benchmark: first online pose, ICP 2 cm) and distance to the GT surface per category
    ids, gt_poses, K_seq, hw = load_gt_poses(args.dataset, args.video_dir)
    gt_mesh = load_gt_mesh(args.dataset, args.video_dir.name)
    first = sorted(os.listdir(args.run_dir / "ob_in_cam"))[0]
    pred0 = np.loadtxt(args.run_dir / "ob_in_cam" / first).reshape(4, 4)
    T0 = np.linalg.inv(gt_poses[0]) @ pred0
    if args.dataset == "ho3d":
        p1_gt = np.asarray(o3d.io.read_point_cloud(str(args.video_dir / "visible_mesh.ply")).voxel_down_sample(VOXEL).points).copy()
    else:
        p1_gt = np.asarray(trimesh.sample.sample_surface(gt_mesh, 99999, seed=0)[0], dtype=np.float64)
    good = (normal | cats["prior_verified"])
    pts = means[good] @ T0[:3, :3].T + T0[:3, 3]
    pts = pts[((pts <= p1_gt.max(0) + 0.3) & (pts >= p1_gt.min(0) - 0.3)).all(1)]
    reg = o3d.pipelines.registration.registration_icp(to_o3d(pts).voxel_down_sample(VOXEL), to_o3d(p1_gt), ICP_THRES, np.eye(4),
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint())
    T = np.asarray(reg.transformation) @ T0
    gt_dense = np.asarray(trimesh.sample.sample_surface(gt_mesh, 300000, seed=1)[0], dtype=np.float64)
    tree_gt = cKDTree(gt_dense)
    all_gt = means @ T[:3, :3].T + T[:3, 3]
    d_gt = tree_gt.query(all_gt)[0] * 1000
    res["distance_to_gt_mm"] = {k: stats(d_gt[v]) | {"share_gt_1cm": float((d_gt[v] > 10).mean()) if v.any() else None} for k, v in cats.items()}
    Tinv = np.linalg.inv(T)
    gt_in_map = gt_mesh.copy(); gt_in_map.apply_transform(Tinv); gt_in_map.export(out / "gt_aligned.obj")

    # 3. rendering with / without the opacity < 0.1 Gaussians
    step = max(1, len(r.views) // int(args.views)); rows = []
    opac_param = r.splats["opacities"].data; saved = opac_param.clone()
    hide = torch.from_numpy(low).to(opac_param.device)
    for view in r.views[::step]:
        c2w = r.normalization.metric_c2w(view.c2w_normalized.numpy()).astype(np.float64)
        K = view.K.numpy(); obs_d = view.depth.numpy(); m = view.mask.numpy() & (obs_d > 0.05)
        rgb_a, alpha_a, dep_a = render_depth(r, K, c2w, view.width, view.height, "median")
        opac_param[hide] = -30.0
        rgb_b, alpha_b, dep_b = render_depth(r, K, c2w, view.width, view.height, "median")
        opac_param.copy_(saved)
        ok_a = m & (dep_a > 0.05); ok_b = m & (dep_b > 0.05)
        rows.append({"alpha_in_mask_all": float(alpha_a[m].mean()), "alpha_in_mask_no_low": float(alpha_b[m].mean()),
                     "depth_err_mm_all": float(np.median(np.abs(dep_a[ok_a] - obs_d[ok_a])) * 1000) if ok_a.any() else None,
                     "depth_err_mm_no_low": float(np.median(np.abs(dep_b[ok_b] - obs_d[ok_b])) * 1000) if ok_b.any() else None,
                     "rgb_l1_between": float(np.abs(rgb_a[m] - rgb_b[m]).mean()),
                     "hole_share_no_low(alpha<0.5 in mask)": float((alpha_b[m] < 0.5).mean()), "hole_share_all": float((alpha_a[m] < 0.5).mean())})
    keys = rows[0].keys()
    res["render_without_low_opacity"] = {k: float(np.nanmean([row[k] for row in rows if row[k] is not None])) for k in keys}

    # 4. layers
    for k, v in cats.items():
        if k == "big_radius":
            write_obj_discs(out / "big_radius_discs.obj", means[v], R[v], s01[v], COLORS[k])
            write_obj_points(out / "big_radius_centres.obj", means[v], COLORS[k])
        else:
            write_obj_points(out / f"{k}.obj", means[v], COLORS[k])
    col = np.zeros((len(means), 3)); col[prior] = COLORS["prior_verified"]; col[cats["prior_unseen"]] = COLORS["prior_unseen"]
    col[cats["prior_suspect"]] = COLORS["prior_suspect"]; col[normal] = COLORS["observed_normal"]; col[obs & low] = COLORS["observed_low_opacity"]
    col[single] = COLORS["observed_single_view"]; col[big] = COLORS["big_radius"]; col[far] = COLORS["far"]
    write_obj_points(out / "all_gaussians_colored.obj", means, col)
    json.dump(res, open(out / "analysis.json", "w"), indent=1)
    with open(out / "README.txt", "w") as f:
        f.write("Layers (metric map frame, open together in MeshLab; gt_aligned.obj = GT model in the same frame)\n")
        for k, c in COLORS.items():
            f.write(f"  {k:24s} rgb{c}  n={res['counts'][k]}\n")
        f.write(f"big_radius of prior lineage: {res['big_radius_prior_lineage']} / {res['counts']['big_radius']}; far of observed lineage: {res['far_observed_lineage']} / {res['counts']['far']}\n")
        f.write(json.dumps({k: res[k] for k in ("overlaps", "support_count_observed", "offset_observed_to_prior_mm", "distance_to_gt_mm", "render_without_low_opacity")}, indent=1))
    o = res["offset_observed_to_prior_mm"]; g = res["distance_to_gt_mm"]; rw = res["render_without_low_opacity"]
    print(f"{args.run_dir.name}: N {len(means)} counts {res['counts']} | offsets obs->prior: total {o['total']['median']:.2f} normal {o['along_normal']['median']:.2f} "
          f"tangential {o['tangential']['median']:.2f} mm, normal-dominant {100*o['share_normal_dominant']:.0f}%, inside footprint {100*o['share_inside_prior_disc_footprint']:.0f}% | "
          f"support<=1 obs {res['counts']['observed_single_view']} | dist to GT (median mm): " + ", ".join(f"{k} {g[k].get('median', float('nan')):.1f}" for k in g)
          + f" | render w/o low: alpha {rw['alpha_in_mask_all']:.2f}->{rw['alpha_in_mask_no_low']:.2f}, depth err {rw['depth_err_mm_all']:.2f}->{rw['depth_err_mm_no_low']:.2f} mm, "
          f"holes {100*rw['hole_share_all']:.1f}%->{100*rw['hole_share_no_low(alpha<0.5 in mask)']:.1f}%, rgb L1 {rw['rgb_l1_between']:.3f}")


if __name__ == "__main__":
    main()
