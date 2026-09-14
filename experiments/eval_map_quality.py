"""Map-quality evaluation of a Gaussian checkpoint (B6, 2026-09-12) — how good is the MAP itself, independent of the
mesh extraction.  Reads ``checkpoint_global.pt`` (or ``gs_online/checkpoint_final.pt``) plus the run's ``ob_in_cam``.

Reports
  opacity   : share of Gaussians with opacity < 0.1 / 0.1–0.5 / >= 0.5, per lineage (prior / observed)
  outliers  : in-plane radius > 10 mm; normalized distance > 1.25 from the object centre
  render    : |2DGS median depth − observed depth| at the training views (median / p90 mm) — what the map renders
  centres   : observed depth point → nearest Gaussian centre (opacity >= 0.1) (median / p90 mm) — where the surfels are
  vs GT     : Gaussian centres (opacity >= 0.1, radius <= 10 mm, dist <= 1.25) moved to the GT object frame with the
              benchmark alignment (first online pose, ICP 2 cm onto the visible GT), then: centres → GT model distance
              (accuracy), GT → centres per region (completeness: seen / unseen, share within 5 mm)
  lifecycle : state counts
usage: eval_map_quality.py --dataset ho3d --video-dir ... --run-dir ... [--checkpoint ...] --out-json ...
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

from eval_mesh_cd import ICP_THRES, N_SAMPLES, VOXEL, load_gt_mesh, load_gt_poses, seen_labels_ycb, to_o3d  # noqa: E402
from gaussian_global import load_online_checkpoint, render_depth  # noqa: E402
from prior_lifecycle import STATE_CONTRADICTED, STATE_SUSPECT, STATE_UNSEEN, STATE_VERIFIED  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None, help="default <run-dir>/final/gs/checkpoint_global.pt")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--views", type=int, default=15, help="training views sampled for the render/centre residuals")
    args = ap.parse_args()
    ckpt = args.checkpoint or args.run_dir / "final" / "gs" / "checkpoint_global.pt"
    r = load_online_checkpoint(ckpt, device=args.device)
    out = {"checkpoint": str(ckpt), "gaussians": r.num_gaussians, "views": len(r.views)}

    with torch.no_grad():
        means = r.normalization.metric_points(r.splats["means"].detach().cpu().numpy()).astype(np.float64)
        opac = torch.sigmoid(r.splats["opacities"].detach()).cpu().numpy()
        radius_mm = (torch.exp(r.splats["scales"][:, :2]).max(1).values / r.normalization.scale * 1000).detach().cpu().numpy()
        dist_norm = np.linalg.norm(r.splats["means"].detach().cpu().numpy(), axis=1)
    lineage = r.lifecycle_fields.lineage.cpu().numpy() if r.lifecycle_fields is not None else np.zeros(len(means), bool)
    state = r.lifecycle_fields.state.cpu().numpy() if r.lifecycle_fields is not None else np.full(len(means), STATE_VERIFIED)
    active = state != STATE_CONTRADICTED

    def opac_bins(m):
        n = max(int(m.sum()), 1)
        return {"n": int(m.sum()), "lt0.1": float((opac[m] < 0.1).mean()) if m.any() else None,
                "0.1_0.5": float(((opac[m] >= 0.1) & (opac[m] < 0.5)).mean()) if m.any() else None,
                "ge0.5": float((opac[m] >= 0.5).mean()) if m.any() else None, "median": float(np.median(opac[m])) if m.any() else None}
    out["opacity"] = {"all_active": opac_bins(active), "prior": opac_bins(active & lineage), "observed": opac_bins(active & ~lineage)}
    out["outliers"] = {"radius_gt_10mm": int(((radius_mm > 10) & active).sum()), "radius_gt_10mm_prior": int(((radius_mm > 10) & active & lineage).sum()),
                       "dist_gt_1.25": int(((dist_norm > 1.25) & active).sum()), "radius_p99_mm": float(np.percentile(radius_mm[active], 99))}
    out["lifecycle"] = {"unseen": int((state == STATE_UNSEEN).sum()), "verified": int((state == STATE_VERIFIED).sum()),
                        "suspect": int((state == STATE_SUSPECT).sum()), "contradicted": int((state == STATE_CONTRADICTED).sum())}

    # render vs observed depth, centres vs observed depth
    keep_c = (opac >= 0.1) & (radius_mm <= 10) & (dist_norm <= 1.25) & active
    tree_c = cKDTree(means[keep_c]) if keep_c.sum() > 10 else None
    rr, sg, cc = [], [], []
    step = max(1, len(r.views) // int(args.views))
    for view in r.views[::step]:
        c2w = r.normalization.metric_c2w(view.c2w_normalized.numpy()).astype(np.float64)
        rgb, alpha, depth = render_depth(r, view.K.numpy(), c2w, view.width, view.height, "median")
        obs = view.depth.numpy(); m = view.mask.numpy() & (obs > 0.05) & (alpha > 0.05) & (depth > 0.05)
        if not m.any():
            continue
        rr.append(np.abs(depth[m] - obs[m]) * 1000); sg.append((depth[m] - obs[m]) * 1000)
        if tree_c is not None:
            ys, xs = np.nonzero(m); z = obs[ys, xs]; K = view.K.numpy()
            pc = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
            cc.append(tree_c.query(pc @ c2w[:3, :3].T + c2w[:3, 3])[0] * 1000)
    rr, sg = np.concatenate(rr), np.concatenate(sg)
    out["render_depth_residual_mm"] = {"median": float(np.median(rr)), "p90": float(np.percentile(rr, 90)), "signed_median": float(np.median(sg))}
    if cc:
        cc = np.concatenate(cc); out["centre_residual_mm"] = {"median": float(np.median(cc)), "p90": float(np.percentile(cc, 90))}

    # vs GT (benchmark alignment)
    ids, gt_poses, K_seq, hw = load_gt_poses(args.dataset, args.video_dir)
    gt_mesh = load_gt_mesh(args.dataset, args.video_dir.name)
    first = sorted(os.listdir(args.run_dir / "ob_in_cam"))[0]
    pred0 = np.loadtxt(args.run_dir / "ob_in_cam" / first).reshape(4, 4)
    T = np.linalg.inv(gt_poses[0]) @ pred0
    pts = means[keep_c] @ T[:3, :3].T + T[:3, 3]
    full_pts, full_faces = trimesh.sample.sample_surface(gt_mesh, N_SAMPLES, seed=0); full_pts = np.asarray(full_pts, dtype=np.float64)
    if args.dataset == "ho3d":
        vis = o3d.io.read_point_cloud(str(args.video_dir / "visible_mesh.ply")).voxel_down_sample(VOXEL); p1_gt = np.asarray(vis.points).copy()
        vmesh = trimesh.load(str(args.video_dir / "visible_mesh.obj"), force="mesh", process=False)
        seen = np.isfinite(vmesh.vertices).all(1)[gt_mesh.faces[np.asarray(full_faces)]].all(1)
    else:
        p1_gt = full_pts
        import glob, yaml
        kf_yml = sorted(glob.glob(str(args.run_dir / "*/keyframes.yml")))[-1]
        kf_ids = [k.replace("keyframe_", "") for k in yaml.safe_load(open(kf_yml)).keys()]
        kf_poses = [gt_poses[ids.index(f)] for f in kf_ids if f in ids and gt_poses[ids.index(f)] is not None]
        seen = seen_labels_ycb(gt_mesh, full_pts, kf_poses, K_seq, hw)
    bbox_ok = ((pts <= p1_gt.max(0) + 0.3) & (pts >= p1_gt.min(0) - 0.3)).all(1)
    pts = pts[bbox_ok]
    pcd = to_o3d(pts).voxel_down_sample(VOXEL)
    reg = o3d.pipelines.registration.registration_icp(pcd, to_o3d(p1_gt), ICP_THRES, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())
    Ti = np.asarray(reg.transformation); pts_i = pts @ Ti[:3, :3].T + Ti[:3, 3]
    d_c2g = cKDTree(full_pts).query(pts_i)[0]; d_g2c = cKDTree(pts_i).query(full_pts)[0]
    out["vs_gt"] = {"centres_used": int(len(pts_i)), "icp_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(Ti[:3, :3]) - 1) / 2, -1, 1)))),
                    "centres_to_gt_mm": {"median": float(np.median(d_c2g) * 1000), "mean": float(d_c2g.mean() * 1000), "gt1cm_frac": float((d_c2g > 0.01).mean())},
                    "gt_to_centres_mm": {"seen_mean": float(d_g2c[seen].mean() * 1000), "unseen_mean": float(d_g2c[~seen].mean() * 1000) if (~seen).any() else None,
                                         "seen_within5mm": float((d_g2c[seen] < 0.005).mean()), "unseen_within5mm": float((d_g2c[~seen] < 0.005).mean()) if (~seen).any() else None,
                                         "seen_frac": float(seen.mean())}}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1)
    o, g, v = out["opacity"], out["vs_gt"]["gt_to_centres_mm"], out["vs_gt"]["centres_to_gt_mm"]
    print(f"{args.run_dir.name}: gaussians {r.num_gaussians} | opac<0.1 obs {o['observed']['lt0.1']:.2f} prior {o['prior']['lt0.1']:.2f} | big {out['outliers']['radius_gt_10mm']} far {out['outliers']['dist_gt_1.25']} | "
          f"render {out['render_depth_residual_mm']['median']:.2f} mm, centres {out.get('centre_residual_mm', {}).get('median', float('nan')):.2f} mm | "
          f"vsGT: centres->GT {v['median']:.2f} mm, GT->centres seen {g['seen_mean']:.2f} unseen {g['unseen_mean'] if g['unseen_mean'] is not None else float('nan'):.2f} mm, unseen<5mm {100*(g['unseen_within5mm'] or 0):.0f}% | "
          f"lifecycle {out['lifecycle']}")


if __name__ == "__main__":
    main()
