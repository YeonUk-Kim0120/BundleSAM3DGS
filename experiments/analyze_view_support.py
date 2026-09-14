"""Replay the lifecycle evidence over all keyframes of a finished run and count, per Gaussian, in how many views it was
judged / supported / contradicted — the statistic behind "single-view" Gaussians and the PROVISIONAL design
(B-track discussion, 2026-09-14).  No training; reads ``checkpoint_global.pt`` and the stored keyframes.

Per Gaussian: judged views (projects inside the eroded mask with valid depth), supported views (|delta| <= tolerance,
any angle), independent supported views (greedy, >= 10 deg apart), free-space views, behind-occluded views.
Categories of the observed lineage:
  never_judged            never inside an eroded mask with valid depth (far leaks, silhouette rim)
  judged_never_supported  judged but always > tolerance from the observed surface (in front: free space, or hidden behind)
  support_1               supported from one viewpoint cluster only
  support_2plus           supported from >= 2 independent viewpoints
Writes analysis_support.json, layers support_<cat>.obj/.ply, xsection_support.png and xsection_layers.png (3 mm slabs
through the object centre; the layer figure colours by lineage / opacity) into ``--out-dir``.
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
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))

from analyze_map_layers import write_obj_points  # noqa: E402
from eval_mesh_cd import ICP_THRES, VOXEL, load_gt_mesh, load_gt_poses, to_o3d  # noqa: E402
from gaussian_global import load_online_checkpoint  # noqa: E402
from prior_lifecycle import STATE_CONTRADICTED, STATE_VERIFIED, depth_evidence_masks, erode_mask  # noqa: E402

CAT_COLORS = {"never_judged": (160, 0, 200), "judged_never_supported_front": (230, 0, 0), "judged_never_supported_hidden": (140, 80, 20),
              "support_1": (255, 140, 0), "support_2plus": (120, 120, 120)}


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90))} if len(x) else {"n": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint or args.run_dir / "final" / "gs" / "checkpoint_global.pt"
    r = load_online_checkpoint(ckpt, device=args.device)
    dev = r.device; thr = r.lifecycle_thresholds
    with torch.no_grad():
        means_np = r.normalization.metric_points(r.splats["means"].detach().cpu().numpy()).astype(np.float32)
        means = torch.from_numpy(means_np).to(dev)
        normals = r._splat_normals()
        opac = torch.sigmoid(r.splats["opacities"].detach()).cpu().numpy()
    lf = r.lifecycle_fields
    state = lf.state.cpu().numpy(); lineage = lf.lineage.cpu().numpy()
    active = torch.from_numpy(state != STATE_CONTRADICTED).to(dev)
    N = len(means_np)
    judged = torch.zeros(N, dtype=torch.int32, device=dev); support = judged.clone(); free = judged.clone(); hidden = judged.clone(); miss = judged.clone()
    indep = judged.clone(); last_dir = torch.zeros(N, 3, device=dev)
    cos_thr = float(np.cos(np.radians(thr.min_view_angle_deg)))
    min_d, max_d = float(r.config["min_depth"]), float(r.config["max_depth"])
    with torch.no_grad():
        for view in r.views:
            c2w_np = r.normalization.metric_c2w(view.c2w_normalized.numpy()).astype(np.float32)
            c2w = torch.from_numpy(c2w_np).to(dev); w2c = torch.linalg.inv(c2w)
            pc = means @ w2c[:3, :3].T + w2c[:3, 3][None, :]
            z = pc[:, 2]; safe = z.clamp_min(1e-6); K = view.K.numpy()
            H, W = view.height, view.width
            u = pc[:, 0] / safe * float(K[0, 0]) + float(K[0, 2]); v = pc[:, 1] / safe * float(K[1, 1]) + float(K[1, 2])
            in_image = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            ui = u.round().clamp(0, W - 1).long(); vi = v.round().clamp(0, H - 1).long()
            observed = view.depth.to(dev) if torch.is_tensor(view.depth) else torch.from_numpy(view.depth).to(dev)
            mask_t = view.mask.to(dev) if torch.is_tensor(view.mask) else torch.from_numpy(view.mask).to(dev)
            mask_e = erode_mask(mask_t.bool(), thr.mask_erode_px)
            valid_px = mask_e & torch.isfinite(observed) & (observed > min_d) & (observed < max_d)
            valid = active & in_image & valid_px[vi, ui]
            K_t = torch.from_numpy(K.astype(np.float32)).to(dev)
            front_depth, front_alpha = r._geometric_front_depth(view.c2w_normalized.to(dev), K_t, W, H)
            view_dir = F.normalize(means - c2w[:3, 3][None, :], dim=-1, eps=1e-8)
            ev = depth_evidence_masks(gaussian_depth=z, observed_depth=observed[vi, ui], valid_observation=valid,
                                      geometric_front_depth=front_depth[vi, ui], geometric_alpha=front_alpha[vi, ui],
                                      depth_tolerance=thr.depth_tolerance_m, geometric_alpha_threshold=thr.geometric_alpha_threshold,
                                      view_abs_cos=(normals * view_dir).sum(-1).abs(), grazing_tolerance_cap=thr.grazing_tolerance_cap)
            judged += valid.int(); support += ev["support"].int(); free += ev["free_space"].int(); hidden += ev["behind_occluded"].int(); miss += ev["behind_miss"].int()
            new_cluster = ev["support"] & ((last_dir * view_dir).sum(-1) < cos_thr)
            indep += new_cluster.int(); last_dir[new_cluster] = view_dir[new_cluster]
    J, S, Fr, Hd, I = (t.cpu().numpy() for t in (judged, support, free, hidden, indep))
    obs = ~lineage
    cats = {"never_judged": J == 0, "judged_never_supported_front": (J > 0) & (S == 0) & (Fr >= Hd), "judged_never_supported_hidden": (J > 0) & (S == 0) & (Fr < Hd),
            "support_1": I == 1, "support_2plus": I >= 2}
    res = {"checkpoint": str(ckpt), "views": len(r.views), "gaussians": int(N),
           "observed": {k: int((v & obs).sum()) for k, v in cats.items()}, "prior": {k: int((v & lineage).sum()) for k, v in cats.items()},
           "observed_low_opacity_share_per_cat": {k: float((opac[v & obs] < 0.1).mean()) if (v & obs).any() else None for k, v in cats.items()},
           "observed_judged_views": {"median": float(np.median(J[obs])), "p10": float(np.percentile(J[obs], 10))},
           "observed_indep_support_hist": {str(k): int(((I == k) & obs).sum()) for k in range(0, 5)} | {">=5": int(((I >= 5) & obs).sum())}}
    # GT distance per category (benchmark alignment on the well-supported set)
    ids, gt_poses, K_seq, hw = load_gt_poses(args.dataset, args.video_dir)
    gt_mesh = load_gt_mesh(args.dataset, args.video_dir.name)
    first = sorted(os.listdir(args.run_dir / "ob_in_cam"))[0]
    pred0 = np.loadtxt(args.run_dir / "ob_in_cam" / first).reshape(4, 4); T0 = np.linalg.inv(gt_poses[0]) @ pred0
    if args.dataset == "ho3d":
        p1_gt = np.asarray(o3d.io.read_point_cloud(str(args.video_dir / "visible_mesh.ply")).voxel_down_sample(VOXEL).points).copy()
    else:
        p1_gt = np.asarray(trimesh.sample.sample_surface(gt_mesh, 99999, seed=0)[0], dtype=np.float64)
    good = cats["support_2plus"] & (opac >= 0.1)
    pts = means_np[good].astype(np.float64) @ T0[:3, :3].T + T0[:3, 3]
    pts = pts[((pts <= p1_gt.max(0) + 0.3) & (pts >= p1_gt.min(0) - 0.3)).all(1)]
    reg = o3d.pipelines.registration.registration_icp(to_o3d(pts).voxel_down_sample(VOXEL), to_o3d(p1_gt), ICP_THRES, np.eye(4),
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint())
    T = np.asarray(reg.transformation) @ T0
    gt_dense = np.asarray(trimesh.sample.sample_surface(gt_mesh, 300000, seed=1)[0], dtype=np.float64); tree = cKDTree(gt_dense)
    d_gt = tree.query(means_np.astype(np.float64) @ T[:3, :3].T + T[:3, 3])[0] * 1000
    res["distance_to_gt_mm_observed"] = {k: stats(d_gt[v & obs]) | ({"share_gt_1cm": float((d_gt[v & obs] > 10).mean())} if (v & obs).any() else {}) for k, v in cats.items()}
    res["distance_to_gt_mm_prior"] = {k: stats(d_gt[v & lineage]) for k, v in cats.items()}
    json.dump(res, open(out / "analysis_support.json", "w"), indent=1)
    for k, v in cats.items():
        write_obj_points(out / f"support_{k}.obj", means_np[v & obs].astype(np.float64), CAT_COLORS[k])
    np.savez_compressed(out / "view_support_counts.npz", judged=J, support=S, free_space=Fr, hidden=Hd, indep_support=I, lineage=lineage, opacity=opac, d_gt_mm=d_gt)

    # cross sections: 3 mm slabs through the GT centre, in the map frame
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Tinv = np.linalg.inv(T); gt_map = gt_dense @ Tinv[:3, :3].T + Tinv[:3, 3]; centre = gt_map.mean(0)
    for fname, title in (("xsection_support.png", "view support"), ("xsection_layers.png", "lineage / opacity")):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        for ax, (i, j, k, lab) in zip(axes, [(0, 1, 2, "XY slab (|z-c|<1.5mm)"), (0, 2, 1, "XZ slab (|y-c|<1.5mm)"), (1, 2, 0, "YZ slab (|x-c|<1.5mm)")]):
            slab_g = np.abs(gt_map[:, k] - centre[k]) < 0.0015; ax.scatter(gt_map[slab_g, i], gt_map[slab_g, j], s=1, c="k", label="GT")
            slab = np.abs(means_np[:, k] - centre[k]) < 0.0015
            if fname == "xsection_support.png":
                for c, m in cats.items():
                    sel = slab & m & obs
                    ax.scatter(means_np[sel, i], means_np[sel, j], s=3, c=[np.array(CAT_COLORS[c]) / 255], label=f"{c} ({int((m & obs).sum())})")
                sel = slab & lineage; ax.scatter(means_np[sel, i], means_np[sel, j], s=3, c=[(0, 0.7, 0)], label="prior")
            else:
                for name, m, col in (("prior", lineage, (0, 0.7, 0)), ("observed opac>=0.1", obs & (opac >= 0.1), (0.4, 0.4, 0.4)), ("observed opac<0.1", obs & (opac < 0.1), (0.15, 0.35, 1.0))):
                    sel = slab & m; ax.scatter(means_np[sel, i], means_np[sel, j], s=3, c=[col], label=f"{name} ({int(m.sum())})")
            ax.set_aspect("equal"); ax.set_title(f"{args.run_dir.name} {lab}"); ax.legend(markerscale=4, fontsize=7)
        fig.suptitle(f"{args.run_dir.name}: cross sections, {title} (map frame, GT aligned)"); fig.tight_layout(); fig.savefig(out / fname, dpi=90); plt.close(fig)
    o = res["observed"]; d = res["distance_to_gt_mm_observed"]
    print(f"{args.run_dir.name}: observed {int(obs.sum())} | never_judged {o['never_judged']} | judged_never_supported front {o['judged_never_supported_front']} hidden {o['judged_never_supported_hidden']} | "
          f"support_1 {o['support_1']} | support_2plus {o['support_2plus']} | median judged views {res['observed_judged_views']['median']:.0f} | dist to GT (median mm): "
          + ", ".join(f"{k} {d[k].get('median', float('nan')):.1f}" for k in d) + f" | prior: {res['prior']}")


if __name__ == "__main__":
    main()
