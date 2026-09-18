"""Prior Sim(3) alignment with 1 / 3 / 5 keyframes at a fixed step budget (2026-09-16, user request).

The online backend aligns the SAM3D prior to the FIRST keyframe only (bundlesdf.run_gaussian → align_prior_sim3).
Here the same optimiser (RtsParameters, refine_loss, same config as online: use_ssim True, 400 steps) sees N keyframes
round-robin (one frame per step), so the total number of steps is unchanged.  Frame k is rendered through the relative
tracker pose  viewmat_k = ob_in_cam_k · ob_in_cam_0⁻¹  (poses of a finished run), i.e. the surfels stay in the
camera-0 frame as online.

Reference placement: the SAM3D mesh at the 1-frame result is registered onto the GT model at the GT pose of frame 0
by scaled ICP (as experiments/make_oracle_prior.py does with the online-refined pose); every 50 steps each run's
current pose is compared with that placement on the SAM3D mesh vertices: mean displacement (mm), rotation (deg,
Kabsch), translation (mm, centroids), scale ratio.
usage: exp_prior_align_multiframe.py --dataset ho3d --video-dir ... --run-dir <finished run> --run-log <its log> --out-dir ...
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))

from make_oracle_prior import cv_from_pose, load_gt_frame0, load_gt_mesh, sim3_icp  # noqa: E402
from run_sam3d_alignment import (DEFAULT_CONFIG as ALIGN_DEFAULTS, RtsParameters, axis_angle_to_matrix,  # noqa: E402
                                 make_photo_metrics, prepare_target, refine_loss)
from sam3d_prior import (Sim3Pose, load_mesh_prior, load_sam3d_gaussian_ply, load_sam3d_pose_or_refined,  # noqa: E402
                         quats_from_normals, sample_surfels, transfer_gaussian_colors)

SAM3D_ROOT = Path("/home/kist/Desktop/sam-3d-objects")


def prior_paths(dataset: str, seq: str) -> dict:
    root = SAM3D_ROOT / ("output_HO3D_sam2mask_mesh" if dataset == "ho3d" else "output_YCBInEOAT_sam2mask_mesh")
    return {"mesh_npz": root / f"{seq}_mesh_depth.npz", "pose_json": root / f"{seq}_mesh_depth.json", "gaussian_ply": root / f"{seq}_splat_depth.ply"}


def render_surfels_view(surfels, pose, rot_vec, trans_delta, scale_delta, target, viewmat, config, device):
    """run_sam3d_alignment.render_surfels with an explicit CV view matrix (camera k relative to camera 0)."""
    from gsplat import rasterization_2dgs
    scale = pose.scale.to(device)
    xyz0 = surfels.means.to(device) * scale[None, :]
    delta_r = axis_angle_to_matrix(rot_vec)
    xyz_delta = (xyz0 * scale_delta) @ delta_r.T + trans_delta[None, :]
    xyz_p3d = xyz_delta @ pose.R_row.to(device) + pose.T.to(device)[None, :]
    normals_p3d = (surfels.normals.to(device) @ delta_r.T) @ pose.R_row.to(device)
    xyz_cv = xyz_p3d.clone(); xyz_cv[:, 0] *= -1.0; xyz_cv[:, 1] *= -1.0
    normals_cv = normals_p3d.clone(); normals_cv[:, 0] *= -1.0; normals_cv[:, 1] *= -1.0
    radii = (surfels.radii.to(device) * float(scale.mean())) * scale_delta
    scales = torch.stack((radii, radii, radii * 0.1), dim=-1)
    size = int(config["render_size"])
    sh0 = (surfels.colors.to(device) - 0.5) / 0.28209479177387814
    renders, alphas, *_ = rasterization_2dgs(
        means=xyz_cv, quats=quats_from_normals(normals_cv), scales=scales, opacities=surfels.opacities.to(device),
        colors=sh0[:, None, :], sh_degree=0, viewmats=viewmat[None], Ks=target["K"][None], width=size, height=size,
        render_mode="RGB+ED", near_plane=float(config["near_plane"]), far_plane=float(config["far_plane"]), eps2d=float(config["eps2d"]))
    image = renders[0]
    return {"rgb": image[..., :3].permute(2, 0, 1), "depth": image[..., 3], "alpha": alphas[0, ..., 0]}


def compose(init_pose: Sim3Pose, rot_vec, trans_delta, scale_delta) -> Sim3Pose:
    delta_r = axis_angle_to_matrix(rot_vec).detach().cpu()
    return Sim3Pose(scale=init_pose.scale * float(scale_delta), R_row=delta_r.T @ init_pose.R_row,
                    T=trans_delta.detach().cpu() @ init_pose.R_row + init_pose.T)


def compare_points(est: np.ndarray, ref: np.ndarray) -> dict:
    """Mean displacement, Kabsch rotation, centroid translation and RMS-radius scale ratio between two vertex sets."""
    ce, cr = est.mean(0), ref.mean(0)
    a, b = est - ce, ref - cr
    H = a.T @ b; U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    return {"mesh_disp_mm": float(np.linalg.norm(est - ref, axis=1).mean() * 1000), "rot_deg": float(ang),
            "trans_mm": float(np.linalg.norm(ce - cr) * 1000), "scale_ratio": float(np.sqrt((a ** 2).sum(1).mean() / (b ** 2).sum(1).mean()))}


def keyframe_ids(run_log: Path, n: int) -> list[str]:
    ids = re.findall(r"Added frame (\S+) as keyframe", open(run_log, errors="ignore").read())
    return ids[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--run-log", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trace", action="store_true", help="record every step (loss parts + pose) for the best-selection analysis")
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); seq = args.video_dir.name
    config = dict(ALIGN_DEFAULTS); config["use_ssim"] = True; config["use_ms_ssim"] = False
    if args.steps: config["steps"] = int(args.steps)
    paths = prior_paths(args.dataset, seq)
    prior = load_mesh_prior(str(paths["mesh_npz"])); init_pose = load_sam3d_pose_or_refined(str(paths["pose_json"]))
    surfels = sample_surfels(prior, 20000, seed=0, radius_multiplier=0.75)
    surfels, _ = transfer_gaussian_colors(surfels, load_sam3d_gaussian_ply(str(paths["gaussian_ply"])))
    mesh_v = prior.vertices.numpy().astype(np.float64)

    if args.dataset == "ho3d":
        from data_reader import Ho3dReader  # noqa: E402
        reader = Ho3dReader(str(args.video_dir))
    else:
        from data_reader import YcbineoatReader  # noqa: E402
        reader = YcbineoatReader(str(args.video_dir), mask_dir="masks_sam2")
    names = [os.path.basename(f).split(".")[0] for f in reader.color_files]

    def frame(i):
        import cv2
        rgb = cv2.cvtColor(cv2.imread(reader.color_files[i]), cv2.COLOR_BGR2RGB)
        return {"frame_id": names[i], "rgb": rgb, "depth": np.asarray(reader.get_depth(i), dtype=np.float32),
                "mask": np.asarray(reader.get_mask(i)) > 0, "K": np.asarray(reader.K, dtype=np.float64)}
    kf = keyframe_ids(args.run_log, max(args.frames))
    if len(kf) < max(args.frames):
        raise RuntimeError(f"only {len(kf)} keyframes found in {args.run_log}")
    idx = [names.index(k) for k in kf]
    ob_in_cam = {k: np.loadtxt(args.run_dir / "ob_in_cam" / f"{k}.txt").reshape(4, 4) for k in kf}
    targets, viewmats = [], []
    for k, i in zip(kf, idx):
        targets.append(prepare_target(frame(i), config, device))
        vm = ob_in_cam[k] @ np.linalg.inv(ob_in_cam[kf[0]])
        viewmats.append(torch.from_numpy(vm.astype(np.float32)).to(device))
    ssim_metric, ms_ssim_metric = make_photo_metrics(config, device)

    def run(n: int):
        torch.manual_seed(int(config["surfel_seed"]))
        params = RtsParameters(config, device); optimizer = torch.optim.AdamW(params.parameters(), lr=float(config["lr"]))
        steps, warmup, lr_max, lr_end = int(config["steps"]), int(config["warmup"]), float(config["lr"]), float(config["end_lr"])
        def lr_at(s):
            return lr_max * (s + 1) / max(warmup, 1) if s < warmup else lr_end + 0.5 * (lr_max - lr_end) * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1)))
        best = {"loss": float("inf"), "step": -1, "pose": None}; snaps = []
        for step in range(steps):
            for g in optimizer.param_groups: g["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            rot_vec, trans_delta, log_scale_delta, scale_delta = params.current()
            j = step % n
            rendered = render_surfels_view(surfels, init_pose, rot_vec, trans_delta, scale_delta, targets[j], viewmats[j], config, device)
            loss, parts = refine_loss(rendered, targets[j], rot_vec, trans_delta, log_scale_delta, config, ssim_metric, ms_ssim_metric)
            if (np.isfinite(parts["total"]) and parts["visible_ratio"] >= float(config["best_min_visible_ratio"])
                    and parts["depth_valid_ratio"] >= float(config["best_min_depth_valid_ratio"]) and parts["total"] < best["loss"]):
                best = {"loss": float(parts["total"]), "step": step, "pose": compose(init_pose, rot_vec, trans_delta, scale_delta)}
            loss.backward(); torch.nn.utils.clip_grad_norm_(params.parameters(), float(config["grad_clip"])); optimizer.step()
            if step == 0 or (step + 1) % 50 == 0 or args.trace:
                r, t, ls, sd = params.current()
                snaps.append({"step": step + 1, "loss": float(parts["total"]), "pose": compose(init_pose, r, t, sd),
                              "parts": {k: float(v) for k, v in parts.items() if isinstance(v, (int, float, np.floating))}})
        if best["pose"] is None:
            best["pose"] = snaps[-1]["pose"]
        return best, snaps

    results = {"seq": seq, "dataset": args.dataset, "keyframes": kf, "steps": int(config["steps"]), "runs": {}}
    runs = {n: run(n) for n in args.frames}
    # reference placement from the 1-frame best pose (or the smallest N run)
    n_ref = 1 if 1 in runs else min(runs)
    src_ref = cv_from_pose(mesh_v, runs[n_ref][0]["pose"])
    gt_obj_in_cam0 = load_gt_frame0(args.dataset, args.video_dir); gt_mesh = load_gt_mesh(args.dataset, seq)
    gt_pts = np.asarray(gt_mesh.vertices, dtype=np.float64) @ gt_obj_in_cam0[:3, :3].T + gt_obj_in_cam0[:3, 3]
    T_icp, icp_info = sim3_icp(src_ref, gt_pts)
    ref_pts = src_ref @ T_icp[:3, :3].T + T_icp[:3, 3]
    results["gt_registration"] = {"from_frames": n_ref, "icp": icp_info, "start_correction": compare_points(src_ref, ref_pts)}
    init_err = compare_points(cv_from_pose(mesh_v, init_pose), ref_pts); results["init_error"] = init_err
    for n, (best, snaps) in runs.items():
        curve = [{"step": s["step"], "loss": s["loss"], **({"parts": s["parts"]} if args.trace else {}), **compare_points(cv_from_pose(mesh_v, s["pose"]), ref_pts)} for s in snaps]
        results["runs"][str(n)] = {"best_step": best["step"], "best_loss": best["loss"], "best": compare_points(cv_from_pose(mesh_v, best["pose"]), ref_pts), "curve": curve}
        print(f"{seq} frames={n}: init {init_err['mesh_disp_mm']:.1f} mm -> " + " ".join(f"s{c['step']}:{c['mesh_disp_mm']:.1f}" for c in curve[::2]) + f" | best(step {best['step']}) {results['runs'][str(n)]['best']['mesh_disp_mm']:.1f} mm, rot {results['runs'][str(n)]['best']['rot_deg']:.2f} deg, trans {results['runs'][str(n)]['best']['trans_mm']:.1f} mm, scale {results['runs'][str(n)]['best']['scale_ratio']:.3f}")
    json.dump(results, open(out / f"{seq}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
