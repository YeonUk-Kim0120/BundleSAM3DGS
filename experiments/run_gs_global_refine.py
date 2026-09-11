"""Experiment wrapper around ``gaussian_global`` (stage 1/2 of the global-refinement study, 2026-09-11).

The confirmed pipeline lives in ``gaussian_global.run_global_refine`` (main code).  This wrapper keeps the study's
extra options: ``--init prior|fresh`` rebuilds the map from the keyframes instead of continuing the online map,
pose refinement can be switched on, and the virtual-view TSDF can be added.

  python3 experiments/run_gs_global_refine.py --run-dir outputs/<run> --steps 2000 --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gaussian_global import DEFAULT_GLOBAL_CONFIG, extract_meshes, load_online_checkpoint, run_global_refine  # noqa: E402
from gaussian_runner import GaussianFrame, GaussianRunner, SceneNormalization  # noqa: E402


def keyframes_from_run(run_dir: Path, online_runner: GaussianRunner):
    """Full-resolution keyframes (RGB, depth, mask) of the online run with their final online poses (metric)."""
    from run_gaussian_incremental import load_frame
    ids = open(sorted(glob.glob(str(run_dir / "*/nerf_frames.txt")))[-1]).read().split()
    if len(ids) != len(online_runner.views):
        raise RuntimeError(f"keyframe count mismatch: nerf_frames {len(ids)} vs checkpoint views {len(online_runner.views)}")
    K = np.loadtxt(run_dir / "cam_K.txt").reshape(3, 3).astype(np.float32)
    frames = []
    for fid, view in zip(ids, online_runner.views):
        c2w = online_runner.normalization.metric_c2w(view.c2w_normalized.numpy()).astype(np.float32)
        f = load_frame(run_dir, fid, K, c2w)
        frames.append(GaussianFrame(view.frame_id, f.rgb, f.depth, f.mask, f.K, f.c2w_cv).validated())
    return frames


def build_runner_from_prior(config: dict, device: str, frames, steps: int, prior_root: Path, dataset: str, seq: str) -> GaussianRunner:
    """Mirror of bundlesdf.run_gaussian's first batch (SAM3D prior + Sim(3) alignment on frame 0 + initialize_from_prior),
    with ALL keyframes in the single initial training call."""
    from run_sam3d_alignment import DEFAULT_CONFIG as ALIGN_DEFAULTS
    from run_sam3d_alignment import align_prior_sim3, prepare_target
    from sam3d_prior import (load_mesh_prior, load_sam3d_gaussian_ply, load_sam3d_pose_or_refined, sample_surfels,
                             transfer_gaussian_colors, transform_surfels_canonical_to_cv_camera)
    sub = "output_YCBInEOAT_sam2mask_mesh" if dataset == "ycb" else "output_HO3D_sam2mask_mesh"
    paths = {"mesh_npz": prior_root / sub / f"{seq}_mesh_depth.npz", "pose_json": prior_root / sub / f"{seq}_mesh_depth.json",
             "gaussian_ply": prior_root / sub / f"{seq}_splat_depth.ply"}
    dev = torch.device(device)
    prior = load_mesh_prior(paths["mesh_npz"])
    init_pose = load_sam3d_pose_or_refined(paths["pose_json"])
    surfels = sample_surfels(prior, 20000, seed=0, radius_multiplier=0.75)
    surfels, _ = transfer_gaussian_colors(surfels, load_sam3d_gaussian_ply(paths["gaussian_ply"]))
    align_config = dict(ALIGN_DEFAULTS); align_config["use_ssim"] = True; align_config["use_ms_ssim"] = False
    first = frames[0]
    target = prepare_target({"frame_id": first.frame_id, "rgb": first.rgb, "depth": first.depth, "mask": first.mask, "K": first.K}, align_config, dev)
    refined_pose, _, _, status = align_prior_sim3(surfels, init_pose, target, align_config, dev, verbose=False)
    surfels_cv = transform_surfels_canonical_to_cv_camera(surfels, refined_pose)
    first_c2w = np.asarray(first.c2w_cv, dtype=np.float64)
    means_obj = surfels_cv.means.numpy() @ first_c2w[:3, :3].T + first_c2w[:3, 3][None, :]
    centre = means_obj.mean(axis=0); radius = float(np.linalg.norm(means_obj - centre, axis=1).max())
    normalization = SceneNormalization(scale=1.0 / max(radius * 1.2, 1e-6), translation=-centre)
    runner = GaussianRunner(config, normalization, device=device)
    print(f"[global] prior alignment {status}; normalization scale {normalization.scale:.3f}", flush=True)
    runner.initialize_from_prior(surfels_cv, first.c2w_cv, frames, train_steps=steps)
    return runner


def build_runner_fresh(config: dict, device: str, frames, steps: int) -> GaussianRunner:
    """No prior: normalization from the keyframes' masked depth points, RGB-D seeding (GaussianRunner.initialize)."""
    pts = []
    for f in frames:
        ys, xs = np.nonzero(f.mask & (f.depth > 0.05))
        z = f.depth[ys, xs]
        pc = np.stack([(xs - f.K[0, 2]) / f.K[0, 0] * z, (ys - f.K[1, 2]) / f.K[1, 1] * z, z], 1)
        pts.append(pc @ f.c2w_cv[:3, :3].T + f.c2w_cv[:3, 3])
    pts = np.concatenate(pts).astype(np.float64)
    centre = np.median(pts, axis=0)
    radius = float(np.percentile(np.linalg.norm(pts - centre, axis=1), 99))
    normalization = SceneNormalization(scale=1.0 / max(radius * 1.2, 1e-6), translation=-centre)
    runner = GaussianRunner(config, normalization, device=device)
    print(f"[global] fresh map: {len(pts)} depth points, radius {radius:.3f} m, scale {normalization.scale:.3f}", flush=True)
    runner.initialize(frames, train_steps=steps)
    return runner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None, help="default <run-dir>/gs_online/checkpoint_final.pt")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <run-dir>/final/gs (time-stamped if it exists)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=DEFAULT_GLOBAL_CONFIG["steps"])
    ap.add_argument("--pose-refine", action="store_true", help="joint map + v1 pose refinement (default off)")
    ap.add_argument("--init", choices=("online", "prior", "fresh"), default="online")
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), default=None)
    ap.add_argument("--seq", default=None)
    ap.add_argument("--prior-root", type=Path, default=Path("/home/kist/Desktop/sam-3d-objects"))
    ap.add_argument("--opacity-min", type=float, default=DEFAULT_GLOBAL_CONFIG["opacity_min"])
    ap.add_argument("--max-scale-mm", type=float, default=DEFAULT_GLOBAL_CONFIG["max_scale_mm"])
    ap.add_argument("--exclude-suspect", action="store_true")
    ap.add_argument("--depth-mode", choices=("median", "expected"), default=DEFAULT_GLOBAL_CONFIG["depth_mode"])
    ap.add_argument("--tsdf-virtual", action="store_true")
    ap.add_argument("--virtual-views", type=int, default=DEFAULT_GLOBAL_CONFIG["virtual_views"])
    ap.add_argument("--poisson-depth", type=int, default=DEFAULT_GLOBAL_CONFIG["poisson_depth"])
    args = ap.parse_args()

    cfg = {"steps": args.steps, "pose_refine": args.pose_refine, "opacity_min": args.opacity_min, "max_scale_mm": args.max_scale_mm,
           "include_suspect": not args.exclude_suspect, "depth_mode": args.depth_mode, "tsdf_virtual": args.tsdf_virtual,
           "virtual_views": args.virtual_views, "poisson_depth": args.poisson_depth}
    if args.init == "online":
        run_global_refine(args.run_dir, args.out_dir, device=args.device, config=cfg, checkpoint=args.checkpoint)
        return

    # rebuilt maps (stage-2 study): frames + poses from the online run, new runner, one training call
    run_dir = args.run_dir
    ckpt = args.checkpoint or run_dir / "gs_online" / "checkpoint_final.pt"
    out_dir = args.out_dir or run_dir / "final" / f"gs_{args.init}"
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    t0 = time.time()
    online = load_online_checkpoint(ckpt, device=args.device)
    frames = keyframes_from_run(run_dir, online)
    config = copy.deepcopy(online.config)
    config["pose_feedback"]["enabled"] = args.pose_refine
    config["pose_feedback"]["in_initial"] = args.pose_refine
    np.savetxt(out_dir / "poses_before_global.txt", online.view_poses_metric().reshape(-1, 4))
    with open(out_dir / "view_ids.txt", "w") as fh:
        fh.write("\n".join(v.frame_id for v in online.views) + "\n")
    dataset = args.dataset or ("ho3d" if "_ho3d_" in run_dir.name else "ycb")
    seq = args.seq or run_dir.name.split("_")[3]
    del online; torch.cuda.empty_cache()
    if args.init == "prior":
        runner = build_runner_from_prior(config, args.device, frames, args.steps, args.prior_root, dataset, seq)
    else:
        runner = build_runner_fresh(config, args.device, frames, args.steps)
    manifest = {"run_dir": str(run_dir), "init": args.init, "steps": args.steps, "pose_refine": args.pose_refine,
                "views": len(runner.views), "gaussians_after": runner.num_gaussians, "config": cfg}
    if runner.feedback_log:
        manifest["pose_record"] = {k: v for k, v in runner.feedback_log[-1].items() if not k.startswith("per_view")}
    np.savetxt(out_dir / "poses_after_global.txt", runner.view_poses_metric().reshape(-1, 4))
    runner.save_checkpoint(out_dir / "checkpoint_global.pt")
    full = copy.deepcopy(DEFAULT_GLOBAL_CONFIG); full.update(cfg)
    K_seq = np.loadtxt(run_dir / "cam_K.txt").reshape(3, 3) if args.tsdf_virtual else None
    manifest.update(extract_meshes(runner, out_dir, full, K_seq))
    manifest["seconds_total"] = time.time() - t0
    with open(out_dir / "global_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"[global] done in {manifest['seconds_total']:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
