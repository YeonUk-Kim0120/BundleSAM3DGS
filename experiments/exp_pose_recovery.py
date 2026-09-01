"""EXPERIMENT ⑤-2: pose-feedback perturbation-recovery study (offline).

Injects known small pose errors into incoming keyframes of a mustard0
prior+lifecycle replay and measures how much of each injected error the
clamped feedback recovers — and whether unperturbed keyframes stay put.

  python3 experiments/exp_pose_recovery.py --device cuda:1 \
    --output-dir logs/mustard0_pose_recovery_<date>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaussian_runner import (  # noqa: E402
    GaussianRunner,
    SceneNormalization,
    load_gaussian_config,
)
from run_gaussian_incremental import (  # noqa: E402
    find_latest_keyframe_snapshot,
    load_fixed_poses,
    load_frame,
)
from sam3d_prior import (  # noqa: E402
    load_mesh_prior,
    load_sam3d_gaussian_ply,
    load_sam3d_pose_or_refined,
    sample_surfels,
    transfer_gaussian_colors,
    transform_surfels_canonical_to_cv_camera,
)

TRACK = Path("/home/kist/Desktop/BundleSAM3DGS/logs/"
             "mustard0_bundlesam3dgs_exact_baseline_rematch_backprojection_20260811")
MESH = Path("/home/kist/Desktop/sam-3d-objects/output_YCBInEOAT_sam2mask_mesh")
POSE_JSON = Path("/home/kist/Desktop/BundleSAM3DGS/logs/"
                 "mustard0_sam3d_align_sam2mask_rawdepth_20260831/"
                 "sam3d_rts_refined.json")


def perturb(c2w: np.ndarray, rng: np.random.Generator,
            trans_mm: float, rot_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rot_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = direction * trans_mm / 1000.0
    return (c2w.astype(np.float64) @ T).astype(np.float32)


def pose_errors(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    trans_mm = float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000)
    cos = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return trans_mm, float(np.degrees(np.arccos(cos)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--initial-steps", type=int, default=4000)
    parser.add_argument("--update-steps", type=int, default=500)
    parser.add_argument("--test-frames", type=int, default=6)
    parser.add_argument("--inject-trans-mm", type=float, default=2.5)
    parser.add_argument("--inject-rot-deg", type=float, default=1.5)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    config = load_gaussian_config(
        "/home/kist/Desktop/BundleSAM3DGS/config_gs_2dgs_1mm_lifecycle.yml"
    )
    config["device"] = args.device
    config["pose_feedback"] = dict(config["pose_feedback"])
    config["pose_feedback"]["enabled"] = True

    frame_ids, poses = load_fixed_poses(find_latest_keyframe_snapshot(TRACK))
    K = np.loadtxt(TRACK / "cam_K.txt", dtype=np.float32).reshape(3, 3)
    n_needed = 5 + args.test_frames
    frame_ids, poses = frame_ids[:n_needed], poses[:n_needed]

    prior = load_mesh_prior(MESH / "mustard0_mesh_depth.npz")
    pose0 = load_sam3d_pose_or_refined(POSE_JSON)
    surfels = sample_surfels(prior, 20000, seed=0, radius_multiplier=0.75)
    surfels, _ = transfer_gaussian_colors(
        surfels, load_sam3d_gaussian_ply(MESH / "mustard0_splat_depth.ply")
    )
    surfels_cv = transform_surfels_canonical_to_cv_camera(surfels, pose0)
    first_c2w = poses[0].astype(np.float64)
    means_obj = (surfels_cv.means.numpy() @ first_c2w[:3, :3].T
                 + first_c2w[:3, 3][None, :])
    center = means_obj.mean(axis=0)
    radius = float(np.linalg.norm(means_obj - center, axis=1).max())
    normalization = SceneNormalization(
        scale=1.0 / max(radius * 1.2, 1e-6), translation=-center
    )
    runner = GaussianRunner(config, normalization, device=args.device)

    initial_frames = [
        load_frame(TRACK, frame_ids[i], K, poses[i]) for i in range(5)
    ]
    runner.initialize_from_prior(
        surfels_cv, poses[0], initial_frames, train_steps=args.initial_steps
    )

    rng = np.random.default_rng(7)
    rows = []
    for index in range(5, n_needed):
        true_pose = poses[index]
        bad_pose = perturb(true_pose, rng, args.inject_trans_mm,
                           args.inject_rot_deg)
        frame = load_frame(TRACK, frame_ids[index], K, bad_pose)
        runner.update([frame], train_steps=args.update_steps)
        feedback, stats = runner.get_feedback_poses()
        fb_pose = feedback[frame_ids[index]]

        injected_t, injected_r = pose_errors(bad_pose, true_pose)
        residual_t, residual_r = pose_errors(fb_pose, true_pose)
        # drift of the UNPERTURBED views (their tracker poses are true)
        drifts = []
        for view_index, view in enumerate(runner.views[:-1]):
            fb = feedback.get(view.frame_id)
            if fb is None:
                continue
            drifts.append(pose_errors(fb, poses[view_index])[0])
        rows.append({
            "frame": frame_ids[index],
            "injected_mm": injected_t, "injected_deg": injected_r,
            "residual_mm": residual_t, "residual_deg": residual_r,
            "recovery_mm_pct": 100 * (1 - residual_t / max(injected_t, 1e-9)),
            "recovery_deg_pct": 100 * (1 - residual_r / max(injected_r, 1e-9)),
            "clean_view_drift_mm_mean": float(np.mean(drifts)) if drifts else 0.0,
            "clean_view_drift_mm_max": float(np.max(drifts)) if drifts else 0.0,
            "feedback_stats": stats,
        })
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in rows[-1].items()
                          if k != "feedback_stats"}))

    summary = {
        "mean_recovery_mm_pct": float(np.mean([r["recovery_mm_pct"] for r in rows])),
        "mean_recovery_deg_pct": float(np.mean([r["recovery_deg_pct"] for r in rows])),
        "mean_residual_mm": float(np.mean([r["residual_mm"] for r in rows])),
        "mean_injected_mm": float(np.mean([r["injected_mm"] for r in rows])),
        "clean_drift_mm_max": float(np.max([r["clean_view_drift_mm_max"] for r in rows])),
    }
    with (out_dir / "result.json").open("w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)
    print("SUMMARY " + json.dumps({k: round(v, 3) for k, v in summary.items()}))


if __name__ == "__main__":
    main()
