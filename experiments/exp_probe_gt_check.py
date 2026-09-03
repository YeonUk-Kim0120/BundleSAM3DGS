"""EXPERIMENT ⑤-probe-GT: is the pose-only probe's converged offset REAL?

The frozen-map probe (exp_pose_only_probe) converges to ~0.7 deg away from
the tracker pose on a held-out keyframe regardless of injection. Two
readings: (a) map bias — the offset is noise the feedback would inject into
a healthy tracker; (b) genuine tracker error — the feedback found it.

This script decides between them with GT: run the probe with ZERO injection
on the tracker's own pose, then compare tracker-vs-GT and corrected-vs-GT
errors under the milestone-4 first-frame alignment convention
(aligned_pred_i = pred_i @ inv(pred_0) @ gt_0, ob_in_cam matrices).

  python3 experiments/exp_probe_gt_check.py --device cuda:1 \
    --output-dir logs/mustard0_probe_gtcheck_<date>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_pose_only_probe import (  # noqa: E402
    MESH,
    POSE_JSON,
    TRACK,
    run_probe,
)
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

DATA = Path("/home/kist/Desktop/BundleSAM3DGS/datasets/YCBInEOAT/mustard0")


def rot_trans_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    trans_mm = float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000)
    cos = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return trans_mm, float(np.degrees(np.arccos(cos)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--initial-steps", type=int, default=4000)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--lr-rot", type=float, default=2e-3)
    parser.add_argument("--lr-trans", type=float, default=5e-4)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    frame_ids, poses = load_fixed_poses(find_latest_keyframe_snapshot(TRACK))
    K = np.loadtxt(TRACK / "cam_K.txt", dtype=np.float32).reshape(3, 3)
    n_init = 5
    frame_ids, poses = frame_ids[: n_init + 1], poses[: n_init + 1]

    # frame_id (timestamp) -> dataset frame index -> GT ob_in_cam
    color_names = sorted(p.stem for p in (DATA / "rgb").glob("*.png"))
    gt_files = sorted((DATA / "annotated_poses").glob("*"))
    index_of = {name: i for i, name in enumerate(color_names)}

    def gt_ob_in_cam(frame_id: str) -> np.ndarray:
        return np.loadtxt(gt_files[index_of[frame_id]]).reshape(4, 4)

    # Milestone-4 alignment anchor: pred_i @ inv(pred_0) @ gt_0
    pred_ob_in_cam = [np.linalg.inv(p.astype(np.float64)) for p in poses]
    anchor = np.linalg.inv(pred_ob_in_cam[0]) @ gt_ob_in_cam(frame_ids[0])

    tracker_rows = []
    for i in range(n_init + 1):
        aligned = pred_ob_in_cam[i] @ anchor
        trans_mm, rot_deg = rot_trans_error(aligned, gt_ob_in_cam(frame_ids[i]))
        tracker_rows.append({
            "kf": i, "frame": frame_ids[i],
            "tracker_vs_gt_mm": trans_mm, "tracker_vs_gt_deg": rot_deg,
        })
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in tracker_rows[-1].items()}), flush=True)

    config = load_gaussian_config(
        "/home/kist/Desktop/BundleSAM3DGS/config_gs_2dgs_1mm_lifecycle.yml"
    )
    config["device"] = args.device
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
        load_frame(TRACK, frame_ids[i], K, poses[i]) for i in range(n_init)
    ]
    runner.initialize_from_prior(
        surfels_cv, poses[0], initial_frames, train_steps=args.initial_steps
    )

    scale = float(runner.normalization.scale)
    translation = np.asarray(runner.normalization.translation, dtype=np.float64)
    held_out_frame = load_frame(TRACK, frame_ids[n_init], K, poses[n_init])
    probe_targets = {
        "in_map_kf4": (runner.views[n_init - 1], n_init - 1),
        "held_out_kf5": (runner._prepare_view(held_out_frame), n_init),
    }

    probe_rows = []
    for name, (view, kf_index) in probe_targets.items():
        true_n = torch.from_numpy(
            runner.normalization.normalize_c2w(poses[kf_index])
        ).to(torch.float32)
        probe = run_probe(
            runner, view, true_n,
            steps=args.probe_steps,
            lr_rot=args.lr_rot, lr_trans=args.lr_trans,
        )
        corrected_n = probe["corrected_n"].cpu().numpy().astype(np.float64)
        corrected_metric = corrected_n.copy()
        corrected_metric[:3, 3] = corrected_n[:3, 3] / scale - translation
        aligned_corr = np.linalg.inv(corrected_metric) @ anchor
        aligned_track = pred_ob_in_cam[kf_index] @ anchor
        gt = gt_ob_in_cam(frame_ids[kf_index])
        corr_mm, corr_deg = rot_trans_error(aligned_corr, gt)
        track_mm, track_deg = rot_trans_error(aligned_track, gt)
        probe_rows.append({
            "target": name,
            "tracker_vs_gt_mm": track_mm, "tracker_vs_gt_deg": track_deg,
            "corrected_vs_gt_mm": corr_mm, "corrected_vs_gt_deg": corr_deg,
            "applied_rot_deg": probe["delta_rot_deg"],
            "applied_trans_mm": probe["delta_trans_mm"],
            "final_loss": probe["final_loss"],
        })
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in probe_rows[-1].items()}), flush=True)

    with (out_dir / "result.json").open("w") as f:
        json.dump({
            "tracker_rows": tracker_rows,
            "probe_rows": probe_rows,
            "settings": vars(args) | {"output_dir": str(out_dir)},
        }, f, indent=2)


if __name__ == "__main__":
    main()
