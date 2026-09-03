"""EXPERIMENT ⑤-v4: design-C simulation — refine-before-append recovery.

Simulates milestone-5 design C entirely offline, main code untouched: each
incoming keyframe carries an injected OUTLIER-scale pose error (5 mm /
3 deg, the tracker-rematch scale); arm "corrected" first refines that pose
against the FROZEN current map (pose-only probe, corrected se3_exp), clamps
the correction to the online safety limits (3 mm / 3 deg per cycle), and
appends + jointly trains with the corrected pose. Arm "uncorrected" appends
the erroneous pose directly (what the pipeline does today).

There is no joint pose optimization anywhere, so clean-view drift is zero
by construction — the failure mode of attempt 1 is structurally absent.

Per-frame metrics: residual vs the true tracker pose after correction and
recovery %. Per-arm end state: a zero-injection probe on the next unseen
keyframe (the correction the map "wants" — a proxy for accumulated map
contamination) and the joint-training losses.

  python3 experiments/exp_c_sim_recovery.py --device cuda:1 \
    --output-dir logs/mustard0_c_sim_recovery_<date>
"""

from __future__ import annotations

import argparse
import json
import math
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
    perturbation,
    run_probe,
    se3_exp,
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

CLAMP_TRANS_M = 0.003
CLAMP_ROT_DEG = 3.0


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
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--lr-rot", type=float, default=2e-3)
    parser.add_argument("--lr-trans", type=float, default=5e-4)
    parser.add_argument("--test-frames", type=int, default=6)
    parser.add_argument("--inject-trans-mm", type=float, default=5.0)
    parser.add_argument("--inject-rot-deg", type=float, default=3.0)
    parser.add_argument("--zero-trans", action="store_true",
                        help="rotation-only feedback: discard the "
                             "translation part of the probe correction")
    parser.add_argument("--arms", choices=("both", "corrected", "uncorrected"),
                        default="both")
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    frame_ids, poses = load_fixed_poses(find_latest_keyframe_snapshot(TRACK))
    K = np.loadtxt(TRACK / "cam_K.txt", dtype=np.float32).reshape(3, 3)
    n_init = 5
    n_needed = n_init + args.test_frames + 1  # +1 for the end-state probe
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

    def build_runner() -> GaussianRunner:
        config = load_gaussian_config(
            "/home/kist/Desktop/BundleSAM3DGS/config_gs_2dgs_1mm_lifecycle.yml"
        )
        config["device"] = args.device
        normalization = SceneNormalization(
            scale=1.0 / max(radius * 1.2, 1e-6), translation=-center
        )
        runner = GaussianRunner(config, normalization, device=args.device)
        initial_frames = [
            load_frame(TRACK, frame_ids[i], K, poses[i]) for i in range(n_init)
        ]
        runner.initialize_from_prior(
            surfels_cv, poses[0], initial_frames,
            train_steps=args.initial_steps,
        )
        return runner

    arm_list = (
        ("uncorrected", "corrected") if args.arms == "both" else (args.arms,)
    )
    results = {}
    for arm in arm_list:
        runner = build_runner()
        scale = float(runner.normalization.scale)
        translation = np.asarray(
            runner.normalization.translation, dtype=np.float64
        )
        rng = np.random.default_rng(11)  # same injections for both arms
        rows = []
        for index in range(n_init, n_init + args.test_frames):
            true_pose = poses[index]
            T = perturbation(rng, args.inject_trans_mm, args.inject_rot_deg)
            bad_pose = (true_pose.astype(np.float64) @ T).astype(np.float32)
            use_pose = bad_pose
            clipped = False
            probe_stats = None
            if arm == "corrected":
                frame_bad = load_frame(TRACK, frame_ids[index], K, bad_pose)
                view = runner._prepare_view(frame_bad)
                bad_n = torch.from_numpy(
                    runner.normalization.normalize_c2w(bad_pose)
                ).to(torch.float32)
                probe = run_probe(
                    runner, view, bad_n,
                    steps=args.probe_steps,
                    lr_rot=args.lr_rot, lr_trans=args.lr_trans,
                )
                rot = probe["rot"].to(torch.float64)
                trans = probe["trans"].to(torch.float64)
                rot_norm = float(torch.linalg.norm(rot))
                max_rot = math.radians(CLAMP_ROT_DEG)
                if rot_norm > max_rot:
                    rot = rot * (max_rot / rot_norm)
                    clipped = True
                if args.zero_trans:
                    trans = torch.zeros_like(trans)
                else:
                    trans_norm = float(torch.linalg.norm(trans))
                    max_trans = CLAMP_TRANS_M * scale
                    if trans_norm > max_trans:
                        trans = trans * (max_trans / trans_norm)
                        clipped = True
                corrected_n = (
                    bad_n.to(torch.float64)
                    @ se3_exp(torch.cat((rot, trans)))
                ).numpy()
                corrected_metric = corrected_n.copy()
                corrected_metric[:3, 3] = (
                    corrected_n[:3, 3] / scale - translation
                )
                use_pose = corrected_metric.astype(np.float32)
                probe_stats = {
                    "delta_rot_deg": probe["delta_rot_deg"],
                    "delta_trans_mm": probe["delta_trans_mm"],
                    "probe_loss": probe["final_loss"],
                }
            injected_t, injected_r = pose_errors(bad_pose, true_pose)
            residual_t, residual_r = pose_errors(use_pose, true_pose)
            stats = runner.update(
                [load_frame(TRACK, frame_ids[index], K, use_pose)],
                train_steps=args.update_steps,
            )
            rows.append({
                "frame": frame_ids[index],
                "injected_mm": injected_t, "injected_deg": injected_r,
                "residual_mm": residual_t, "residual_deg": residual_r,
                "recovery_mm_pct": 100 * (1 - residual_t / injected_t),
                "recovery_deg_pct": 100 * (1 - residual_r / injected_r),
                "clipped": clipped,
                "probe": probe_stats,
                "update_final_loss": stats.final_loss,
            })
            print(f"[{arm}] " + json.dumps({
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in rows[-1].items() if k != "probe"
            }), flush=True)

        # End-state probe: zero-injection on the next unseen keyframe. The
        # correction the map asks for on a CLEAN pose is a proxy for how
        # much tracker/map disagreement the arm accumulated.
        end_index = n_init + args.test_frames
        end_frame = load_frame(TRACK, frame_ids[end_index], K, poses[end_index])
        end_view = runner._prepare_view(end_frame)
        end_true_n = torch.from_numpy(
            runner.normalization.normalize_c2w(poses[end_index])
        ).to(torch.float32)
        end_probe = run_probe(
            runner, end_view, end_true_n,
            steps=args.probe_steps,
            lr_rot=args.lr_rot, lr_trans=args.lr_trans,
        )
        end_state = {
            "wanted_rot_deg": end_probe["delta_rot_deg"],
            "wanted_trans_mm": end_probe["delta_trans_mm"],
            "probe_loss": end_probe["final_loss"],
        }
        summary = {
            "mean_recovery_mm_pct": float(
                np.mean([r["recovery_mm_pct"] for r in rows])
            ),
            "mean_recovery_deg_pct": float(
                np.mean([r["recovery_deg_pct"] for r in rows])
            ),
            "mean_residual_mm": float(np.mean([r["residual_mm"] for r in rows])),
            "mean_residual_deg": float(
                np.mean([r["residual_deg"] for r in rows])
            ),
            "mean_update_final_loss": float(
                np.mean([r["update_final_loss"] for r in rows])
            ),
            "end_state": end_state,
        }
        results[arm] = {"rows": rows, "summary": summary}
        print(f"SUMMARY[{arm}] " + json.dumps({
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in summary.items() if k != "end_state"
        }) + " end_state=" + json.dumps({
            k: round(v, 3) for k, v in end_state.items()
        }), flush=True)
        del runner
        torch.cuda.empty_cache()

    with (out_dir / "result.json").open("w") as f:
        json.dump({
            "arms": results,
            "settings": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            } | {
                "clamp_trans_m": CLAMP_TRANS_M,
                "clamp_rot_deg": CLAMP_ROT_DEG,
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
