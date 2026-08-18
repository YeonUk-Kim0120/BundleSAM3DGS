"""Continue one 5+1 operator branch through a 36-keyframe append-only replay."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from gaussian_runner import GaussianFrame, GaussianRunner
from run_gaussian_incremental import (
    INITIAL_KEYFRAMES,
    find_latest_keyframe_snapshot,
    git_metadata,
    load_fixed_poses,
    load_frame,
    save_render_artifacts,
    validate_replay_paths,
    write_json,
)


START_KEYFRAMES = INITIAL_KEYFRAMES + 1
DEFAULT_MILESTONES = (12, 18, 24, 30, 36)
METRIC_NAMES = (
    "rgb_mae",
    "rgb_psnr_db",
    "alpha_iou_at_0.5",
    "outside_alpha_mean",
    "depth_mae_mm",
    "rendered_depth_coverage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue a recovered keyframe-6 checkpoint to keyframe 36 using only "
            "RGB-D append and optimization; no manual topology events are applied"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--branch-name", required=True)
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--keyframes-yaml", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-keyframes", type=int, default=36)
    parser.add_argument("--update-steps", type=int, default=500)
    parser.add_argument(
        "--milestones", type=int, nargs="+", default=list(DEFAULT_MILESTONES)
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(items: Iterable[float | None]) -> float | None:
    finite = [float(item) for item in items if item is not None and np.isfinite(item)]
    return float(np.mean(finite)) if finite else None


def _target_depth(view: Any, frame: GaussianFrame) -> np.ndarray:
    x0, y0, x1, y1 = view.crop_xyxy
    cropped = np.ascontiguousarray(frame.depth[y0:y1, x0:x1])
    if cropped.shape != (view.height, view.width):
        raise ValueError(
            f"Depth crop for {view.frame_id} is {cropped.shape}, "
            f"expected {(view.height, view.width)}"
        )
    return cropped


def _view_metrics(
    *,
    view_index: int,
    view: Any,
    target_depth: np.ndarray,
    rendered: np.ndarray,
    alpha: np.ndarray,
    rendered_depth: np.ndarray,
) -> dict[str, Any]:
    target = view.rgb.numpy()
    mask = view.mask.numpy().astype(bool)
    rgb_errors = rendered[mask] - target[mask]
    rgb_mae = float(np.mean(np.abs(rgb_errors)))
    rgb_mse = float(np.mean(np.square(rgb_errors)))
    rgb_psnr = float(-10.0 * math.log10(max(rgb_mse, 1e-12)))

    predicted_mask = alpha >= 0.5
    union = predicted_mask | mask
    alpha_iou = (
        float(np.count_nonzero(predicted_mask & mask) / np.count_nonzero(union))
        if np.any(union)
        else 1.0
    )
    outside = ~mask
    outside_alpha = float(np.mean(alpha[outside])) if np.any(outside) else 0.0

    valid_depth = (
        mask
        & np.isfinite(target_depth)
        & (target_depth > 0.0)
        & np.isfinite(rendered_depth)
    )
    depth_mae_mm = (
        float(
            np.mean(np.abs(rendered_depth[valid_depth] - target_depth[valid_depth]))
            * 1000.0
        )
        if np.any(valid_depth)
        else None
    )
    rendered_depth_coverage = (
        float(np.mean(alpha[valid_depth] > 1e-6)) if np.any(valid_depth) else None
    )
    return {
        "view_index": view_index,
        "frame_id": view.frame_id,
        "rgb_mae": rgb_mae,
        "rgb_psnr_db": rgb_psnr,
        "alpha_iou_at_0.5": alpha_iou,
        "outside_alpha_mean": outside_alpha,
        "depth_mae_mm": depth_mae_mm,
        "rendered_depth_coverage": rendered_depth_coverage,
        "object_pixels": int(np.count_nonzero(mask)),
        "valid_depth_pixels": int(np.count_nonzero(valid_depth)),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, float | None]:
    return {name: _mean(record[name] for record in records) for name in METRIC_NAMES}


@torch.no_grad()
def evaluate_views(
    runner: GaussianRunner,
    frames_by_id: dict[str, GaussianFrame],
    view_indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    indices = (
        list(range(len(runner.views)))
        if view_indices is None
        else [int(index) for index in view_indices]
    )
    per_view: list[dict[str, Any]] = []
    for view_index in indices:
        view = runner.views[view_index]
        frame = frames_by_id[view.frame_id]
        rendered, alpha, rendered_depth = runner.render(view_index, include_depth=True)
        if rendered_depth is None:
            raise RuntimeError("Depth rendering unexpectedly returned None")
        per_view.append(
            _view_metrics(
                view_index=view_index,
                view=view,
                target_depth=_target_depth(view, frame),
                rendered=rendered,
                alpha=alpha,
                rendered_depth=rendered_depth,
            )
        )

    groups = {
        "all_views": per_view,
        "initial_five": [record for record in per_view if record["view_index"] < 5],
        "topology_view": [record for record in per_view if record["view_index"] == 5],
        "first_six": [record for record in per_view if record["view_index"] < 6],
        "post_topology_views": [
            record for record in per_view if record["view_index"] >= 6
        ],
        "latest_view": [
            record
            for record in per_view
            if record["view_index"] == len(runner.views) - 1
        ],
    }
    return {
        "definitions": {
            "aggregation": "unweighted mean of per-view metrics",
            "rgb_region": "target object mask",
            "alpha_iou_threshold": 0.5,
            "outside_alpha_region": "crop pixels outside target object mask",
            "depth_region": "fixed valid target-depth pixels inside the object mask",
            "rendered_depth_coverage": (
                "fraction of depth-region pixels with rendered alpha > 1e-6"
            ),
        },
        "groups": {
            name: _summarize(records) if records else None
            for name, records in groups.items()
        },
        "per_view": per_view,
    }


def _assert_automatic_topology_disabled(
    runner: GaussianRunner, update_steps: int
) -> None:
    strategy = dict(runner.config["strategy"])
    strategy.update(dict(runner.config["update_strategy"]))
    if int(strategy["refine_start_iter"]) < update_steps:
        raise ValueError(
            "Automatic DefaultStrategy topology must be disabled throughout each "
            "long-followup update"
        )
    if int(strategy["reset_every"]) < update_steps:
        raise ValueError(
            "Opacity reset must remain disabled throughout each long-followup update"
        )


def _assert_checkpoint_prefix(
    runner: GaussianRunner, frames: list[GaussianFrame], frame_ids: list[str]
) -> None:
    if runner.update_index != 1 or len(runner.views) != START_KEYFRAMES:
        raise ValueError(
            "Long followup requires a recovered 5+1 checkpoint at update_index 1"
        )
    if float(runner.config["voxel_size"]) != 0.001:
        raise ValueError("Long followup requires the 1 mm voxel configuration")
    if float(runner.config["novelty_distance"]) != 0.001:
        raise ValueError("Long followup requires the 1 mm novelty configuration")
    checkpoint_view_ids = [view.frame_id for view in runner.views]
    if checkpoint_view_ids != frame_ids[:START_KEYFRAMES]:
        raise ValueError(
            "Checkpoint views do not match the first six frames of the saved trajectory"
        )
    for index in range(START_KEYFRAMES):
        actual = runner.views[index]
        expected = runner._prepare_view(frames[index])
        if actual.crop_xyxy != expected.crop_xyxy:
            raise ValueError(f"Crop mismatch for checkpoint view {index}")
        for name in ("K", "c2w_normalized"):
            if not torch.allclose(
                getattr(actual, name), getattr(expected, name), rtol=0.0, atol=1e-6
            ):
                raise ValueError(f"{name} mismatch for checkpoint view {index}")
        if not torch.equal(actual.rgb, expected.rgb) or not torch.equal(
            actual.mask, expected.mask
        ):
            raise ValueError(f"RGB or mask mismatch for checkpoint view {index}")


def run(args: argparse.Namespace) -> None:
    if args.max_keyframes != 36:
        raise ValueError("This controlled followup requires exactly 36 keyframes")
    if args.update_steps <= 0:
        raise ValueError("update_steps must be positive")
    milestones = sorted(set(int(value) for value in args.milestones))
    if not milestones or milestones[-1] != args.max_keyframes:
        raise ValueError("milestones must end at max_keyframes")
    if any(value <= START_KEYFRAMES or value > args.max_keyframes for value in milestones):
        raise ValueError("milestones must be in (6, max_keyframes]")

    checkpoint = args.checkpoint.resolve()
    track_dir = args.track_dir.resolve()
    snapshot_path = (
        args.keyframes_yaml.resolve()
        if args.keyframes_yaml is not None
        else find_latest_keyframe_snapshot(track_dir).resolve()
    )
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not track_dir.is_dir():
        raise FileNotFoundError(track_dir)

    frame_ids, c2w_cv = load_fixed_poses(snapshot_path)
    frame_ids = frame_ids[: args.max_keyframes]
    c2w_cv = c2w_cv[: args.max_keyframes]
    if len(frame_ids) != args.max_keyframes:
        raise ValueError(
            f"Expected {args.max_keyframes} saved keyframes, found {len(frame_ids)}"
        )
    validate_replay_paths(track_dir, frame_ids)
    K = np.loadtxt(track_dir / "cam_K.txt", dtype=np.float32).reshape(3, 3)
    frames = [
        load_frame(track_dir, frame_id, K, c2w_cv[index])
        for index, frame_id in enumerate(frame_ids)
    ]
    frames_by_id = {frame.frame_id: frame for frame in frames}

    if args.device.startswith("cuda"):
        torch_device = torch.device(args.device)
        torch.cuda.set_device(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    runner = GaussianRunner.load_checkpoint(checkpoint, device=args.device)
    _assert_checkpoint_prefix(runner, frames, frame_ids)
    _assert_automatic_topology_disabled(runner, args.update_steps)
    output_dir.mkdir(parents=True, exist_ok=False)

    resolved_update_strategy = dict(runner.config["strategy"])
    resolved_update_strategy.update(dict(runner.config["update_strategy"]))

    manifest: dict[str, Any] = {
        "status": "running",
        "experiment": "single keyframe-6 topology event followed by append-only updates",
        "branch_name": args.branch_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "track_dir": str(track_dir),
        "keyframes_yaml": str(snapshot_path),
        "keyframes_yaml_sha256": _sha256(snapshot_path),
        "device": args.device,
        "start_keyframes": START_KEYFRAMES,
        "max_keyframes": args.max_keyframes,
        "update_steps": args.update_steps,
        "start_total_steps": runner.total_steps,
        "start_strategy_step": runner.strategy_step,
        "start_gaussians": runner.num_gaussians,
        "start_observed_points": int(len(runner.observed_points_metric)),
        "manual_topology_after_start": False,
        "automatic_topology": False,
        "resolved_update_strategy": resolved_update_strategy,
        "milestones": milestones,
        "frame_ids": frame_ids,
        "git": git_metadata(),
        "updates": [],
    }
    write_json(output_dir / "experiment_manifest.json", manifest)
    experiment_start = time.perf_counter()
    try:
        start_metrics = evaluate_views(runner, frames_by_id)
        write_json(output_dir / "metrics_kf006.json", start_metrics)
        save_render_artifacts(runner, 5, output_dir, "start_kf006")

        for frame_count in range(START_KEYFRAMES + 1, args.max_keyframes + 1):
            frame = frames[frame_count - 1]
            update_start = time.perf_counter()
            expected_total_steps = runner.total_steps + args.update_steps
            expected_update_index = runner.update_index + 1
            expected_view_count = len(runner.views) + 1
            update = runner.update([frame], train_steps=args.update_steps)
            optimization_seconds = time.perf_counter() - update_start
            if (
                runner.num_gaussians != update.gaussians_after_append
                or update.gaussians_after_train != update.gaussians_after_append
            ):
                raise RuntimeError(
                    "Gaussian count changed during an append-only optimization update"
                )
            if runner.strategy_step != args.update_steps:
                raise RuntimeError(
                    "Unexpected strategy step after append-only optimization update"
                )
            if runner.total_steps != expected_total_steps:
                raise RuntimeError("Unexpected total step count after update")
            if runner.update_index != expected_update_index:
                raise RuntimeError("Unexpected update index after update")
            if len(runner.views) != expected_view_count:
                raise RuntimeError("Unexpected training-view count after update")
            if runner.views[-1].frame_id != frame.frame_id:
                raise RuntimeError("The appended training view has the wrong frame ID")

            if frame_count in milestones:
                metrics = evaluate_views(runner, frames_by_id)
                write_json(output_dir / f"metrics_kf{frame_count:03d}.json", metrics)
                runner.save_checkpoint(output_dir / f"checkpoint_kf{frame_count:03d}.pt")
                save_render_artifacts(
                    runner,
                    len(runner.views) - 1,
                    output_dir,
                    f"milestone_kf{frame_count:03d}",
                )
                latest_metrics = metrics["groups"]["latest_view"]
            else:
                latest = evaluate_views(
                    runner, frames_by_id, [len(runner.views) - 1]
                )
                latest_metrics = latest["groups"]["latest_view"]

            record = {
                "frame_count": frame_count,
                "frame_id": frame.frame_id,
                "update": update.to_dict(),
                "latest_view_metrics": latest_metrics,
                "milestone": frame_count in milestones,
                "total_steps": runner.total_steps,
                "strategy_step": runner.strategy_step,
                "observed_points": int(len(runner.observed_points_metric)),
                "optimization_seconds": optimization_seconds,
                "elapsed_seconds": time.perf_counter() - update_start,
            }
            manifest["updates"].append(record)
            write_json(output_dir / "experiment_manifest.json", manifest)
            print(
                {
                    "branch": args.branch_name,
                    "frame_count": frame_count,
                    "novel_points": update.novel_points,
                    "gaussians": runner.num_gaussians,
                    "latest_rgb_mae": latest_metrics["rgb_mae"],
                    "latest_depth_mae_mm": latest_metrics["depth_mae_mm"],
                },
                flush=True,
            )

        runner.export_ply(
            output_dir / "splats_normalized.ply",
            output_dir / "splats_metric.ply",
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_type"] = type(exc).__name__
        manifest["failure_message"] = str(exc)
        write_json(output_dir / "experiment_manifest.json", manifest)
        raise

    manifest["status"] = "complete"
    manifest["gaussians_final"] = runner.num_gaussians
    manifest["elapsed_seconds"] = time.perf_counter() - experiment_start
    if args.device.startswith("cuda"):
        manifest["peak_cuda_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(torch.device(args.device))
        )
    write_json(output_dir / "experiment_manifest.json", manifest)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
