"""Stress-test one exact-rank topology policy after every RGB-D keyframe append."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import torch

from gaussian_budget_strategy import (
    TOPOLOGY_MODES,
    TopologyMode,
    apply_budgeted_topology,
)
from gaussian_runner import GaussianFrame, GaussianRunner
from run_gaussian_incremental import (
    INITIAL_KEYFRAMES,
    find_latest_keyframe_snapshot,
    git_metadata,
    load_fixed_poses,
    load_frame,
    validate_replay_paths,
    write_json,
)
from run_gaussian_long_followup import METRIC_NAMES, evaluate_views


EXPECTED_START_GAUSSIANS = 13_345
DEFAULT_MILESTONES = (6, 12, 18, 24, 30, 36)
MODE_NAMES = ("none", *TOPOLOGY_MODES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay keyframes 6-36 from one five-keyframe checkpoint, applying "
            "one controlled topology event after every 201-step prefix"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--branch-name", required=True)
    parser.add_argument("--topology-mode", choices=MODE_NAMES, required=True)
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--keyframes-yaml", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-keyframes", type=int, default=36)
    parser.add_argument("--budget-fraction", type=float, default=0.03)
    parser.add_argument("--prefix-steps", type=int, default=201)
    parser.add_argument("--recovery-steps", type=int, default=500)
    parser.add_argument("--lr-decay-horizon-steps", type=int, default=701)
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


def _assert_automatic_mutation_disabled(
    runner: GaussianRunner, total_update_steps: int
) -> dict[str, Any]:
    strategy = dict(runner.config["strategy"])
    strategy.update(dict(runner.config["update_strategy"]))
    if int(strategy["refine_start_iter"]) < total_update_steps:
        raise ValueError(
            "Automatic DefaultStrategy topology must remain disabled throughout "
            "each controlled update"
        )
    if int(strategy["reset_every"]) < total_update_steps:
        raise ValueError(
            "Automatic opacity reset must remain disabled throughout each "
            "controlled update"
        )
    return strategy


def _assert_checkpoint_prefix(
    runner: GaussianRunner, frames: list[GaussianFrame], frame_ids: list[str]
) -> None:
    if runner.update_index != 0 or len(runner.views) != INITIAL_KEYFRAMES:
        raise ValueError(
            "Repeated topology requires the common five-keyframe initial checkpoint"
        )
    if runner.num_gaussians != EXPECTED_START_GAUSSIANS:
        raise ValueError(
            f"Expected {EXPECTED_START_GAUSSIANS} initial Gaussians, "
            f"found {runner.num_gaussians}"
        )
    if runner.total_steps != 651 or runner.strategy_step != 651:
        raise ValueError("Expected the common checkpoint after exactly 651 initial steps")
    if float(runner.config["voxel_size"]) != 0.001:
        raise ValueError("Repeated topology requires the 1 mm voxel configuration")
    if float(runner.config["novelty_distance"]) != 0.001:
        raise ValueError("Repeated topology requires the 1 mm novelty configuration")
    if [view.frame_id for view in runner.views] != frame_ids[:INITIAL_KEYFRAMES]:
        raise ValueError("Checkpoint views do not match the first five saved frames")

    for index in range(INITIAL_KEYFRAMES):
        actual = runner.views[index]
        expected = runner._prepare_view(frames[index])
        if actual.crop_xyxy != expected.crop_xyxy:
            raise ValueError(f"Crop mismatch for checkpoint view {index}")
        for name in ("K", "c2w_normalized"):
            if not torch.allclose(
                getattr(actual, name), getattr(expected, name), rtol=0.0, atol=1e-6
            ):
                raise ValueError(f"{name} mismatch for checkpoint view {index}")
        if not torch.equal(actual.rgb, expected.rgb):
            raise ValueError(f"RGB mismatch for checkpoint view {index}")
        if not torch.equal(actual.mask, expected.mask):
            raise ValueError(f"Mask mismatch for checkpoint view {index}")


def _latest_summary(metrics: dict[str, Any]) -> dict[str, float | None]:
    latest = metrics["groups"]["latest_view"]
    if latest is None:
        raise RuntimeError("Latest-view metric group is unexpectedly empty")
    return latest


def _metric_delta(
    after: dict[str, float | None], before: dict[str, float | None]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        lhs = after[name]
        rhs = before[name]
        result[name] = None if lhs is None or rhs is None else float(lhs - rhs)
    return result


def _group_deltas(
    after: dict[str, Any], before: dict[str, Any]
) -> dict[str, dict[str, float | None] | None]:
    result: dict[str, dict[str, float | None] | None] = {}
    for group_name, before_summary in before["groups"].items():
        after_summary = after["groups"].get(group_name)
        result[group_name] = (
            None
            if before_summary is None or after_summary is None
            else _metric_delta(after_summary, before_summary)
        )
    return result


def _evaluate_stage(
    runner: GaussianRunner,
    frames_by_id: dict[str, GaussianFrame],
    *,
    full: bool,
) -> dict[str, Any]:
    indices: Iterable[int] | None = None if full else [len(runner.views) - 1]
    return evaluate_views(runner, frames_by_id, indices)


def _assert_topology_event(
    event: Any,
    *,
    mode: TopologyMode,
    protected_start: int,
) -> None:
    expected = event.max_replacements
    if event.prune_count != expected or len(event.grow_indices) != expected:
        raise RuntimeError(
            "The event could not apply the requested exact-rank replacement budget"
        )
    if any(index >= protected_start for index in event.prune_indices):
        raise RuntimeError("A newly appended Gaussian was selected for pruning")
    expected_counts = {
        "prune_only": (0, 0),
        "duplicate_only": (expected, 0),
        "split_only": (0, expected),
        "mixed": ((expected + 1) // 2, expected // 2),
    }
    if (event.duplicate_count, event.split_count) != expected_counts[mode]:
        raise RuntimeError(f"Unexpected {mode} operator partition")


def _validate_args(args: argparse.Namespace) -> tuple[list[int], int]:
    if args.max_keyframes != 36:
        raise ValueError("This controlled experiment requires exactly 36 keyframes")
    if not np.isfinite(args.budget_fraction) or not 0.0 < args.budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be finite and in (0, 1]")
    if args.prefix_steps <= 0 or args.recovery_steps <= 0:
        raise ValueError("prefix_steps and recovery_steps must be positive")
    total_update_steps = args.prefix_steps + args.recovery_steps
    if args.lr_decay_horizon_steps != total_update_steps:
        raise ValueError(
            "lr_decay_horizon_steps must equal prefix_steps + recovery_steps"
        )
    milestones = sorted(set(int(value) for value in args.milestones))
    if not milestones or milestones[-1] != args.max_keyframes:
        raise ValueError("milestones must end at max_keyframes")
    if any(
        value <= INITIAL_KEYFRAMES or value > args.max_keyframes
        for value in milestones
    ):
        raise ValueError("milestones must be in (5, max_keyframes]")
    return milestones, total_update_steps


def run(args: argparse.Namespace) -> None:
    milestones, total_update_steps = _validate_args(args)
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
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    if output_dir.exists():
        raise FileExistsError(output_dir)

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
    runner = GaussianRunner.load_checkpoint(checkpoint, device=args.device)
    _assert_checkpoint_prefix(runner, frames, frame_ids)
    resolved_strategy = _assert_automatic_mutation_disabled(
        runner, total_update_steps
    )
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))

    mode = (
        None
        if args.topology_mode == "none"
        else cast(TopologyMode, args.topology_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "status": "running",
        "experiment": "exact-rank topology after every RGB-D append",
        "branch_name": args.branch_name,
        "topology_mode": mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "track_dir": str(track_dir),
        "keyframes_yaml": str(snapshot_path),
        "keyframes_yaml_sha256": _sha256(snapshot_path),
        "device": args.device,
        "start_keyframes": INITIAL_KEYFRAMES,
        "max_keyframes": args.max_keyframes,
        "budget_fraction": args.budget_fraction,
        "prefix_steps": args.prefix_steps,
        "recovery_steps": args.recovery_steps,
        "lr_decay_horizon_steps": args.lr_decay_horizon_steps,
        "manual_topology_every_update": mode is not None,
        "new_append_tail_protected_from_same_event_prune": True,
        "automatic_topology": False,
        "automatic_opacity_reset": False,
        "resolved_update_strategy": resolved_strategy,
        "start_total_steps": runner.total_steps,
        "start_strategy_step": runner.strategy_step,
        "start_gaussians": runner.num_gaussians,
        "start_observed_points": int(len(runner.observed_points_metric)),
        "milestones": milestones,
        "frame_ids": frame_ids,
        "git": git_metadata(),
        "updates": [],
    }
    write_json(output_dir / "experiment_manifest.json", manifest)
    experiment_start = time.perf_counter()

    try:
        start_metrics = evaluate_views(runner, frames_by_id)
        write_json(output_dir / "metrics_start_kf005.json", start_metrics)

        for frame_count in range(INITIAL_KEYFRAMES + 1, args.max_keyframes + 1):
            frame = frames[frame_count - 1]
            update_start = time.perf_counter()
            count_before_append = runner.num_gaussians
            observed_before = int(len(runner.observed_points_metric))
            expected_total_steps = runner.total_steps + total_update_steps
            expected_update_index = runner.update_index + 1
            expected_view_count = len(runner.views) + 1

            append_start = time.perf_counter()
            append = runner.update([frame], train_steps=0)
            append_seconds = time.perf_counter() - append_start
            protected_start = append.gaussians_before
            count_after_append = runner.num_gaussians
            observed_after_append = runner.observed_points_metric.copy()
            if append.gaussians_before != count_before_append:
                raise RuntimeError("Append reported the wrong starting Gaussian count")
            if append.gaussians_after_append != count_after_append:
                raise RuntimeError("Append reported the wrong final Gaussian count")
            if append.gaussians_after_train != count_after_append:
                raise RuntimeError("Zero-step append unexpectedly changed Gaussian count")
            if count_after_append != count_before_append + append.novel_points:
                raise RuntimeError("Novel-point and appended-Gaussian counts disagree")
            if runner.strategy_step != 0:
                raise RuntimeError("Append did not reset the update strategy step")
            if runner.update_index != expected_update_index:
                raise RuntimeError("Unexpected reconstruction update index")
            if len(runner.views) != expected_view_count:
                raise RuntimeError("Unexpected training-view count after append")
            if runner.views[-1].frame_id != frame.frame_id:
                raise RuntimeError("The appended view has the wrong frame ID")

            prefix_start = time.perf_counter()
            prefix_first_loss, prefix_final_loss = runner.train(
                args.prefix_steps,
                lr_decay_horizon_steps=args.lr_decay_horizon_steps,
            )
            prefix_seconds = time.perf_counter() - prefix_start
            if runner.strategy_step != args.prefix_steps:
                raise RuntimeError("Unexpected strategy step after prefix training")
            if runner.num_gaussians != count_after_append:
                raise RuntimeError("Automatic topology changed count during prefix")

            is_milestone = frame_count in milestones
            metrics_before = _evaluate_stage(
                runner, frames_by_id, full=is_milestone
            )
            latest_before = _latest_summary(metrics_before)
            if is_milestone:
                write_json(
                    output_dir / f"metrics_kf{frame_count:03d}_before_event.json",
                    metrics_before,
                )

            event_start = time.perf_counter()
            event = None
            if mode is not None:
                event = apply_budgeted_topology(
                    runner.splats,
                    runner.optimizers,
                    runner.strategy_state,
                    fraction=args.budget_fraction,
                    protected_start=protected_start,
                    mode=mode,
                )
                _assert_topology_event(
                    event, mode=mode, protected_start=protected_start
                )
            event_seconds = time.perf_counter() - event_start
            count_after_event = runner.num_gaussians
            expected_after_event = (
                count_after_append
                if mode != "prune_only"
                else count_after_append - event.prune_count
            )
            if count_after_event != expected_after_event:
                raise RuntimeError("Topology event produced an unexpected Gaussian count")

            metrics_after = _evaluate_stage(
                runner, frames_by_id, full=is_milestone
            )
            latest_after = _latest_summary(metrics_after)
            if is_milestone:
                write_json(
                    output_dir / f"metrics_kf{frame_count:03d}_after_event.json",
                    metrics_after,
                )

            recovery_start = time.perf_counter()
            recovery_first_loss, recovery_final_loss = runner.train(
                args.recovery_steps,
                lr_decay_horizon_steps=args.lr_decay_horizon_steps,
            )
            recovery_seconds = time.perf_counter() - recovery_start
            if runner.strategy_step != total_update_steps:
                raise RuntimeError("Unexpected strategy step after recovery")
            if runner.total_steps != expected_total_steps:
                raise RuntimeError("Unexpected total training-step count")
            if runner.num_gaussians != count_after_event:
                raise RuntimeError("Automatic topology changed count during recovery")
            if not np.array_equal(
                runner.observed_points_metric, observed_after_append
            ):
                raise RuntimeError("Training or topology changed observed-point history")

            metrics_recovery = _evaluate_stage(
                runner, frames_by_id, full=is_milestone
            )
            latest_recovery = _latest_summary(metrics_recovery)
            if is_milestone:
                write_json(
                    output_dir / f"metrics_kf{frame_count:03d}_after_recovery.json",
                    metrics_recovery,
                )
                runner.save_checkpoint(
                    output_dir / f"checkpoint_kf{frame_count:03d}.pt"
                )

            record = {
                "frame_count": frame_count,
                "frame_id": frame.frame_id,
                "append": append.to_dict(),
                "protected_start": protected_start,
                "observed_points_before": observed_before,
                "observed_points_after": int(len(runner.observed_points_metric)),
                "prefix": {
                    "steps": args.prefix_steps,
                    "first_loss": prefix_first_loss,
                    "final_loss": prefix_final_loss,
                },
                "event": None if event is None else event.to_dict(),
                "recovery": {
                    "steps": args.recovery_steps,
                    "first_loss": recovery_first_loss,
                    "final_loss": recovery_final_loss,
                },
                "gaussians_before_append": count_before_append,
                "gaussians_after_append": count_after_append,
                "gaussians_after_event": count_after_event,
                "gaussians_after_recovery": runner.num_gaussians,
                "latest_view_metrics": {
                    "before_event": latest_before,
                    "after_event": latest_after,
                    "after_recovery": latest_recovery,
                },
                "latest_view_deltas": {
                    "event_shock": _metric_delta(latest_after, latest_before),
                    "recovery_from_event": _metric_delta(
                        latest_recovery, latest_after
                    ),
                    "net_after_recovery": _metric_delta(
                        latest_recovery, latest_before
                    ),
                },
                "milestone": is_milestone,
                "milestone_group_deltas": (
                    {
                        "event_shock": _group_deltas(
                            metrics_after, metrics_before
                        ),
                        "net_after_recovery": _group_deltas(
                            metrics_recovery, metrics_before
                        ),
                    }
                    if is_milestone
                    else None
                ),
                "total_steps": runner.total_steps,
                "strategy_step": runner.strategy_step,
                "timing_seconds": {
                    "append": append_seconds,
                    "prefix": prefix_seconds,
                    "event": event_seconds,
                    "recovery": recovery_seconds,
                    "total": time.perf_counter() - update_start,
                },
            }
            manifest["updates"].append(record)
            write_json(output_dir / "experiment_manifest.json", manifest)
            print(
                {
                    "branch": args.branch_name,
                    "frame_count": frame_count,
                    "novel_points": append.novel_points,
                    "event_count": 0 if event is None else event.prune_count,
                    "gaussians": runner.num_gaussians,
                    "latest_rgb_mae": latest_recovery["rgb_mae"],
                    "latest_depth_mae_mm": latest_recovery["depth_mae_mm"],
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
    manifest["observed_points_final"] = int(len(runner.observed_points_metric))
    manifest["total_steps_final"] = runner.total_steps
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
