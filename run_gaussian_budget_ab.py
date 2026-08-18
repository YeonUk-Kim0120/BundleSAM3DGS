"""Controlled A/B replay for one count-neutral Gaussian topology event."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gaussian_budget_strategy import apply_budgeted_topology
from gaussian_runner import GaussianRunner, load_gaussian_config
from run_gaussian_incremental import (
    INITIAL_KEYFRAMES,
    REPO_ROOT,
    _dump_yaml,
    compute_initial_normalization,
    find_latest_keyframe_snapshot,
    git_metadata,
    load_fixed_poses,
    load_frame,
    save_render_artifacts,
    validate_replay_paths,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Branch one fixed-pose 5+1 replay checkpoint into append-only and "
            "count-neutral topology variants"
        )
    )
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config_gs_1mm_append_only.yml",
    )
    parser.add_argument("--keyframes-yaml", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--initial-steps", type=int, default=651)
    parser.add_argument("--prefix-steps", type=int, default=201)
    parser.add_argument("--recovery-steps", type=int, default=250)
    parser.add_argument("--recovery-snapshot-steps", type=int, default=50)
    parser.add_argument("--budget-fraction", type=float, default=0.02)
    return parser.parse_args()


def _current_means_lr(runner: GaussianRunner) -> float:
    return float(runner.optimizers["means"].param_groups[0]["lr"])


def _assert_topology_disabled(config: dict[str, Any], total_update_steps: int) -> None:
    update_strategy = dict(config["strategy"])
    update_strategy.update(dict(config["update_strategy"]))
    if int(update_strategy["refine_start_iter"]) < total_update_steps:
        raise ValueError(
            "The A/B driver requires automatic update topology to be disabled; "
            "set update_strategy.refine_start_iter beyond the update horizon"
        )


def _run_branch(
    *,
    name: str,
    common_checkpoint: Path,
    output_dir: Path,
    device: str,
    protected_start: int,
    budget_fraction: float,
    lr_decay_horizon_steps: int,
    recovery_steps: int,
    recovery_snapshot_steps: int,
    apply_topology: bool,
) -> dict[str, Any]:
    branch_dir = output_dir / name
    branch_dir.mkdir()
    runner = GaussianRunner.load_checkpoint(common_checkpoint, device=device)
    count_before = runner.num_gaussians
    lr_before = _current_means_lr(runner)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    branch_start = time.perf_counter()
    save_render_artifacts(runner, len(runner.views) - 1, branch_dir, "before_event_kf006")

    event_stats = None
    if apply_topology:
        event_stats = apply_budgeted_topology(
            runner.splats,
            runner.optimizers,
            runner.strategy_state,
            fraction=budget_fraction,
            protected_start=protected_start,
        )
        write_json(branch_dir / "topology_event.json", event_stats.to_dict())
        runner.save_checkpoint(branch_dir / "checkpoint_after_event.pt")
    if runner.num_gaussians != count_before:
        raise RuntimeError(f"{name} changed Gaussian count at the branch point")
    save_render_artifacts(runner, len(runner.views) - 1, branch_dir, "after_event_kf006")

    first_segment = min(recovery_snapshot_steps, recovery_steps)
    first_loss, snapshot_loss = runner.train(
        first_segment,
        lr_decay_horizon_steps=lr_decay_horizon_steps,
    )
    runner.save_checkpoint(branch_dir / f"checkpoint_after_{first_segment:03d}.pt")
    save_render_artifacts(
        runner,
        len(runner.views) - 1,
        branch_dir,
        f"after_{first_segment:03d}_kf006",
    )
    remaining_steps = recovery_steps - first_segment
    remaining_first_loss, final_loss = runner.train(
        remaining_steps,
        lr_decay_horizon_steps=lr_decay_horizon_steps,
    )
    if remaining_steps == 0:
        final_loss = snapshot_loss
    runner.save_checkpoint(branch_dir / "checkpoint_final.pt")
    runner.export_ply(
        branch_dir / "splats_normalized.ply",
        branch_dir / "splats_metric.ply",
    )
    save_render_artifacts(runner, len(runner.views) - 1, branch_dir, "final_kf006")
    branch_record = {
        "name": name,
        "topology_applied": apply_topology,
        "event": None if event_stats is None else event_stats.to_dict(),
        "gaussian_count": runner.num_gaussians,
        "recovery_steps": recovery_steps,
        "recovery_snapshot_steps": first_segment,
        "first_recovery_loss": first_loss,
        "snapshot_loss": snapshot_loss,
        "remaining_first_loss": remaining_first_loss,
        "final_loss": final_loss,
        "means_lr_before_recovery": lr_before,
        "means_lr_after_recovery": _current_means_lr(runner),
        "elapsed_seconds": time.perf_counter() - branch_start,
    }
    if device.startswith("cuda"):
        branch_record["peak_cuda_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(torch.device(device))
        )
    write_json(branch_dir / "branch_manifest.json", branch_record)
    del runner
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return branch_record


def run(args: argparse.Namespace) -> None:
    if args.initial_steps < 0 or args.prefix_steps <= 0 or args.recovery_steps <= 0:
        raise ValueError("initial_steps must be non-negative; training steps must be positive")
    if not 0 <= args.recovery_snapshot_steps <= args.recovery_steps:
        raise ValueError("recovery_snapshot_steps must be in [0, recovery_steps]")
    if not np.isfinite(args.budget_fraction) or not 0.0 < args.budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be finite and in (0, 1]")

    track_dir = args.track_dir.resolve()
    config_path = args.config.resolve()
    snapshot_path = (
        args.keyframes_yaml.resolve()
        if args.keyframes_yaml is not None
        else find_latest_keyframe_snapshot(track_dir).resolve()
    )
    config = load_gaussian_config(config_path)
    if args.device is not None:
        config["device"] = args.device
    device = str(config["device"])
    total_update_steps = args.prefix_steps + args.recovery_steps
    _assert_topology_disabled(config, total_update_steps)

    frame_ids, c2w_cv = load_fixed_poses(snapshot_path)
    frame_ids = frame_ids[: INITIAL_KEYFRAMES + 1]
    c2w_cv = c2w_cv[: INITIAL_KEYFRAMES + 1]
    validate_replay_paths(track_dir, frame_ids)
    K = np.loadtxt(track_dir / "cam_K.txt", dtype=np.float32).reshape(3, 3)
    frames = [
        load_frame(track_dir, frame_ids[index], K, c2w_cv[index])
        for index in range(INITIAL_KEYFRAMES + 1)
    ]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _dump_yaml(output_dir / "resolved_config.yml", config)
    normalization, normalization_metadata = compute_initial_normalization(
        frames[:INITIAL_KEYFRAMES], output_dir, config
    )
    manifest = {
        "status": "running",
        "experiment": "shared-checkpoint budget-neutral topology A/B",
        "track_dir": str(track_dir),
        "keyframe_snapshot": str(snapshot_path),
        "frame_ids": frame_ids,
        "device": device,
        "initial_steps": args.initial_steps,
        "prefix_steps": args.prefix_steps,
        "recovery_steps": args.recovery_steps,
        "lr_decay_horizon_steps": total_update_steps,
        "budget_fraction": args.budget_fraction,
        "normalization": normalization_metadata,
        "git": git_metadata(),
    }
    write_json(output_dir / "experiment_manifest.json", manifest)

    if device.startswith("cuda"):
        torch_device = torch.device(device)
        torch.cuda.set_device(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    experiment_start = time.perf_counter()
    runner = GaussianRunner(config, normalization, device=device)
    initial_stats = runner.initialize(
        frames[:INITIAL_KEYFRAMES], train_steps=args.initial_steps
    )
    runner.save_checkpoint(output_dir / "checkpoint_initial.pt")
    append_stats = runner.update([frames[INITIAL_KEYFRAMES]], train_steps=0)
    protected_start = append_stats.gaussians_before
    lr_before_prefix = _current_means_lr(runner)
    prefix_first_loss, prefix_final_loss = runner.train(
        args.prefix_steps,
        lr_decay_horizon_steps=total_update_steps,
    )
    common_checkpoint = output_dir / "checkpoint_common_step200.pt"
    runner.save_checkpoint(common_checkpoint)
    save_render_artifacts(
        runner, len(runner.views) - 1, output_dir, "common_step200_kf006"
    )
    common_record = {
        "initial": initial_stats.to_dict(),
        "append": append_stats.to_dict(),
        "protected_index_range": [protected_start, append_stats.gaussians_after_append],
        "prefix_steps": args.prefix_steps,
        "strategy_step": runner.strategy_step,
        "prefix_first_loss": prefix_first_loss,
        "prefix_final_loss": prefix_final_loss,
        "means_lr_before_prefix": lr_before_prefix,
        "means_lr_after_prefix": _current_means_lr(runner),
        "gaussian_count": runner.num_gaussians,
    }
    write_json(output_dir / "common_manifest.json", common_record)
    del runner
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    branches = [
        _run_branch(
            name="A_append_only",
            common_checkpoint=common_checkpoint,
            output_dir=output_dir,
            device=device,
            protected_start=protected_start,
            budget_fraction=args.budget_fraction,
            lr_decay_horizon_steps=total_update_steps,
            recovery_steps=args.recovery_steps,
            recovery_snapshot_steps=args.recovery_snapshot_steps,
            apply_topology=False,
        ),
        _run_branch(
            name="B_budget_neutral",
            common_checkpoint=common_checkpoint,
            output_dir=output_dir,
            device=device,
            protected_start=protected_start,
            budget_fraction=args.budget_fraction,
            lr_decay_horizon_steps=total_update_steps,
            recovery_steps=args.recovery_steps,
            recovery_snapshot_steps=args.recovery_snapshot_steps,
            apply_topology=True,
        ),
    ]
    manifest["status"] = "complete"
    manifest["common"] = common_record
    manifest["branches"] = branches
    manifest["elapsed_seconds"] = time.perf_counter() - experiment_start
    write_json(output_dir / "experiment_manifest.json", manifest)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        manifest_path = args.output_dir.resolve() / "experiment_manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            manifest["status"] = "failed"
            manifest["failure_type"] = type(exc).__name__
            manifest["failure_message"] = str(exc)
            write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
