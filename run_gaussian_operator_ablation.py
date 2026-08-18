"""Replay one common checkpoint through controlled topology operator branches."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from gaussian_budget_strategy import TopologyMode, apply_budgeted_topology
from gaussian_runner import GaussianRunner
from run_gaussian_incremental import git_metadata, save_render_artifacts, write_json


BRANCHES: tuple[tuple[str, TopologyMode | None], ...] = (
    ("A_append_only", None),
    ("P_prune_only", "prune_only"),
    ("D_duplicate_only", "duplicate_only"),
    ("S_split_only", "split_only"),
    ("M_mixed", "mixed"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one Gaussian checkpoint independently for append-only, prune-only, "
            "duplicate-only, split-only, and mixed topology diagnostics"
        )
    )
    parser.add_argument("--common-checkpoint", type=Path, required=True)
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protected-start", type=int, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--budget-fraction", type=float, default=0.03)
    parser.add_argument("--recovery-steps", type=int, default=500)
    parser.add_argument(
        "--snapshot-steps", type=int, nargs="+", default=[50, 250, 500]
    )
    parser.add_argument("--lr-decay-horizon-steps", type=int, default=701)
    return parser.parse_args()


def _target_depths(runner: GaussianRunner, track_dir: Path) -> list[np.ndarray]:
    if len(runner.views) != 6:
        raise ValueError(
            f"The 5+1 operator ablation requires exactly 6 views, got {len(runner.views)}"
        )
    targets: list[np.ndarray] = []
    for view in runner.views:
        depth_path = track_dir / "depth_filtered" / f"{view.frame_id}.png"
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing target depth: {depth_path}")
        depth_raw = np.asarray(imageio.imread(depth_path))
        if not np.issubdtype(depth_raw.dtype, np.integer):
            raise ValueError(f"Target depth must contain integer millimeters: {depth_path}")
        x0, y0, x1, y1 = view.crop_xyxy
        depth_metric = depth_raw.astype(np.float32) / 1000.0
        cropped = np.ascontiguousarray(depth_metric[y0:y1, x0:x1])
        if cropped.shape != (view.height, view.width):
            raise ValueError(
                f"Depth crop for {view.frame_id} is {cropped.shape}, "
                f"expected {(view.height, view.width)}"
            )
        targets.append(cropped)
    return targets


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


@torch.no_grad()
def _evaluate_views(
    runner: GaussianRunner, target_depths: list[np.ndarray]
) -> dict[str, Any]:
    if len(target_depths) != len(runner.views):
        raise ValueError("Target depth count does not match checkpoint views")
    per_view: list[dict[str, Any]] = []
    for view_index, (view, target_depth) in enumerate(
        zip(runner.views, target_depths, strict=True)
    ):
        rendered, alpha, rendered_depth = runner.render(view_index, include_depth=True)
        if rendered_depth is None:
            raise RuntimeError("Depth rendering unexpectedly returned None")
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
        rendered_depth_coverage = (
            float(np.mean(alpha[valid_depth] > 1e-6)) if np.any(valid_depth) else None
        )
        depth_mae_mm = (
            float(np.mean(np.abs(rendered_depth[valid_depth] - target_depth[valid_depth])) * 1000.0)
            if np.any(valid_depth)
            else None
        )
        per_view.append(
            {
                "view_index": view_index,
                "frame_id": view.frame_id,
                "group": "new" if view_index == len(runner.views) - 1 else "initial",
                "rgb_mae": rgb_mae,
                "rgb_psnr_db": rgb_psnr,
                "alpha_iou_at_0.5": alpha_iou,
                "outside_alpha_mean": outside_alpha,
                "depth_mae_mm": depth_mae_mm,
                "rendered_depth_coverage": rendered_depth_coverage,
                "object_pixels": int(np.count_nonzero(mask)),
                "valid_depth_pixels": int(np.count_nonzero(valid_depth)),
            }
        )

    metric_names = (
        "rgb_mae",
        "rgb_psnr_db",
        "alpha_iou_at_0.5",
        "outside_alpha_mean",
        "depth_mae_mm",
        "rendered_depth_coverage",
    )

    def summarize(items: list[dict[str, Any]]) -> dict[str, float | None]:
        return {name: _mean([item[name] for item in items]) for name in metric_names}

    initial = [item for item in per_view if item["group"] == "initial"]
    newest = [item for item in per_view if item["group"] == "new"]
    return {
        "definitions": {
            "aggregation": "unweighted mean of per-view metrics",
            "rgb_region": "target object mask",
            "alpha_iou_threshold": 0.5,
            "outside_alpha_region": "crop pixels outside target object mask",
            "depth_region": "fixed valid target-depth pixels inside the object mask",
            "rendered_depth_coverage": "fraction of depth-region pixels with rendered alpha > 1e-6",
        },
        "all_views": summarize(per_view),
        "initial_views": summarize(initial),
        "new_view": summarize(newest),
        "per_view": per_view,
    }


def _run_branch(
    *,
    name: str,
    mode: TopologyMode | None,
    common_checkpoint: Path,
    track_dir: Path,
    output_dir: Path,
    device: str,
    protected_start: int,
    budget_fraction: float,
    recovery_steps: int,
    snapshot_steps: list[int],
    lr_decay_horizon_steps: int,
) -> dict[str, Any]:
    branch_dir = output_dir / name
    branch_dir.mkdir()
    runner = GaussianRunner.load_checkpoint(common_checkpoint, device=device)
    recovery_end_step = runner.strategy_step + recovery_steps
    if int(runner.strategy.refine_start_iter) < recovery_end_step:
        raise ValueError(
            "Automatic DefaultStrategy topology must remain disabled through "
            f"recovery step {recovery_end_step}"
        )
    target_depths = _target_depths(runner, track_dir)
    count_before = runner.num_gaussians
    if not 0 <= protected_start <= count_before:
        raise ValueError("protected_start is outside the checkpoint Gaussian range")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    branch_start = time.perf_counter()

    event = None
    if mode is not None:
        event = apply_budgeted_topology(
            runner.splats,
            runner.optimizers,
            runner.strategy_state,
            fraction=budget_fraction,
            protected_start=protected_start,
            mode=mode,
        )
        write_json(branch_dir / "topology_event.json", event.to_dict())
        runner.save_checkpoint(branch_dir / "checkpoint_after_event.pt")

    metrics: dict[str, Any] = {}
    metrics["after_event"] = _evaluate_views(runner, target_depths)
    write_json(branch_dir / "metrics_after_event.json", metrics["after_event"])
    save_render_artifacts(
        runner, len(runner.views) - 1, branch_dir, "after_event_kf006"
    )

    recovery_records: list[dict[str, Any]] = []
    completed_steps = 0
    for snapshot_step in snapshot_steps:
        segment_steps = snapshot_step - completed_steps
        first_loss, final_loss = runner.train(
            segment_steps,
            lr_decay_horizon_steps=lr_decay_horizon_steps,
        )
        completed_steps = snapshot_step
        stage = f"after_{snapshot_step:03d}"
        metrics[stage] = _evaluate_views(runner, target_depths)
        write_json(branch_dir / f"metrics_{stage}.json", metrics[stage])
        save_render_artifacts(
            runner,
            len(runner.views) - 1,
            branch_dir,
            f"{stage}_kf006",
        )
        recovery_records.append(
            {
                "cumulative_steps": snapshot_step,
                "segment_steps": segment_steps,
                "first_loss": first_loss,
                "final_loss": final_loss,
            }
        )

    runner.save_checkpoint(branch_dir / "checkpoint_final.pt")
    runner.export_ply(
        branch_dir / "splats_normalized.ply",
        branch_dir / "splats_metric.ply",
    )
    record = {
        "name": name,
        "mode": mode,
        "diagnostic_only": mode == "prune_only",
        "budget_fraction": budget_fraction,
        "gaussians_before": count_before,
        "gaussians_after_event": (
            count_before if event is None else event.gaussians_after
        ),
        "gaussians_final": runner.num_gaussians,
        "event": None if event is None else event.to_dict(),
        "recovery": recovery_records,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - branch_start,
    }
    if device.startswith("cuda"):
        record["peak_cuda_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(torch.device(device))
        )
    write_json(branch_dir / "branch_manifest.json", record)
    del runner
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return record


def run(args: argparse.Namespace) -> None:
    if not np.isfinite(args.budget_fraction) or not 0.0 < args.budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be finite and in (0, 1]")
    if args.recovery_steps <= 0:
        raise ValueError("recovery_steps must be positive")
    snapshot_steps = sorted(set(args.snapshot_steps))
    if not snapshot_steps or snapshot_steps[-1] != args.recovery_steps:
        raise ValueError("snapshot_steps must end at recovery_steps")
    if snapshot_steps[0] <= 0 or any(step > args.recovery_steps for step in snapshot_steps):
        raise ValueError("snapshot_steps must be within (0, recovery_steps]")
    if args.lr_decay_horizon_steps <= 0:
        raise ValueError("lr_decay_horizon_steps must be positive")

    common_checkpoint = args.common_checkpoint.resolve()
    track_dir = args.track_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not common_checkpoint.is_file():
        raise FileNotFoundError(common_checkpoint)
    if not track_dir.is_dir():
        raise FileNotFoundError(track_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    if args.device.startswith("cuda"):
        torch_device = torch.device(args.device)
        torch.cuda.set_device(torch_device)

    manifest: dict[str, Any] = {
        "status": "running",
        "experiment": "Gaussian topology operator ablation",
        "common_checkpoint": str(common_checkpoint),
        "track_dir": str(track_dir),
        "device": args.device,
        "protected_start": args.protected_start,
        "budget_fraction": args.budget_fraction,
        "recovery_steps": args.recovery_steps,
        "snapshot_steps": snapshot_steps,
        "lr_decay_horizon_steps": args.lr_decay_horizon_steps,
        "branches": [],
        "git": git_metadata(),
    }
    write_json(output_dir / "experiment_manifest.json", manifest)
    experiment_start = time.perf_counter()
    candidate_reference: tuple[list[int], list[int]] | None = None
    try:
        for name, mode in BRANCHES:
            record = _run_branch(
                name=name,
                mode=mode,
                common_checkpoint=common_checkpoint,
                track_dir=track_dir,
                output_dir=output_dir,
                device=args.device,
                protected_start=args.protected_start,
                budget_fraction=args.budget_fraction,
                recovery_steps=args.recovery_steps,
                snapshot_steps=snapshot_steps,
                lr_decay_horizon_steps=args.lr_decay_horizon_steps,
            )
            event = record["event"]
            if event is not None:
                candidates = (event["grow_indices"], event["prune_indices"])
                if candidate_reference is None:
                    candidate_reference = candidates
                elif candidates != candidate_reference:
                    raise RuntimeError(
                        f"{name} did not use the common grow/prune candidate sets"
                    )
            manifest["branches"].append(record)
            write_json(output_dir / "experiment_manifest.json", manifest)
            print(
                json.dumps(
                    {
                        "branch": name,
                        "mode": mode,
                        "gaussians": record["gaussians_final"],
                        "final_all_views": record["metrics"][f"after_{args.recovery_steps:03d}"]["all_views"],
                    }
                ),
                flush=True,
            )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_type"] = type(exc).__name__
        manifest["failure_message"] = str(exc)
        write_json(output_dir / "experiment_manifest.json", manifest)
        raise
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = time.perf_counter() - experiment_start
    write_json(output_dir / "experiment_manifest.json", manifest)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
