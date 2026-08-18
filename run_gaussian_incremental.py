"""Replay saved BundleSDF keyframes through the fixed-pose Gaussian runner.

This entry point is intentionally independent of ``bundlesdf.py``.  It uses a
completed tracking log as immutable input, initializes from the first five
keyframes, and then submits every later keyframe as one reconstruction update.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import torch
from ruamel.yaml import YAML

from gaussian_runner import (
    GSPLAT_VERSION,
    GaussianFrame,
    GaussianRunner,
    SceneNormalization,
    load_gaussian_config,
    validate_c2w_cv,
)


INITIAL_KEYFRAMES = 5
REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-pose incremental gsplat replay of a saved tracking log"
    )
    parser.add_argument(
        "--track-dir",
        type=Path,
        required=True,
        help="Tracking output containing color/depth_filtered/mask and keyframes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for replay artifacts; it must not already exist",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config_gs.yml",
        help="Gaussian YAML configuration",
    )
    parser.add_argument(
        "--keyframes-yaml",
        type=Path,
        help="Fixed-pose snapshot; defaults to the latest tracking snapshot",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, for example cuda:1",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=0,
        help="Stop after this many keyframes; 0 replays every saved keyframe",
    )
    parser.add_argument(
        "--initial-steps",
        type=int,
        help="Override initial optimization steps for a smoke test",
    )
    parser.add_argument(
        "--update-steps",
        type=int,
        help="Override optimization steps per later keyframe",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save an intermediate checkpoint every N updates; 0 disables it",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    yaml = YAML(typ="safe")
    with path.open("r", encoding="utf-8") as yaml_file:
        document = yaml.load(yaml_file)
    if not isinstance(document, Mapping):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return document


def _dump_yaml(path: Path, value: Mapping[str, Any]) -> None:
    yaml = YAML()
    yaml.default_flow_style = False
    with path.open("w", encoding="utf-8") as yaml_file:
        yaml.dump(dict(value), yaml_file)


def find_latest_keyframe_snapshot(track_dir: Path) -> Path:
    candidates = list(track_dir.glob("*/keyframes.yml"))
    if not candidates:
        raise FileNotFoundError(f"No */keyframes.yml found under {track_dir}")
    return max(candidates, key=lambda path: path.parent.name)


def load_fixed_poses(snapshot_path: Path) -> tuple[list[str], np.ndarray]:
    document = _load_yaml(snapshot_path)
    frame_ids: list[str] = []
    poses: list[np.ndarray] = []
    for key, value in document.items():
        if not str(key).startswith("keyframe_"):
            raise ValueError(f"Unexpected key {key!r} in {snapshot_path}")
        if not isinstance(value, Mapping) or "cam_in_ob" not in value:
            raise ValueError(f"Missing cam_in_ob for {key} in {snapshot_path}")
        frame_id = str(key)[len("keyframe_") :]
        pose = np.asarray(value["cam_in_ob"], dtype=np.float32)
        if pose.size != 16:
            raise ValueError(f"cam_in_ob for {frame_id} must have 16 values")
        pose = pose.reshape(4, 4)
        validate_c2w_cv(pose)
        frame_ids.append(frame_id)
        poses.append(pose)
    if len(frame_ids) < INITIAL_KEYFRAMES:
        raise ValueError(
            f"Replay needs at least {INITIAL_KEYFRAMES} keyframes, found {len(frame_ids)}"
        )
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Keyframe snapshot contains duplicate frame IDs")
    return frame_ids, np.stack(poses)


def validate_replay_paths(track_dir: Path, frame_ids: Sequence[str]) -> None:
    required = [track_dir / "cam_K.txt"]
    for frame_id in frame_ids:
        required.extend(
            [
                track_dir / "color" / f"{frame_id}.png",
                track_dir / "depth_filtered" / f"{frame_id}.png",
                track_dir / "mask" / f"{frame_id}.png",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing replay inputs:\n{preview}")


def load_frame(
    track_dir: Path,
    frame_id: str,
    K: np.ndarray,
    c2w_cv: np.ndarray,
) -> GaussianFrame:
    rgb = np.asarray(imageio.imread(track_dir / "color" / f"{frame_id}.png"))
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Invalid RGB image for frame {frame_id}: {rgb.shape}")
    rgb = np.ascontiguousarray(rgb[..., :3])

    depth_raw = np.asarray(
        imageio.imread(track_dir / "depth_filtered" / f"{frame_id}.png")
    )
    if not np.issubdtype(depth_raw.dtype, np.integer):
        raise ValueError(f"Saved depth for frame {frame_id} must be integer millimeters")
    depth = np.ascontiguousarray(depth_raw.astype(np.float32) / 1000.0)

    mask_raw = np.asarray(imageio.imread(track_dir / "mask" / f"{frame_id}.png"))
    if mask_raw.ndim == 3:
        mask_raw = mask_raw.any(axis=-1)
    mask = np.ascontiguousarray(mask_raw > 0)
    return GaussianFrame(frame_id, rgb, depth, mask, K, c2w_cv).validated()


def compute_initial_normalization(
    frames: Sequence[GaussianFrame],
    output_dir: Path,
    config: Mapping[str, Any],
) -> tuple[SceneNormalization, dict[str, Any]]:
    """Reuse BundleSDF scene bounds on only the first five replay frames."""

    from Utils import glcam_in_cvcam
    from tool import compute_scene_bounds

    if len(frames) != INITIAL_KEYFRAMES:
        raise ValueError(f"Expected exactly {INITIAL_KEYFRAMES} initial frames")
    K = frames[0].K
    if any(not np.allclose(frame.K, K) for frame in frames[1:]):
        raise ValueError("BundleSDF scene bounds require one shared K")

    bounds_dir = output_dir / "initial_bounds"
    bounds_dir.mkdir()
    c2w_cv = np.stack([frame.c2w_cv for frame in frames])
    c2w_gl = c2w_cv @ glcam_in_cvcam.astype(np.float32)
    bounds_config = config["scene_bounds"]
    base_scale, translation, real_cloud, _ = compute_scene_bounds(
        None,
        c2w_gl,
        K,
        use_mask=True,
        base_dir=str(bounds_dir),
        rgbs=np.stack([frame.rgb for frame in frames]),
        depths=np.stack([frame.depth for frame in frames]),
        masks=np.stack([frame.mask for frame in frames]),
        eps=float(bounds_config["dbscan_eps"]),
        min_samples=int(bounds_config["dbscan_min_samples"]),
    )
    scale_multiplier = float(bounds_config["online_scale_multiplier"])
    actual_scale = float(base_scale) * scale_multiplier
    normalization = SceneNormalization(actual_scale, np.asarray(translation))
    metadata = {
        "method": "BundleSDF initial-five-frame compute_scene_bounds",
        "base_scale": float(base_scale),
        "online_scale_multiplier": scale_multiplier,
        "actual_scale": actual_scale,
        "translation_metric": normalization.translation.tolist(),
        "bounds_point_count": int(len(real_cloud.points)),
        "dbscan_eps_metric": float(bounds_config["dbscan_eps"]),
        "dbscan_min_samples": int(bounds_config["dbscan_min_samples"]),
    }
    return normalization, metadata


def git_metadata() -> dict[str, Any]:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={REPO_ROOT}", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "head": run_git("rev-parse", "HEAD"),
            "dirty": bool(run_git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, default=_json_default)
        output_file.write("\n")


def save_render_artifacts(
    runner: GaussianRunner,
    view_index: int,
    output_dir: Path,
    label: str,
) -> None:
    render_dir = output_dir / "renders"
    render_dir.mkdir(exist_ok=True)
    rgb, alpha, depth_metric = runner.render(view_index, include_depth=True)
    view = runner.views[view_index]
    target = view.rgb.numpy()
    mask = view.mask.numpy()
    imageio.imwrite(
        render_dir / f"{label}_render.png",
        np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8),
    )
    imageio.imwrite(
        render_dir / f"{label}_target.png",
        np.clip(target * 255.0, 0.0, 255.0).astype(np.uint8),
    )
    imageio.imwrite(
        render_dir / f"{label}_mask.png", (mask.astype(np.uint8) * 255)
    )
    imageio.imwrite(
        render_dir / f"{label}_alpha.png",
        np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8),
    )
    if depth_metric is not None:
        np.save(render_dir / f"{label}_depth_metric.npy", depth_metric)


def run(args: argparse.Namespace) -> None:
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
    if args.initial_steps is not None:
        config["initial_steps"] = args.initial_steps
    if args.update_steps is not None:
        config["update_steps"] = args.update_steps
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")

    frame_ids, c2w_cv = load_fixed_poses(snapshot_path)
    selected_count = len(frame_ids) if args.max_keyframes == 0 else args.max_keyframes
    if selected_count < INITIAL_KEYFRAMES or selected_count > len(frame_ids):
        raise ValueError(
            f"max_keyframes must be 0 or in [{INITIAL_KEYFRAMES}, {len(frame_ids)}]"
        )
    frame_ids = frame_ids[:selected_count]
    c2w_cv = c2w_cv[:selected_count]
    validate_replay_paths(track_dir, frame_ids)
    K = np.loadtxt(track_dir / "cam_K.txt", dtype=np.float32).reshape(3, 3)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _dump_yaml(output_dir / "resolved_config.yml", config)

    initial_frames = [
        load_frame(track_dir, frame_ids[index], K, c2w_cv[index])
        for index in range(INITIAL_KEYFRAMES)
    ]
    normalization, normalization_metadata = compute_initial_normalization(
        initial_frames, output_dir, config
    )
    _dump_yaml(
        output_dir / "normalization_gs.yml",
        {
            "scale": normalization.scale,
            "translation_metric": normalization.translation.tolist(),
            "formula": "x_normalized = scale * (x_metric + translation_metric)",
        },
    )

    manifest = {
        "status": "running",
        "track_dir": str(track_dir),
        "keyframe_snapshot": str(snapshot_path),
        "pose_source": "final saved keyframe snapshot, held fixed during replay",
        "pose_warning": (
            "Saved poses include prior SDF refinement; this validates the GS data path, "
            "not a tracker-only online accuracy result."
        ),
        "pose_convention": "OpenCV camera-to-object, metric",
        "gsplat_viewmat_convention": "OpenCV object-to-camera after normalization",
        "initial_keyframes": INITIAL_KEYFRAMES,
        "selected_keyframes": frame_ids,
        "intrinsics": K.tolist(),
        "normalization": normalization_metadata,
        "gsplat_version": GSPLAT_VERSION,
        "torch_version": torch.__version__,
        "device": str(config["device"]),
        "git": git_metadata(),
    }
    write_json(output_dir / "replay_manifest.json", manifest)

    device = str(config["device"])
    if device.startswith("cuda"):
        torch_device = torch.device(device)
        torch.cuda.set_device(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    runner = GaussianRunner(config, normalization, device=device)
    records: list[dict[str, Any]] = []
    replay_start = time.perf_counter()

    phase_start = time.perf_counter()
    initial_stats = runner.initialize(initial_frames)
    records.append(
        {
            "phase": "initialize",
            **initial_stats.to_dict(),
            "elapsed_seconds": time.perf_counter() - phase_start,
        }
    )
    write_json(output_dir / "updates.json", records)
    print(json.dumps(records[-1]), flush=True)
    runner.save_checkpoint(output_dir / "checkpoint_init.pt")
    save_render_artifacts(
        runner, INITIAL_KEYFRAMES - 1, output_dir, "after_initial_kf005"
    )

    for index in range(INITIAL_KEYFRAMES, selected_count):
        frame = load_frame(track_dir, frame_ids[index], K, c2w_cv[index])
        phase_start = time.perf_counter()
        update_stats = runner.update([frame])
        records.append(
            {
                "phase": "update",
                **update_stats.to_dict(),
                "elapsed_seconds": time.perf_counter() - phase_start,
            }
        )
        write_json(output_dir / "updates.json", records)
        print(json.dumps(records[-1]), flush=True)
        update_number = index - INITIAL_KEYFRAMES + 1
        if (
            args.checkpoint_every > 0
            and update_number % args.checkpoint_every == 0
        ):
            runner.save_checkpoint(
                output_dir / f"checkpoint_after_kf{index + 1:03d}.pt"
            )

    runner.save_checkpoint(output_dir / "checkpoint_final.pt")
    runner.export_ply(
        output_dir / "splats_normalized.ply",
        output_dir / "splats_metric.ply",
    )
    save_render_artifacts(
        runner, selected_count - 1, output_dir, f"final_kf{selected_count:03d}"
    )
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = time.perf_counter() - replay_start
    manifest["final_gaussian_count"] = runner.num_gaussians
    manifest["total_training_steps"] = runner.total_steps
    if device.startswith("cuda"):
        manifest["peak_cuda_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(torch.device(device))
        )
    write_json(output_dir / "replay_manifest.json", manifest)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        manifest_path = args.output_dir.resolve() / "replay_manifest.json"
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
