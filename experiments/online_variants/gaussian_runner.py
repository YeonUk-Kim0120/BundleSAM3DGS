"""Persistent Gaussian Splatting reconstruction primitives.

The runner deliberately keeps the public boundary in BundleTrack's native
coordinate convention: OpenCV camera-to-object poses in metric units.  Scene
normalization and conversion to gsplat world-to-camera matrices happen only
inside this module.

The ``renderer`` config selects between the 3DGS rasterizer (default,
byte-identical to the original behavior) and gsplat's 2DGS surfel rasterizer,
which additionally supports depth supervision plus the 2DGS normal-consistency
and distortion regularizers.

The first integration milestone uses fixed camera poses.  Learned Gaussian
parameters persist across reconstruction updates, while optimizers and the
densification strategy are recreated for every reconstruction update.  This
keeps the representation persistent without depending on private
optimizer-resize behavior in gsplat.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from prior_lifecycle import (
    STATE_CONTRADICTED,
    STATE_UNSEEN,
    STATE_VERIFIED,
    LifecycleFields,
    TransitionThresholds,
    accumulate_support_residuals,
    apply_transitions,
    depth_evidence_masks,
    erode_mask,
    independent_view_mask,
)

# EXPERIMENT COPY (experiments/online_variants, 2026-09-15).  Variants are switched on with the environment variable
# GS_ONLINE_VARIANTS (comma-separated).  With none set this module behaves exactly like the main gaussian_runner.
#   fusion : experiment F "re-observation fusion" — a candidate depth point whose pixel already holds a map splat
#            within GS_FUSION_BAND_M (default 0.005 m) along the ray is NOT appended; instead that splat's centre moves
#            to the count-weighted mean  c <- (n*c + p) / (n + 1),  n <- min(n + 1, GS_FUSION_NMAX).  Prior surfels
#            start with n = GS_FUSION_PRIOR_N (2), appended splats with n = 1.  Only non-CONTRADICTED splats with
#            opacity >= GS_FUSION_MIN_OPACITY (0.05) can absorb observations.
import os as _os
ONLINE_VARIANTS = {v.strip() for v in _os.environ.get("GS_ONLINE_VARIANTS", "").split(",") if v.strip()}
FUSION_BAND_M = float(_os.environ.get("GS_FUSION_BAND_M", "0.005"))
FUSION_NMAX = int(_os.environ.get("GS_FUSION_NMAX", "20"))
FUSION_PRIOR_N = int(_os.environ.get("GS_FUSION_PRIOR_N", "2"))
FUSION_MIN_OPACITY = float(_os.environ.get("GS_FUSION_MIN_OPACITY", "0.05"))
#   GS_MEANS_UPDATE_LR_MULT : experiment A — multiply the means (position) learning rate in keyframe updates only
#            (default 1.0 = unchanged; 3 / 10 tested).  Other parameter groups keep update_lr_scale.
#   GS_APPEND_MASK_ERODE_PX : experiment B — erode the mask by this many pixels before back-projecting candidate points
#            (default 0 = unchanged; the lifecycle judges inside a 2-px-eroded mask).
MEANS_UPDATE_LR_MULT = float(_os.environ.get("GS_MEANS_UPDATE_LR_MULT", "1.0"))
#   GS_MAP_DEPTH_WEIGHT / GS_POSE_DEPTH_WEIGHT : experiment "loss split" — the map parameters receive
#            colour + GS_MAP_DEPTH_WEIGHT * depth (+ regularisers) and the pose deltas receive colour + GS_POSE_DEPTH_WEIGHT * depth,
#            through two backward passes routed with torch.autograd.backward(inputs=...).  Unset = shared loss as before.
MAP_DEPTH_WEIGHT = float(_os.environ["GS_MAP_DEPTH_WEIGHT"]) if _os.environ.get("GS_MAP_DEPTH_WEIGHT") else None
POSE_DEPTH_WEIGHT = float(_os.environ["GS_POSE_DEPTH_WEIGHT"]) if _os.environ.get("GS_POSE_DEPTH_WEIGHT") else None
APPEND_MASK_ERODE_PX = int(_os.environ.get("GS_APPEND_MASK_ERODE_PX", "2"))  # default 2 = main-code default since 2026-09-16
logging.info(f"[online_variants] runner copy {__file__}; variants={sorted(ONLINE_VARIANTS)}")


GSPLAT_VERSION = "1.5.3"
SH_C0 = 0.28209479177387814


DEFAULT_CONFIG: dict[str, Any] = {
    "gsplat_version": GSPLAT_VERSION,
    "device": "cuda:0",
    "seed": 42,
    "voxel_size": 0.01,
    "novelty_distance": 0.01,
    "min_depth": 0.1,
    "max_depth": 2.0,
    "roi_padding": 16,
    "initial_steps": 30_000,
    "update_steps": 500,
    "sh_degree": 3,
    "sh_degree_interval": 1_000,
    "initial_opacity": 0.1,
    "initial_scale_multiplier": 1.0,
    "ssim_weight": 0.2,
    "packed": False,
    "rasterize_mode": "classic",
    "renderer": "3dgs",
    "depth_loss_weight": 0.0,
    "depth_huber_delta_m": 0.03,
    "depth_alpha_threshold": 0.05,
    "normal_consistency_weight": 0.0,
    "normal_consistency_start_step": 7_000,
    "distortion_weight": 0.0,
    "distortion_start_step": 3_000,
    "initial_lr_scale": 1.0,
    "update_lr_scale": 0.1,
    "learning_rates": {
        "means": 1.6e-4,
        "scales": 5.0e-3,
        "quats": 1.0e-3,
        "opacities": 5.0e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    },
    "strategy": {
        "prune_opa": 0.005,
        "grow_grad2d": 0.0002,
        "grow_scale3d": 0.01,
        "prune_scale3d": 0.1,
        "refine_start_iter": 500,
        "refine_stop_iter": 15_000,
        "reset_every": 3_000,
        "refine_every": 100,
        "pause_refine_after_reset": 0,
        "absgrad": False,
        "revised_opacity": False,
    },
    "update_strategy": {
        "refine_start_iter": 100,
        "refine_stop_iter": 500,
    },
    "scene_bounds": {
        "dbscan_eps": 0.06,
        "dbscan_min_samples": 1,
        "online_scale_multiplier": 0.7,
    },
    # Observation-gated prior lifecycle (milestone ③, validated in
    # experiments/ and PRIOR_LIFECYCLE_RESULTS.md).  Default off; when on it
    # requires the 2DGS renderer and disabled densify/opacity-reset (v1).
    "prior_lifecycle": {
        "enabled": False,
        "depth_tolerance_m": 0.01,
        "min_view_angle_deg": 10.0,
        "conflict_min_views": 2,
        "retract_min_views": 3,
        "geometric_alpha_threshold": 0.5,
        "mask_erode_px": 2,
        "grazing_tolerance_cap": 3.0,
        "prior_opacity": 0.9,
        "prior_flat_axis_ratio": 0.1,
    },
    # Milestone ⑤: per-view pose corrections fed back to the tracker.  Port
    # of BundleSDF's PoseArray: tanh-clamped 6-DoF deltas in the normalized
    # object frame, left-multiplied onto each view's c2w, first view fixed,
    # re-created at zero for every training call and baked into the view
    # poses afterwards.  Default off so offline replay and the existing
    # tests are unchanged.
    "pose_feedback": {
        "enabled": False,
        "max_trans_m": 0.02,
        "max_rot_deg": 20.0,
        "lr": 0.01,
        "lr_decay": 0.1,
        "grad_max_norm": 0.1,
        "fix_first_view": True,
        "in_initial": True,
    },
}


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_update(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_gaussian_config(path: str | Path) -> dict[str, Any]:
    """Load a GS YAML file and fill missing values from ``DEFAULT_CONFIG``."""

    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    with Path(path).open("r", encoding="utf-8") as config_file:
        loaded = yaml.load(config_file) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError("Gaussian config must contain a YAML mapping")
    config = _deep_update(DEFAULT_CONFIG, loaded)
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    if config["gsplat_version"] != GSPLAT_VERSION:
        raise ValueError(
            f"This runner is pinned to gsplat {GSPLAT_VERSION}, got "
            f"{config['gsplat_version']}"
        )
    for key in ("voxel_size", "novelty_distance", "min_depth", "max_depth"):
        if not np.isfinite(config[key]) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if float(config["min_depth"]) >= float(config["max_depth"]):
        raise ValueError("min_depth must be smaller than max_depth")
    if not 0.0 <= float(config["ssim_weight"]) <= 1.0:
        raise ValueError("ssim_weight must be in [0, 1]")
    if config["renderer"] not in ("3dgs", "2dgs"):
        raise ValueError("renderer must be '3dgs' or '2dgs'")
    for key in (
        "depth_loss_weight",
        "normal_consistency_weight",
        "distortion_weight",
    ):
        if not np.isfinite(config[key]) or float(config[key]) < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    for key in ("depth_huber_delta_m", "depth_alpha_threshold"):
        if not np.isfinite(config[key]) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("normal_consistency_start_step", "distortion_start_step"):
        if int(config[key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    if config["renderer"] != "2dgs":
        for key in ("normal_consistency_weight", "distortion_weight"):
            if float(config[key]) > 0:
                raise ValueError(f"{key} requires renderer '2dgs'")
    elif bool(config["strategy"]["absgrad"]):
        raise ValueError("strategy.absgrad is not supported with renderer '2dgs'")
    lifecycle = config["prior_lifecycle"]
    if bool(lifecycle["enabled"]):
        if config["renderer"] != "2dgs":
            raise ValueError(
                "prior_lifecycle requires renderer '2dgs' (median front depth)"
            )
        refine_off = 100_000_000
        for block in ("strategy", "update_strategy"):
            if int(config[block].get("refine_start_iter", 0)) < refine_off:
                raise ValueError(
                    "prior_lifecycle v1 requires densification disabled: set "
                    f"{block}.refine_start_iter >= {refine_off}"
                )
        if int(config["strategy"]["reset_every"]) < refine_off:
            raise ValueError(
                "prior_lifecycle v1 requires opacity reset disabled: set "
                f"strategy.reset_every >= {refine_off}"
            )
        if not 0.0 < float(lifecycle["prior_opacity"]) < 1.0:
            raise ValueError("prior_lifecycle.prior_opacity must be in (0, 1)")
        for key in ("depth_tolerance_m", "min_view_angle_deg",
                    "geometric_alpha_threshold", "grazing_tolerance_cap",
                    "prior_flat_axis_ratio"):
            if not np.isfinite(lifecycle[key]) or float(lifecycle[key]) <= 0:
                raise ValueError(f"prior_lifecycle.{key} must be positive")
        for key in ("conflict_min_views", "retract_min_views",
                    "mask_erode_px"):
            if int(lifecycle[key]) < 0:
                raise ValueError(f"prior_lifecycle.{key} must be non-negative")
    for key in ("initial_steps", "update_steps", "roi_padding"):
        if int(config[key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    if int(config["sh_degree"]) < 0:
        raise ValueError("sh_degree must be non-negative")
    if int(config["sh_degree_interval"]) <= 0:
        raise ValueError("sh_degree_interval must be positive")
    if not 0.0 < float(config["initial_opacity"]) < 1.0:
        raise ValueError("initial_opacity must be strictly between 0 and 1")
    for name, learning_rate in config["learning_rates"].items():
        if not np.isfinite(learning_rate) or float(learning_rate) < 0:
            raise ValueError(
                f"learning rate for {name} must be finite and non-negative"
            )
    for key in ("reset_every", "refine_every"):
        if int(config["strategy"][key]) <= 0:
            raise ValueError(f"strategy.{key} must be positive")
    if float(config["scene_bounds"]["dbscan_eps"]) <= 0:
        raise ValueError("scene_bounds.dbscan_eps must be positive")
    if int(config["scene_bounds"]["dbscan_min_samples"]) <= 0:
        raise ValueError("scene_bounds.dbscan_min_samples must be positive")
    if float(config["scene_bounds"]["online_scale_multiplier"]) <= 0:
        raise ValueError("scene_bounds.online_scale_multiplier must be positive")
    feedback = config["pose_feedback"]
    for key in ("max_trans_m", "max_rot_deg", "lr", "grad_max_norm"):
        if not np.isfinite(feedback[key]) or float(feedback[key]) < 0:
            raise ValueError(f"pose_feedback.{key} must be finite and non-negative")
    if not 0.0 < float(feedback["lr_decay"]) <= 1.0:
        raise ValueError("pose_feedback.lr_decay must be in (0, 1]")


def _as_numpy(array: np.ndarray | torch.Tensor, dtype: np.dtype) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=dtype)


def validate_intrinsics(K: np.ndarray | torch.Tensor) -> np.ndarray:
    K_np = _as_numpy(K, np.float64).copy()
    if K_np.shape != (3, 3) or not np.isfinite(K_np).all():
        raise ValueError("K must be a finite 3x3 matrix")
    if K_np[0, 0] <= 0 or K_np[1, 1] <= 0:
        raise ValueError("K must have positive fx and fy")
    if not np.allclose(K_np[2], [0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("K must have bottom row [0, 0, 1]")
    if not np.allclose([K_np[0, 1], K_np[1, 0]], 0.0, atol=1e-7):
        raise ValueError("Non-zero camera skew is not supported")
    return K_np


def validate_c2w_cv(c2w: np.ndarray | torch.Tensor) -> np.ndarray:
    c2w_np = _as_numpy(c2w, np.float64).copy()
    if c2w_np.shape != (4, 4) or not np.isfinite(c2w_np).all():
        raise ValueError("c2w must be a finite 4x4 matrix")
    if not np.allclose(c2w_np[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("c2w must have bottom row [0, 0, 0, 1]")
    rotation = c2w_np[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError("c2w rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError("c2w rotation must have determinant +1")
    return c2w_np


def crop_intrinsics(K: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Adjust intrinsics for a half-open image crop starting at ``(x0, y0)``."""

    cropped = validate_intrinsics(K)
    cropped[0, 2] -= float(x0)
    cropped[1, 2] -= float(y0)
    return cropped


def resize_intrinsics(
    K: np.ndarray,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> np.ndarray:
    """Scale intrinsics using the project's simple pixel-affine convention.

    Sizes are expressed as ``(width, height)``.  This matches the existing
    stride/downscale behavior in ``nerf_runner.py`` and intentionally does not
    add a half-pixel offset.
    """

    old_width, old_height = old_size
    new_width, new_height = new_size
    if min(old_width, old_height, new_width, new_height) <= 0:
        raise ValueError("Image sizes must be positive")
    resized = validate_intrinsics(K)
    resized[0] *= float(new_width) / float(old_width)
    resized[1] *= float(new_height) / float(old_height)
    return resized


@dataclass(frozen=True)
class SceneNormalization:
    """Uniform object-space normalization ``x_norm = s * (x + t)``."""

    scale: float
    translation: np.ndarray

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation, dtype=np.float64).reshape(-1)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("translation must contain three finite values")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "translation", translation.copy())

    def normalize_points(self, points_metric: np.ndarray) -> np.ndarray:
        points = np.asarray(points_metric, dtype=np.float64)
        return (self.scale * (points + self.translation)).astype(np.float32)

    def metric_points(self, points_normalized: np.ndarray) -> np.ndarray:
        points = np.asarray(points_normalized, dtype=np.float64)
        return (points / self.scale - self.translation).astype(np.float32)

    def normalize_c2w(self, c2w_cv_metric: np.ndarray) -> np.ndarray:
        normalized = validate_c2w_cv(c2w_cv_metric)
        normalized[:3, 3] = self.scale * (
            normalized[:3, 3] + self.translation
        )
        return normalized.astype(np.float32)

    def metric_c2w(self, c2w_cv_normalized: np.ndarray) -> np.ndarray:
        metric = validate_c2w_cv(c2w_cv_normalized)
        metric[:3, 3] = metric[:3, 3] / self.scale - self.translation
        return metric.astype(np.float32)


def se3_exp_batch(delta: torch.Tensor) -> torch.Tensor:
    """Batched SE(3)-style exponential of ``[trans(3), rotvec(3)]`` rows.

    Returns ``[N, 4, 4]``.  Rotation is Rodrigues with a clamped angle so the
    gradient stays alive at the zero initialization; translation is applied
    directly (no V-matrix), exact at zero.  Same corrected antisymmetric
    generator as ``experiments/exp_pose_only_probe.py``.
    """

    if delta.ndim != 2 or delta.shape[1] != 6:
        raise ValueError("delta must have shape [N, 6]")
    trans, rotvec = delta[:, :3], delta[:, 3:]
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-12)
    axis = rotvec / theta
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )
    eye = torch.eye(3, dtype=delta.dtype, device=delta.device)[None]
    sin = torch.sin(theta)[..., None]
    cos = torch.cos(theta)[..., None]
    R = eye + sin * K + (1.0 - cos) * (K @ K)
    top = torch.cat((R, trans[:, :, None]), dim=-1)
    bottom = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=delta.dtype, device=delta.device
    ).reshape(1, 1, 4).expand(delta.shape[0], 1, 4)
    return torch.cat((top, bottom), dim=-2)


class PoseDeltas(torch.nn.Module):
    """Per-view pose corrections in the normalized object frame.

    Port of BundleSDF's ``PoseArray``: six raw values per view squashed with
    ``tanh`` so translation stays within ``max_trans_norm`` and rotation
    within ``max_rot_rad``.  ``matrices`` returns the correction ``T_i`` to
    be left-multiplied onto the view's normalized c2w (``c2w' = T_i @ c2w``,
    i.e. a rigid motion of the object frame).  With ``fix_first`` view 0 is
    pinned to the identity (gauge anchor, as in the original).
    """

    def __init__(
        self,
        num_views: int,
        max_trans_norm: float,
        max_rot_rad: float,
        *,
        fix_first: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if int(num_views) <= 0:
            raise ValueError("num_views must be positive")
        self.num_views = int(num_views)
        self.max_trans_norm = float(max_trans_norm)
        self.max_rot_rad = float(max_rot_rad)
        self.fix_first = bool(fix_first)
        self.data = torch.nn.Parameter(
            torch.zeros((self.num_views, 6), dtype=torch.float32, device=device)
        )

    def deltas(self) -> torch.Tensor:
        """Clamped ``[N, 6]`` rows ``[trans(3), rotvec(3)]``."""

        theta = torch.tanh(self.data)
        trans = theta[:, :3] * self.max_trans_norm
        rot = theta[:, 3:] * self.max_rot_rad
        return torch.cat((trans, rot), dim=-1)

    def matrices(self, indices: Any) -> torch.Tensor:
        indices = torch.as_tensor(
            indices, dtype=torch.long, device=self.data.device
        ).reshape(-1)
        T = se3_exp_batch(self.deltas()[indices])
        if self.fix_first:
            keep = (indices != 0).to(T.dtype)[:, None, None]
            eye = torch.eye(4, dtype=T.dtype, device=T.device)[None]
            T = keep * T + (1.0 - keep) * eye
        return T

    @torch.no_grad()
    def magnitudes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(rotation angle [rad], translation norm [normalized]) per view."""

        d = self.deltas()
        rot = torch.linalg.norm(d[:, 3:], dim=-1)
        trans = torch.linalg.norm(d[:, :3], dim=-1)
        if self.fix_first:
            rot[0] = 0.0
            trans[0] = 0.0
        return rot, trans


def c2w_cv_to_viewmat(
    c2w_cv_metric: np.ndarray,
    normalization: SceneNormalization,
) -> np.ndarray:
    """Convert tracker OpenCV c2w directly to gsplat OpenCV w2c."""

    return np.linalg.inv(normalization.normalize_c2w(c2w_cv_metric)).astype(
        np.float32
    )


@dataclass(frozen=True)
class GaussianFrame:
    frame_id: str
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    c2w_cv: np.ndarray

    def validated(self) -> "GaussianFrame":
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth, dtype=np.float32)
        mask = np.asarray(self.mask)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape [H, W, 3]")
        if depth.shape != rgb.shape[:2] or mask.shape[:2] != rgb.shape[:2]:
            raise ValueError("rgb, depth, and mask must share H and W")
        if mask.ndim == 3:
            mask = mask.any(axis=-1)
        elif mask.ndim != 2:
            raise ValueError("mask must have shape [H, W] or [H, W, C]")
        if rgb.dtype == np.uint8:
            rgb_out = rgb.copy()
        else:
            rgb_float = np.asarray(rgb, dtype=np.float32)
            if not np.isfinite(rgb_float).all():
                raise ValueError("rgb must be finite")
            if rgb_float.min() < 0.0 or rgb_float.max() > 1.0:
                raise ValueError("floating-point rgb must be in [0, 1]")
            rgb_out = rgb_float.copy()
        return GaussianFrame(
            frame_id=str(self.frame_id),
            rgb=rgb_out,
            depth=depth.copy(),
            mask=mask.astype(bool, copy=True),
            K=validate_intrinsics(self.K).astype(np.float32),
            c2w_cv=validate_c2w_cv(self.c2w_cv).astype(np.float32),
        )


@dataclass(frozen=True)
class PointCloudBatch:
    points_metric: np.ndarray
    colors: np.ndarray
    raw_point_count: int


@dataclass
class TrainingView:
    frame_id: str
    rgb: torch.Tensor
    mask: torch.Tensor
    K: torch.Tensor
    c2w_normalized: torch.Tensor
    crop_xyxy: tuple[int, int, int, int]
    depth: torch.Tensor | None = None

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


@dataclass(frozen=True)
class GaussianUpdateStats:
    update_index: int
    frame_ids: tuple[str, ...]
    raw_points: int
    candidate_points: int
    novel_points: int
    gaussians_before: int
    gaussians_after_append: int
    gaussians_after_train: int
    train_steps: int
    first_loss: float | None
    final_loss: float | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["frame_ids"] = list(self.frame_ids)
        return result


@dataclass
class RasterResult:
    """Renderer-agnostic rasterization output.

    ``normals``, ``surf_normals``, and ``distort`` are 2DGS-only and stay
    ``None`` for the 3DGS renderer.
    """

    colors: torch.Tensor
    alpha: torch.Tensor
    normals: torch.Tensor | None
    surf_normals: torch.Tensor | None
    distort: torch.Tensor | None
    info: dict[str, Any]


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically average points and colors within a fixed voxel grid."""

    points_np = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors_np = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    if len(points_np) != len(colors_np):
        raise ValueError("points and colors must have the same length")
    if voxel_size <= 0 or not np.isfinite(voxel_size):
        raise ValueError("voxel_size must be finite and positive")
    if len(points_np) == 0:
        return points_np.astype(np.float32), colors_np.astype(np.float32)
    if not np.isfinite(points_np).all() or not np.isfinite(colors_np).all():
        raise ValueError("points and colors must be finite")

    keys = np.floor(points_np / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    point_sums = np.zeros((len(counts), 3), dtype=np.float64)
    color_sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(point_sums, inverse, points_np)
    np.add.at(color_sums, inverse, colors_np)
    return (
        (point_sums / counts[:, None]).astype(np.float32),
        (color_sums / counts[:, None]).astype(np.float32),
    )


def rgbd_to_point_cloud(
    frame: GaussianFrame,
    voxel_size: float,
    min_depth: float,
    max_depth: float,
) -> PointCloudBatch:
    """Back-project masked RGB-D into metric object coordinates."""

    frame = frame.validated()
    valid = (
        frame.mask
        & np.isfinite(frame.depth)
        & (frame.depth >= min_depth)
        & (frame.depth <= max_depth)
    )
    ys, xs = np.nonzero(valid)
    raw_count = len(xs)
    if raw_count == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return PointCloudBatch(empty, empty.copy(), 0)

    z = frame.depth[ys, xs].astype(np.float64)
    fx, fy = float(frame.K[0, 0]), float(frame.K[1, 1])
    cx, cy = float(frame.K[0, 2]), float(frame.K[1, 2])
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    points_camera = np.stack((x, y, z), axis=-1)
    points_object = (
        frame.c2w_cv[:3, :3].astype(np.float64) @ points_camera.T
    ).T + frame.c2w_cv[:3, 3].astype(np.float64)

    colors = frame.rgb[ys, xs].astype(np.float32)
    if frame.rgb.dtype == np.uint8:
        colors /= 255.0
    points_object, colors = voxel_downsample(
        points_object, colors, voxel_size=voxel_size
    )
    return PointCloudBatch(points_object, colors, raw_count)


class GaussianRunner:
    """Fixed-pose persistent Gaussian reconstruction runner."""

    def __init__(
        self,
        config: Mapping[str, Any],
        normalization: SceneNormalization,
        device: str | None = None,
    ) -> None:
        self.config = _deep_update(DEFAULT_CONFIG, config)
        _validate_config(self.config)
        self.normalization = normalization
        self.device = torch.device(device or self.config["device"])
        if self.device.type == "cuda":
            # gsplat 1.5.3's rasterization_2dgs backward hits an illegal
            # memory access whenever the current CUDA device differs from
            # the tensor device (missing device guard in its 2DGS kernels).
            torch.cuda.set_device(self.device)
        self.splats: torch.nn.ParameterDict | None = None
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.strategy: Any = None
        self.strategy_state: dict[str, Any] = {}
        self.strategy_step = 0
        self.views: list[TrainingView] = []
        self.observed_points_metric = np.empty((0, 3), dtype=np.float32)
        self.observed_colors = np.empty((0, 3), dtype=np.float32)
        self.total_steps = 0
        self.update_index = -1
        lifecycle_cfg = self.config["prior_lifecycle"]
        self.lifecycle_enabled = bool(lifecycle_cfg["enabled"])
        self.lifecycle_thresholds = TransitionThresholds(
            depth_tolerance_m=float(lifecycle_cfg["depth_tolerance_m"]),
            min_view_angle_deg=float(lifecycle_cfg["min_view_angle_deg"]),
            conflict_min_views=int(lifecycle_cfg["conflict_min_views"]),
            retract_min_views=int(lifecycle_cfg["retract_min_views"]),
            geometric_alpha_threshold=float(
                lifecycle_cfg["geometric_alpha_threshold"]
            ),
            mask_erode_px=int(lifecycle_cfg["mask_erode_px"]),
            grazing_tolerance_cap=float(lifecycle_cfg["grazing_tolerance_cap"]),
        )
        self.lifecycle_fields: LifecycleFields | None = None
        self.lifecycle_log: list[dict[str, Any]] = []
        self.feedback_log: list[dict[str, Any]] = []
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(self.config["seed"]))
        torch.manual_seed(int(self.config["seed"]))

    @property
    def is_initialized(self) -> bool:
        return self.splats is not None

    @property
    def num_gaussians(self) -> int:
        if self.splats is None:
            return 0
        return int(self.splats["means"].shape[0])

    def _require_gsplat(self) -> tuple[Any, Any, Any, Any]:
        try:
            import gsplat
            from gsplat.rendering import rasterization, rasterization_2dgs
            from gsplat.strategy import DefaultStrategy
        except ImportError as exc:
            raise RuntimeError(
                "gsplat is required for Gaussian training; install pinned version "
                f"{GSPLAT_VERSION}"
            ) from exc
        if gsplat.__version__ != GSPLAT_VERSION:
            raise RuntimeError(
                f"Expected gsplat {GSPLAT_VERSION}, found {gsplat.__version__}"
            )
        return gsplat, rasterization, rasterization_2dgs, DefaultStrategy

    def _rasterize(
        self,
        K: torch.Tensor,
        c2w: torch.Tensor,
        width: int,
        height: int,
        sh_degree: int,
        render_mode: str,
        absgrad: bool,
    ) -> RasterResult:
        if self.splats is None:
            raise RuntimeError("Runner requires splats to rasterize")
        _, rasterization, rasterization_2dgs, _ = self._require_gsplat()
        common = dict(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=torch.cat((self.splats["sh0"], self.splats["shN"]), dim=1),
            viewmats=torch.linalg.inv(c2w),
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=bool(self.config["packed"]),
            sparse_grad=False,
            render_mode=render_mode,
        )
        if self.config["renderer"] == "2dgs":
            colors, alpha, normals, surf_normals, distort, _, info = (
                rasterization_2dgs(
                    **common,
                    absgrad=absgrad,
                    distloss=float(self.config["distortion_weight"]) > 0,
                    depth_mode="expected",
                )
            )
            return RasterResult(colors, alpha, normals, surf_normals, distort, info)
        colors, alpha, info = rasterization(
            **common,
            absgrad=absgrad,
            rasterize_mode=str(self.config["rasterize_mode"]),
            camera_model="pinhole",
        )
        return RasterResult(colors, alpha, None, None, None, info)

    def _frames_to_cloud(
        self, frames: Sequence[GaussianFrame]
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        point_batches: list[np.ndarray] = []
        color_batches: list[np.ndarray] = []
        raw_count = 0
        for frame in frames:
            if APPEND_MASK_ERODE_PX > 0:
                import dataclasses
                eroded = erode_mask(torch.from_numpy(np.asarray(frame.mask).astype(bool)), APPEND_MASK_ERODE_PX)
                frame = dataclasses.replace(frame, mask=eroded.numpy())
            batch = rgbd_to_point_cloud(
                frame,
                voxel_size=float(self.config["voxel_size"]),
                min_depth=float(self.config["min_depth"]),
                max_depth=float(self.config["max_depth"]),
            )
            raw_count += batch.raw_point_count
            if len(batch.points_metric):
                point_batches.append(batch.points_metric)
                color_batches.append(batch.colors)
        if not point_batches:
            empty = np.empty((0, 3), dtype=np.float32)
            return empty, empty.copy(), raw_count, 0
        points, colors = voxel_downsample(
            np.concatenate(point_batches, axis=0),
            np.concatenate(color_batches, axis=0),
            voxel_size=float(self.config["voxel_size"]),
        )
        return points, colors, raw_count, len(points)

    def _select_novel(
        self, points: np.ndarray, colors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(points) == 0:
            return points, colors, np.empty((0,), dtype=np.float32)
        if len(self.observed_points_metric) == 0:
            distances = np.full(len(points), np.inf, dtype=np.float32)
            return points.copy(), colors.copy(), distances
        tree = cKDTree(self.observed_points_metric)
        distances, _ = tree.query(points, k=1, workers=-1)
        distances = np.asarray(distances, dtype=np.float32)
        novel = distances > float(self.config["novelty_distance"])
        return points[novel], colors[novel], distances

    @torch.no_grad()
    def _fuse_reobservations(
        self, points: np.ndarray, colors: np.ndarray, frames: Sequence[GaussianFrame]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Experiment F: fuse candidate observations into the map splat projected to the same pixel (within the band)."""

        stats: dict[str, Any] = {"candidates": int(len(points)), "fused": 0, "splats_moved": 0, "mean_move_mm": 0.0}
        if len(points) == 0 or self.splats is None or self.lifecycle_fields is None:
            return points, colors, stats
        device = self.device
        fields = self.lifecycle_fields
        means_metric = torch.from_numpy(
            self.normalization.metric_points(
                self.splats["means"].detach().cpu().numpy()
            ).astype(np.float32)
        ).to(device)
        opac = torch.sigmoid(self.splats["opacities"].detach()).to(device)
        eligible = (fields.state.to(device) != STATE_CONTRADICTED) & (opac >= FUSION_MIN_OPACITY)
        pts = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(device)
        fused_any = torch.zeros(len(pts), dtype=torch.bool, device=device)
        acc_sum = torch.zeros_like(means_metric)
        acc_n = torch.zeros(len(means_metric), dtype=torch.int32, device=device)
        band = float(FUSION_BAND_M)
        for frame in frames:
            frame = frame.validated()
            c2w = torch.from_numpy(frame.c2w_cv.astype(np.float32)).to(device)
            w2c = torch.linalg.inv(c2w)
            K = frame.K
            height, width = frame.mask.shape

            def project(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                pc = x @ w2c[:3, :3].T + w2c[:3, 3][None, :]
                z = pc[:, 2]
                safe = z.clamp_min(1e-6)
                u = (pc[:, 0] / safe * float(K[0, 0]) + float(K[0, 2])).round().long()
                v = (pc[:, 1] / safe * float(K[1, 1]) + float(K[1, 2])).round().long()
                ok = (z > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
                key = v.clamp(0, height - 1) * width + u.clamp(0, width - 1)
                return z, key, ok

            zg, keyg, okg = project(means_metric)
            okg &= eligible
            zp, keyp, okp = project(pts)
            observed = torch.from_numpy(frame.depth.astype(np.float32)).to(device).reshape(-1)
            # per pixel: the eligible splat whose ray depth is closest to the observed depth of that pixel
            dg = (zg - observed[keyg]).abs()
            diff = torch.full((height * width,), float("inf"), device=device)
            diff.scatter_reduce_(0, keyg[okg], dg[okg], reduce="amin", include_self=True)
            best = torch.full((height * width,), -1, dtype=torch.long, device=device)
            idx = torch.nonzero(okg & (dg <= diff[keyg]))[:, 0]
            best[keyg[idx]] = idx
            # 3x3 pixel neighbourhood (1 mm splats fall on neighbouring pixels): take the neighbour splat closest in
            # ray depth to the candidate; strict same-pixel matching missed 34-41 % of re-observations (2026-09-15).
            kp_v = keyp // width
            kp_u = keyp % width
            best_d = torch.full((len(pts),), float("inf"), device=device)
            best_g = torch.full((len(pts),), -1, dtype=torch.long, device=device)
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    vv = (kp_v + dv).clamp(0, height - 1)
                    uu = (kp_u + du).clamp(0, width - 1)
                    gg = best[vv * width + uu]
                    has = gg >= 0
                    d = torch.full((len(pts),), float("inf"), device=device)
                    d[has] = (zp[has] - zg[gg[has]]).abs()
                    better = d < best_d
                    best_d = torch.where(better, d, best_d)
                    best_g = torch.where(better, gg, best_g)
            hit = okp & (best_g >= 0) & ~fused_any & (best_d <= band)
            hit_idx = torch.nonzero(hit)[:, 0]
            if len(hit_idx) == 0:
                continue
            gi = best_g[hit_idx]
            acc_sum.index_add_(0, gi, pts[hit_idx])
            acc_n.index_add_(0, gi, torch.ones(len(hit_idx), dtype=torch.int32, device=device))
            fused_any[hit_idx] = True
        moved = acc_n > 0
        if bool(moved.any()):
            n = fields.obs_count.to(device)[moved].to(torch.float32)
            target = acc_sum[moved] / acc_n[moved].to(torch.float32)[:, None]
            new_metric = (n[:, None] * means_metric[moved] + target) / (n[:, None] + 1.0)
            new_norm = torch.from_numpy(
                self.normalization.normalize_points(new_metric.cpu().numpy()).astype(np.float32)
            ).to(self.splats["means"].device)
            self.splats["means"].data[moved.to(self.splats["means"].device)] = new_norm
            fields.obs_count[moved.to(fields.obs_count.device)] = torch.clamp(
                fields.obs_count[moved.to(fields.obs_count.device)] + 1, max=FUSION_NMAX
            )
            stats["mean_move_mm"] = float(((new_metric - means_metric[moved]).norm(dim=1).mean() * 1000.0).item())
        keep = (~fused_any).cpu().numpy()
        stats["fused"] = int(fused_any.sum())
        stats["splats_moved"] = int(moved.sum())
        logging.info("[fusion] " + json.dumps(stats))
        return np.asarray(points)[keep], np.asarray(colors)[keep], stats

    def _prepare_view(self, frame: GaussianFrame) -> TrainingView:
        frame = frame.validated()
        ys, xs = np.nonzero(frame.mask)
        if len(xs) == 0:
            raise ValueError(f"Frame {frame.frame_id} has an empty mask")
        padding = int(self.config["roi_padding"])
        height, width = frame.mask.shape
        x0 = max(0, int(xs.min()) - padding)
        x1 = min(width, int(xs.max()) + 1 + padding)
        y0 = max(0, int(ys.min()) - padding)
        y1 = min(height, int(ys.max()) + 1 + padding)
        rgb = frame.rgb[y0:y1, x0:x1]
        if rgb.dtype == np.uint8:
            rgb_tensor = torch.from_numpy(rgb.copy()).float() / 255.0
        else:
            rgb_tensor = torch.from_numpy(rgb.astype(np.float32, copy=True))
        mask_tensor = torch.from_numpy(frame.mask[y0:y1, x0:x1].copy()).bool()
        depth_tensor = torch.from_numpy(
            frame.depth[y0:y1, x0:x1].astype(np.float32, copy=True)
        )
        K_crop = crop_intrinsics(frame.K, x0=x0, y0=y0).astype(np.float32)
        c2w_normalized = self.normalization.normalize_c2w(frame.c2w_cv)
        return TrainingView(
            frame_id=frame.frame_id,
            rgb=rgb_tensor.contiguous(),
            mask=mask_tensor.contiguous(),
            K=torch.from_numpy(K_crop).contiguous(),
            c2w_normalized=torch.from_numpy(c2w_normalized).contiguous(),
            crop_xyxy=(x0, y0, x1, y1),
            depth=depth_tensor.contiguous(),
        )

    def _initial_log_scales(
        self, points_normalized: np.ndarray, reference_normalized: np.ndarray
    ) -> np.ndarray:
        if len(points_normalized) == 0:
            return np.empty((0, 3), dtype=np.float32)
        reference = np.asarray(reference_normalized, dtype=np.float64).reshape(-1, 3)
        points = np.asarray(points_normalized, dtype=np.float64).reshape(-1, 3)
        if len(reference) < 2:
            distance = np.full(
                len(points),
                float(self.config["voxel_size"]) * self.normalization.scale,
            )
        else:
            tree = cKDTree(reference)
            k = min(4, len(reference))
            distances, _ = tree.query(points, k=k, workers=-1)
            distances = np.asarray(distances, dtype=np.float64)
            if distances.ndim == 1:
                distances = distances[:, None]
            positive = np.where(distances > 1e-12, distances, np.nan)
            distance = np.sqrt(np.nanmean(positive**2, axis=1))
            fallback = float(self.config["voxel_size"]) * self.normalization.scale
            distance = np.where(np.isfinite(distance), distance, fallback)
        minimum = 0.1 * float(self.config["voxel_size"]) * self.normalization.scale
        distance = np.maximum(distance, minimum)
        distance *= float(self.config["initial_scale_multiplier"])
        return np.log(distance)[:, None].repeat(3, axis=1).astype(np.float32)

    def _new_splat_values(
        self,
        points_metric: np.ndarray,
        colors: np.ndarray,
        reference_metric: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        points_normalized = self.normalization.normalize_points(points_metric)
        reference_normalized = self.normalization.normalize_points(reference_metric)
        log_scales = self._initial_log_scales(
            points_normalized, reference_normalized
        )
        count = len(points_normalized)
        sh0 = (np.asarray(colors, dtype=np.float32) - 0.5) / SH_C0
        sh_count = (int(self.config["sh_degree"]) + 1) ** 2
        opacity = float(self.config["initial_opacity"])
        opacity_logit = float(torch.logit(torch.tensor(opacity)).item())
        return {
            "means": torch.from_numpy(points_normalized),
            "scales": torch.from_numpy(log_scales),
            "quats": torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(count, 1),
            "opacities": torch.full((count,), opacity_logit),
            "sh0": torch.from_numpy(sh0[:, None, :]),
            "shN": torch.zeros((count, sh_count - 1, 3), dtype=torch.float32),
        }

    def _set_splats(self, values: Mapping[str, torch.Tensor]) -> None:
        expected = {"means", "scales", "quats", "opacities", "sh0", "shN"}
        if set(values) != expected:
            raise ValueError(f"Splat keys must be {sorted(expected)}")
        lengths = {int(value.shape[0]) for value in values.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("All splat tensors must have the same non-zero length")
        for name, value in values.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite values in {name}")
        self.splats = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(value.detach().to(self.device).contiguous())
                for name, value in values.items()
            }
        )

    def _reset_optimization_state(
        self, lr_scale: float, *, initial: bool
    ) -> None:
        if self.splats is None:
            raise RuntimeError("Cannot create optimizers before splats")
        _, _, _, DefaultStrategy = self._require_gsplat()
        base_lrs = self.config["learning_rates"]
        self.optimizers = {
            name: torch.optim.Adam(
                [
                    {
                        "params": [self.splats[name]],
                        "lr": float(base_lrs[name]) * lr_scale
                        * (MEANS_UPDATE_LR_MULT if (name == "means" and not initial) else 1.0),
                        "name": name,
                    }
                ],
                eps=1e-15,
            )
            for name in self.splats.keys()
        }
        strategy_config = dict(self.config["strategy"])
        if not initial:
            strategy_config.update(dict(self.config["update_strategy"]))
        if self.config["renderer"] == "2dgs":
            strategy_config["key_for_gradient"] = "gradient_2dgs"
        self.strategy = DefaultStrategy(**strategy_config)
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        self.strategy_step = 0

    def _append_splats(self, points_metric: np.ndarray, colors: np.ndarray) -> None:
        if len(points_metric) == 0:
            return
        if self.splats is None:
            raise RuntimeError("Runner must be initialized before append")
        reference = np.concatenate(
            (self.observed_points_metric, points_metric), axis=0
        )
        new_values = self._new_splat_values(points_metric, colors, reference)
        combined: dict[str, torch.Tensor] = {}
        for name in self.splats.keys():
            combined[name] = torch.cat(
                (
                    self.splats[name].detach().cpu(),
                    new_values[name].detach().cpu(),
                ),
                dim=0,
            )
        self._set_splats(combined)

    def initialize(
        self,
        frames: Sequence[GaussianFrame],
        train_steps: int | None = None,
    ) -> GaussianUpdateStats:
        if self.is_initialized:
            raise RuntimeError("GaussianRunner is already initialized")
        if not frames:
            raise ValueError("At least one frame is required")
        validated = [frame.validated() for frame in frames]
        frame_ids = [frame.frame_id for frame in validated]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("Initial frame IDs must be unique")
        points, colors, raw_count, candidate_count = self._frames_to_cloud(validated)
        if len(points) == 0:
            raise ValueError("Initial frames contain no valid masked depth points")
        generator_state = self._generator.get_state().clone()
        torch_rng_state = torch.get_rng_state().clone()
        cuda_rng_state = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        try:
            self.observed_points_metric = points.copy()
            self.observed_colors = colors.copy()
            values = self._new_splat_values(points, colors, points)
            self._set_splats(values)
            if self.lifecycle_enabled:
                # RGB-D-seeded splats are direct observations.
                self.lifecycle_fields = LifecycleFields.create(
                    self.num_gaussians, lineage_prior=False, device=self.device
                )
            self.views = [self._prepare_view(frame) for frame in validated]
            self.update_index = 0
            self._reset_optimization_state(
                float(self.config["initial_lr_scale"]), initial=True
            )
            steps = int(
                self.config["initial_steps"] if train_steps is None else train_steps
            )
            first_loss, final_loss = self.train(steps)
        except Exception:
            self.splats = None
            self.optimizers = {}
            self.strategy = None
            self.strategy_state = {}
            self.strategy_step = 0
            self.lifecycle_fields = None
            self.views = []
            self.observed_points_metric = np.empty((0, 3), dtype=np.float32)
            self.observed_colors = np.empty((0, 3), dtype=np.float32)
            self.total_steps = 0
            self.update_index = -1
            self._generator.set_state(generator_state)
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            raise
        return GaussianUpdateStats(
            update_index=self.update_index,
            frame_ids=tuple(frame.frame_id for frame in validated),
            raw_points=raw_count,
            candidate_points=candidate_count,
            novel_points=len(points),
            gaussians_before=0,
            gaussians_after_append=len(points),
            gaussians_after_train=self.num_gaussians,
            train_steps=steps,
            first_loss=first_loss,
            final_loss=final_loss,
        )

    def update(
        self,
        frames: Sequence[GaussianFrame],
        train_steps: int | None = None,
    ) -> GaussianUpdateStats:
        if not self.is_initialized or self.splats is None:
            raise RuntimeError("GaussianRunner must be initialized before update")
        if not frames:
            raise ValueError("At least one new frame is required")
        validated = [frame.validated() for frame in frames]
        frame_ids = [frame.frame_id for frame in validated]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("Update frame IDs must be unique")
        processed_ids = {view.frame_id for view in self.views}
        duplicate_ids = processed_ids.intersection(frame_ids)
        if duplicate_ids:
            raise ValueError(
                f"Frames were already processed: {sorted(duplicate_ids)}"
            )
        points, colors, raw_count, candidate_count = self._frames_to_cloud(validated)
        fusion_stats = None
        if "fusion" in ONLINE_VARIANTS and self.lifecycle_fields is not None:
            points, colors, fusion_stats = self._fuse_reobservations(points, colors, validated)
        novel_points, novel_colors, _ = self._select_novel(points, colors)
        before = self.num_gaussians

        snapshot = {
            "splats": {
                name: value.detach().cpu().clone()
                for name, value in self.splats.items()
            },
            "view_count": len(self.views),
            "observed_points": self.observed_points_metric.copy(),
            "observed_colors": self.observed_colors.copy(),
            "total_steps": self.total_steps,
            "update_index": self.update_index,
            "generator_state": self._generator.get_state().clone(),
            "torch_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_state": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if self.device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
            "lifecycle_fields": (
                self.lifecycle_fields.keep(
                    torch.ones(len(self.lifecycle_fields), dtype=torch.bool,
                               device=self.lifecycle_fields.state.device)
                )
                if self.lifecycle_fields is not None else None
            ),
        }

        try:
            self._append_splats(novel_points, novel_colors)
            after_append = self.num_gaussians
            if self.lifecycle_fields is not None and len(novel_points):
                self.lifecycle_fields = self.lifecycle_fields.concat(
                    LifecycleFields.create(
                        len(novel_points), lineage_prior=False,
                        device=self.device,
                    )
                )
            self.views.extend(self._prepare_view(frame) for frame in validated)
            self.update_index += 1
            self.classify_lifecycle(
                validated, event=f"update_{self.update_index:03d}"
            )
            if fusion_stats is not None and self.lifecycle_log:
                self.lifecycle_log[-1]["fusion"] = fusion_stats
            self._remove_contradicted()
            self._reset_optimization_state(
                float(self.config["update_lr_scale"]), initial=False
            )
            steps = int(
                self.config["update_steps"] if train_steps is None else train_steps
            )
            first_loss, final_loss = self.train(
                steps, optimize_poses=bool(self.config["pose_feedback"]["enabled"])
            )
            if len(novel_points):
                self.observed_points_metric, self.observed_colors = voxel_downsample(
                    np.concatenate(
                        (self.observed_points_metric, novel_points), axis=0
                    ),
                    np.concatenate((self.observed_colors, novel_colors), axis=0),
                    voxel_size=float(self.config["voxel_size"]),
                )
        except Exception:
            self._set_splats(snapshot["splats"])
            self.lifecycle_fields = snapshot["lifecycle_fields"]
            self.views = self.views[: snapshot["view_count"]]
            self.observed_points_metric = snapshot["observed_points"]
            self.observed_colors = snapshot["observed_colors"]
            self.total_steps = snapshot["total_steps"]
            self.update_index = snapshot["update_index"]
            self._generator.set_state(snapshot["generator_state"])
            is_initial_state = self.update_index == 0
            lr_scale = float(
                self.config[
                    "initial_lr_scale" if is_initial_state else "update_lr_scale"
                ]
            )
            self._reset_optimization_state(
                lr_scale, initial=is_initial_state
            )
            torch.set_rng_state(snapshot["torch_rng_state"])
            if snapshot["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(snapshot["cuda_rng_state"])
            raise

        return GaussianUpdateStats(
            update_index=self.update_index,
            frame_ids=tuple(frame.frame_id for frame in validated),
            raw_points=raw_count,
            candidate_points=candidate_count,
            novel_points=len(novel_points),
            gaussians_before=before,
            gaussians_after_append=after_append,
            gaussians_after_train=self.num_gaussians,
            train_steps=steps,
            first_loss=first_loss,
            final_loss=final_loss,
        )

    def _masked_dssim(
        self, rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        try:
            from kornia.losses import ssim_loss
        except ImportError as exc:
            raise RuntimeError("kornia is required for the DSSIM loss") from exc
        mask_channels = mask[..., None].to(rendered.dtype)
        rendered_chw = (rendered * mask_channels).permute(0, 3, 1, 2)
        target_chw = (target * mask_channels).permute(0, 3, 1, 2)
        dssim_map = ssim_loss(
            rendered_chw,
            target_chw,
            window_size=11,
            reduction="none",
        )
        expanded_mask = mask_channels.permute(0, 3, 1, 2).expand_as(dssim_map)
        denominator = expanded_mask.sum().clamp_min(1.0)
        # Kornia returns (1 - SSIM) / 2.  Standard 3DGS uses (1 - SSIM).
        return 2.0 * (dssim_map * expanded_mask).sum() / denominator

    def _strategy_post_backward(
        self, strategy_step: int, info: Mapping[str, Any]
    ) -> None:
        """Apply DefaultStrategy and correct v1.5.3's opacity-reset condition."""

        import gsplat
        from gsplat.strategy.ops import reset_opa

        if gsplat.__version__ != GSPLAT_VERSION:
            raise RuntimeError("Opacity-reset compatibility is only for gsplat 1.5.3")
        should_reset = (
            self.strategy.reset_every > 0
            and 0 < strategy_step < self.strategy.refine_stop_iter
            and strategy_step % self.strategy.reset_every == 0
        )
        self.strategy.step_post_backward(
            self.splats,
            self.optimizers,
            self.strategy_state,
            strategy_step,
            info,
            packed=bool(self.config["packed"]),
        )
        if should_reset:
            reset_opa(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                value=self.strategy.prune_opa * 2.0,
            )

    # ----- observation-gated prior lifecycle (milestone ③) -----------------

    def initialize_from_prior(
        self,
        surfels_cv: Any,
        first_c2w_cv: np.ndarray,
        frames: Sequence[GaussianFrame],
        train_steps: int | None = None,
    ) -> GaussianUpdateStats:
        """Initialize the map from aligned first-camera-frame surfels.

        ``surfels_cv`` is a ``sam3d_prior.SurfelSet`` in the metric OpenCV
        frame of the first camera (the ② alignment output); ``first_c2w_cv``
        transforms it into the object frame. RGB-D novelty appends then only
        add regions the prior does not cover.
        """

        import torch.nn.functional as F

        from sam3d_prior import quats_from_normals

        if self.is_initialized:
            raise RuntimeError("GaussianRunner is already initialized")
        if not self.lifecycle_enabled:
            raise RuntimeError(
                "initialize_from_prior requires prior_lifecycle.enabled"
            )
        if not frames:
            raise ValueError("At least one frame is required")
        validated = [frame.validated() for frame in frames]
        first_c2w = validate_c2w_cv(first_c2w_cv).astype(np.float32)
        lifecycle_cfg = self.config["prior_lifecycle"]

        R0 = torch.from_numpy(first_c2w[:3, :3])
        t0 = torch.from_numpy(first_c2w[:3, 3])
        means_metric = surfels_cv.means @ R0.T + t0[None, :]
        normals_obj = F.normalize(surfels_cv.normals @ R0.T, dim=-1)
        means_norm = torch.from_numpy(
            self.normalization.normalize_points(means_metric.numpy())
        )
        radii_norm = surfels_cv.radii * float(self.normalization.scale)
        flat = float(lifecycle_cfg["prior_flat_axis_ratio"])
        log_scales = torch.log(torch.stack(
            (radii_norm, radii_norm, radii_norm * flat), dim=-1
        ).clamp_min(1e-9))
        opacity_logit = float(torch.logit(
            torch.tensor(float(lifecycle_cfg["prior_opacity"]))
        ))
        sh_count = (int(self.config["sh_degree"]) + 1) ** 2
        count = len(means_norm)
        self._set_splats({
            "means": means_norm.float(),
            "scales": log_scales.float(),
            "quats": quats_from_normals(normals_obj).float(),
            "opacities": torch.full((count,), opacity_logit),
            "sh0": ((surfels_cv.colors - 0.5) / SH_C0)[:, None, :].float(),
            "shN": torch.zeros((count, sh_count - 1, 3), dtype=torch.float32),
        })
        self.lifecycle_fields = LifecycleFields.create(
            count, lineage_prior=True, device=self.device,
            obs_count=FUSION_PRIOR_N if "fusion" in ONLINE_VARIANTS else 1,
        )
        self.observed_points_metric = means_metric.numpy().astype(np.float32)
        self.observed_colors = surfels_cv.colors.numpy().astype(np.float32)
        self.views = [self._prepare_view(frame) for frame in validated]
        self.update_index = 0
        # Classify and compact BEFORE creating optimizers so Adam rows align.
        self.classify_lifecycle(validated, event="initialize")
        removed = self._remove_contradicted()
        self._reset_optimization_state(
            float(self.config["initial_lr_scale"]), initial=True
        )
        steps = int(
            self.config["initial_steps"] if train_steps is None else train_steps
        )
        feedback_cfg = self.config["pose_feedback"]
        first_loss, final_loss = self.train(
            steps,
            optimize_poses=bool(feedback_cfg["enabled"])
            and bool(feedback_cfg["in_initial"]),
        )
        return GaussianUpdateStats(
            update_index=self.update_index,
            frame_ids=tuple(frame.frame_id for frame in validated),
            raw_points=count,
            candidate_points=count,
            novel_points=count - removed,
            gaussians_before=0,
            gaussians_after_append=count,
            gaussians_after_train=self.num_gaussians,
            train_steps=steps,
            first_loss=first_loss,
            final_loss=final_loss,
        )

    def _splat_normals(self) -> torch.Tensor:
        from sam3d_prior import quat_wxyz_to_matrix

        return quat_wxyz_to_matrix(self.splats["quats"].detach())[:, :, 2]

    @torch.no_grad()
    def _geometric_front_depth(
        self, c2w_norm: torch.Tensor, K: torch.Tensor, width: int, height: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Opacity-independent front depth: 2DGS median depth at opacity 1."""

        _, _, rasterization_2dgs, _ = self._require_gsplat()
        renders, alphas, *_ = rasterization_2dgs(
            means=self.splats["means"].detach(),
            quats=self.splats["quats"].detach(),
            scales=torch.exp(self.splats["scales"].detach()),
            opacities=torch.ones(self.num_gaussians, device=self.device),
            colors=torch.zeros(self.num_gaussians, 1, 3, device=self.device),
            sh_degree=0,
            viewmats=torch.linalg.inv(c2w_norm)[None],
            Ks=K[None],
            width=width,
            height=height,
            render_mode="RGB+ED",
            depth_mode="median",
        )
        depth_metric = renders[0, ..., 3] / float(self.normalization.scale)
        return depth_metric, alphas[0, ..., 0]

    @torch.no_grad()
    def classify_lifecycle(
        self,
        frames: Sequence[GaussianFrame],
        event: str,
        occ_masks: Sequence[np.ndarray] | None = None,
    ) -> dict[str, Any] | None:
        """Update lifecycle states from a batch of keyframes (metric space)."""

        import torch.nn.functional as F

        if not self.lifecycle_enabled or self.lifecycle_fields is None:
            return None
        if occ_masks is not None and len(occ_masks) != len(frames):
            raise ValueError("occ_masks must align with frames")
        fields = self.lifecycle_fields.validated()
        thresholds = self.lifecycle_thresholds
        device = self.device
        supported = torch.zeros(len(fields), dtype=torch.bool, device=device)
        means_metric = torch.from_numpy(
            self.normalization.metric_points(
                self.splats["means"].detach().cpu().numpy()
            )
        ).to(device)
        normals = self._splat_normals()
        active = fields.state != STATE_CONTRADICTED
        totals = {"projected_valid": 0, "support": 0, "free_space": 0,
                  "behind_occluded": 0, "behind_miss": 0,
                  "accepted_conflict": 0}

        for index, frame in enumerate(frames):
            frame = frame.validated()
            c2w = torch.from_numpy(frame.c2w_cv.astype(np.float32)).to(device)
            w2c = torch.linalg.inv(c2w)
            points_cam = means_metric @ w2c[:3, :3].T + w2c[:3, 3][None, :]
            depth_cam = points_cam[:, 2]
            safe = depth_cam.clamp_min(1e-6)
            K = frame.K
            height, width = frame.mask.shape
            u = points_cam[:, 0] / safe * float(K[0, 0]) + float(K[0, 2])
            v = points_cam[:, 1] / safe * float(K[1, 1]) + float(K[1, 2])
            in_image = ((depth_cam > 0.01) & (u >= 0) & (u < width)
                        & (v >= 0) & (v < height))
            ui = u.round().clamp(0, width - 1).long()
            vi = v.round().clamp(0, height - 1).long()

            observed = torch.from_numpy(frame.depth).to(device)
            mask = erode_mask(
                torch.from_numpy(frame.mask).to(device),
                thresholds.mask_erode_px,
            )
            valid_pixels = (
                mask & torch.isfinite(observed)
                & (observed > float(self.config["min_depth"]))
                & (observed < float(self.config["max_depth"]))
            )
            if occ_masks is not None and occ_masks[index] is not None:
                occ = torch.from_numpy(
                    np.asarray(occ_masks[index]) > 0
                ).to(device)
                valid_pixels &= ~occ
            valid = active & in_image & valid_pixels[vi, ui]

            c2w_norm = torch.from_numpy(
                self.normalization.normalize_c2w(frame.c2w_cv).astype(
                    np.float32
                )
            ).to(device)
            K_t = torch.from_numpy(frame.K.astype(np.float32)).to(device)
            front_depth, front_alpha = self._geometric_front_depth(
                c2w_norm, K_t, width, height
            )

            camera_center = c2w[:3, 3]
            view_dir = F.normalize(
                means_metric - camera_center[None, :], dim=-1, eps=1e-8
            )
            view_abs_cos = (normals * view_dir).sum(dim=-1).abs()
            evidence = depth_evidence_masks(
                gaussian_depth=depth_cam,
                observed_depth=observed[vi, ui],
                valid_observation=valid,
                geometric_front_depth=front_depth[vi, ui],
                geometric_alpha=front_alpha[vi, ui],
                depth_tolerance=thresholds.depth_tolerance_m,
                geometric_alpha_threshold=thresholds.geometric_alpha_threshold,
                view_abs_cos=view_abs_cos,
                grazing_tolerance_cap=thresholds.grazing_tolerance_cap,
            )
            supported |= evidence["support"]
            accumulate_support_residuals(
                fields, evidence["support"], depth_cam, observed[vi, ui],
                normals, view_dir,
            )
            candidate = evidence["free_space"] | evidence["behind_miss"]
            accepted = independent_view_mask(
                candidate, fields.conflict_count, fields.last_conflict_view,
                view_dir, thresholds.min_view_angle_deg,
            )
            if bool(accepted.any()):
                fields.conflict_count[accepted] += 1
                fields.last_conflict_view[accepted] = view_dir[accepted]

            totals["projected_valid"] += int(valid.sum())
            for key in ("support", "free_space", "behind_occluded",
                        "behind_miss"):
                totals[key] += int(evidence[key].sum())
            totals["accepted_conflict"] += int(accepted.sum())

        masks = apply_transitions(fields, supported, thresholds)
        newly_contradicted = masks["to_contradict"]
        if bool(newly_contradicted.any()):
            self.splats["opacities"].data[newly_contradicted] = -10.0
        record = {
            "event": str(event),
            "n_frames": len(frames),
            "new_verified": int(masks["to_verify"].sum()),
            "new_suspect": int(masks["verified_to_suspect"].sum()),
            "new_contradicted": int(newly_contradicted.sum()),
            **totals,
            "state": fields.summary(),
        }
        self.lifecycle_log.append(record)
        logging.info("[Prior lifecycle] " + json.dumps(record, sort_keys=True))
        return record

    @torch.no_grad()
    def _remove_contradicted(self) -> int:
        if not self.lifecycle_enabled or self.lifecycle_fields is None:
            return 0
        drop = self.lifecycle_fields.state == STATE_CONTRADICTED
        n_drop = int(drop.sum())
        if n_drop == 0:
            return 0
        keep = ~drop
        self._set_splats({
            name: tensor.detach()[keep].cpu()
            for name, tensor in self.splats.items()
        })
        self.lifecycle_fields = self.lifecycle_fields.keep(keep)
        return n_drop

    def refresh_view_poses(
        self, poses_by_frame_id: Mapping[str, np.ndarray]
    ) -> int:
        """Adopt updated tracker poses for already-stored views.

        The online tracker keeps bundle-adjusting past keyframes; this
        applies the newest metric OpenCV c2w poses to matching views.
        """

        refreshed = 0
        for view in self.views:
            pose = poses_by_frame_id.get(view.frame_id)
            if pose is None:
                continue
            normalized = self.normalization.normalize_c2w(pose)
            view.c2w_normalized = torch.from_numpy(
                normalized.astype(np.float32)
            ).contiguous()
            refreshed += 1
        return refreshed

    @torch.no_grad()
    def export_state_ply(self, path: str | Path) -> None:
        """Debug PLY with lifecycle-state colors (metric object frame)."""

        colors_by_state = np.array(
            [[128, 128, 128], [40, 200, 60], [240, 200, 40], [220, 40, 40]],
            dtype=np.uint8,
        )
        means = self.normalization.metric_points(
            self.splats["means"].detach().cpu().numpy()
        )
        if self.lifecycle_fields is not None:
            state = self.lifecycle_fields.state.cpu().numpy()
        else:
            state = np.full(len(means), STATE_VERIFIED, dtype=np.int8)
        rgb = colors_by_state[state.clip(0, 3)]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ("ply\nformat ascii 1.0\n"
                  f"element vertex {len(means)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\n"
                  "property uchar blue\nend_header\n")
        with path.open("w") as f:
            f.write(header)
            for p, c in zip(means, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                        f"{c[0]} {c[1]} {c[2]}\n")

    def _masked_depth_loss(
        self, result: RasterResult, view: TrainingView, mask: torch.Tensor
    ) -> torch.Tensor:
        if view.depth is None:
            raise RuntimeError(
                "depth_loss_weight > 0 requires views with stored depth; "
                f"frame {view.frame_id} has none (old checkpoint?)"
            )
        if result.colors.shape[-1] < 4:
            raise RuntimeError("Depth loss requires an ED depth render channel")
        depth_gt_metric = view.depth.to(self.device, non_blocking=True)[None]
        rendered_depth = result.colors[..., 3]
        valid = (
            mask
            & torch.isfinite(depth_gt_metric)
            & (depth_gt_metric >= float(self.config["min_depth"]))
            & (depth_gt_metric <= float(self.config["max_depth"]))
            & (result.alpha[..., 0] > float(self.config["depth_alpha_threshold"]))
        )
        if not bool(valid.any()):
            return torch.zeros((), device=rendered_depth.device,
                               dtype=rendered_depth.dtype)
        depth_gt = depth_gt_metric * self.normalization.scale
        delta = float(self.config["depth_huber_delta_m"]) * self.normalization.scale
        return torch.nn.functional.huber_loss(
            rendered_depth[valid], depth_gt[valid], delta=delta
        )

    def train(
        self,
        steps: int,
        *,
        lr_decay_horizon_steps: int | None = None,
        optimize_poses: bool | None = None,
    ) -> tuple[float | None, float | None]:
        if self.splats is None or not self.views:
            raise RuntimeError("Runner requires splats and training views")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if lr_decay_horizon_steps is not None and lr_decay_horizon_steps <= 0:
            raise ValueError("lr_decay_horizon_steps must be positive")
        if steps == 0:
            return None, None
        self._require_gsplat()
        means_optimizer = self.optimizers["means"]
        means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            means_optimizer,
            gamma=0.01
            ** (
                1.0
                / float(
                    steps
                    if lr_decay_horizon_steps is None
                    else lr_decay_horizon_steps
                )
            ),
        )
        first_loss: float | None = None
        final_loss: float | None = None
        frozen_rows: torch.Tensor | None = None
        if self.lifecycle_enabled and self.lifecycle_fields is not None:
            frozen = self.lifecycle_fields.state != STATE_VERIFIED
            if bool(frozen.any()):
                frozen_rows = frozen.to(self.device)

        feedback_cfg = self.config["pose_feedback"]
        if optimize_poses is None:
            optimize_poses = bool(feedback_cfg["enabled"])
        pose_deltas = pose_optimizer = pose_scheduler = None
        if optimize_poses:
            # ⑤: fresh zero deltas for every training call (original
            # BundleSDF re-creates its PoseArray per cycle likewise).
            pose_deltas = self._new_pose_deltas(
                len(self.views), fix_first=bool(feedback_cfg["fix_first_view"])
            )
            pose_optimizer, pose_scheduler = self._pose_optimizer(
                pose_deltas, float(feedback_cfg["lr"]), steps
            )

        for _ in range(steps):
            strategy_step = self.strategy_step
            view_index = int(
                torch.randint(
                    len(self.views), (1,), generator=self._generator
                ).item()
            )
            view = self.views[view_index]
            target = view.rgb.to(self.device, non_blocking=True)[None]
            mask = view.mask.to(self.device, non_blocking=True)[None]
            K = view.K.to(self.device, non_blocking=True)[None]
            c2w = view.c2w_normalized.to(self.device, non_blocking=True)[None]
            if pose_deltas is not None:
                c2w = pose_deltas.matrices([view_index]) @ c2w

            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            if pose_optimizer is not None:
                pose_optimizer.zero_grad(set_to_none=True)
            active_sh_degree = min(
                self.total_steps // int(self.config["sh_degree_interval"]),
                int(self.config["sh_degree"]),
            )
            depth_weight = float(self.config["depth_loss_weight"])
            result = self._rasterize(
                K=K,
                c2w=c2w,
                width=view.width,
                height=view.height,
                sh_degree=active_sh_degree,
                render_mode="RGB+ED" if (depth_weight > 0 or MAP_DEPTH_WEIGHT or POSE_DEPTH_WEIGHT) else "RGB",
                absgrad=bool(self.strategy.absgrad),
            )
            info = result.info
            self.strategy.step_pre_backward(
                self.splats,
                self.optimizers,
                self.strategy_state,
                strategy_step,
                info,
            )
            rendered = result.colors[..., :3]
            mask_channels = mask[..., None].to(rendered.dtype)
            denominator = mask_channels.sum().clamp_min(1.0) * 3.0
            l1 = (torch.abs(rendered - target) * mask_channels).sum() / denominator
            dssim = self._masked_dssim(rendered, target, mask)
            ssim_weight = float(self.config["ssim_weight"])
            colour_loss = (1.0 - ssim_weight) * l1 + ssim_weight * dssim
            split = (MAP_DEPTH_WEIGHT is not None or POSE_DEPTH_WEIGHT is not None) and pose_optimizer is not None
            w_map = MAP_DEPTH_WEIGHT if (split and MAP_DEPTH_WEIGHT is not None) else depth_weight
            w_pose = POSE_DEPTH_WEIGHT if (split and POSE_DEPTH_WEIGHT is not None) else depth_weight
            depth_term = self._masked_depth_loss(result, view, mask) if max(w_map, w_pose) > 0 else None
            loss = colour_loss + (w_map * depth_term if (w_map > 0 and depth_term is not None) else 0.0)
            normal_weight = float(self.config["normal_consistency_weight"])
            if (
                result.normals is not None
                and normal_weight > 0
                and self.total_steps
                >= int(self.config["normal_consistency_start_step"])
            ):
                weight_map = result.alpha[..., 0] * mask.to(result.alpha.dtype)
                consistency = 1.0 - (
                    result.normals * result.surf_normals
                ).sum(dim=-1)
                loss = loss + normal_weight * (
                    (consistency * weight_map).sum()
                    / weight_map.sum().clamp_min(1.0)
                )
            distortion_weight = float(self.config["distortion_weight"])
            if (
                result.distort is not None
                and distortion_weight > 0
                and self.total_steps >= int(self.config["distortion_start_step"])
            ):
                mask_float = mask.to(result.distort.dtype)
                loss = loss + distortion_weight * (
                    (result.distort[..., 0] * mask_float).sum()
                    / mask_float.sum().clamp_min(1.0)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Gaussian training loss")
            if split:
                # pose deltas: gradient of the pose-weighted loss; map (and gsplat's screen-space means2d grad that the
                # strategy reads in step_post_backward): full backward of the map-weighted loss, then the pose grads are
                # overwritten with the pose-loss grads.  (A backward restricted with inputs= left means2d.grad None.)
                pose_loss = colour_loss + (w_pose * depth_term if (w_pose > 0 and depth_term is not None) else 0.0)
                pose_params = [p for g in pose_optimizer.param_groups for p in g["params"]]
                pose_grads = torch.autograd.grad(pose_loss, pose_params, retain_graph=True, allow_unused=True)
                loss.backward()
                for p, g in zip(pose_params, pose_grads):
                    p.grad = None if g is None else g.detach().clone()
            else:
                loss.backward()
            if frozen_rows is not None:
                # Lifecycle: only VERIFIED splats may learn.
                for parameter in self.splats.values():
                    if parameter.grad is not None:
                        parameter.grad[frozen_rows] = 0
            for optimizer in self.optimizers.values():
                optimizer.step()
            means_scheduler.step()
            if pose_deltas is not None:
                self._pose_step(pose_deltas, pose_optimizer, pose_scheduler)
            self._strategy_post_backward(strategy_step, info)
            self.strategy_step += 1
            self.total_steps += 1
            final_loss = float(loss.detach().cpu())
            if first_loss is None:
                first_loss = final_loss
        if pose_deltas is not None:
            self._bake_pose_deltas(
                pose_deltas, self.views, event=f"train_{self.update_index:03d}"
            )
        return first_loss, final_loss

    # ---- milestone ⑤: pose feedback helpers --------------------------------

    def _new_pose_deltas(self, num_views: int, *, fix_first: bool) -> PoseDeltas:
        feedback_cfg = self.config["pose_feedback"]
        return PoseDeltas(
            num_views,
            max_trans_norm=float(feedback_cfg["max_trans_m"])
            * float(self.normalization.scale),
            max_rot_rad=math.radians(float(feedback_cfg["max_rot_deg"])),
            fix_first=fix_first,
            device=self.device,
        )

    def _pose_optimizer(
        self, deltas: PoseDeltas, lr: float, steps: int
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.Adam(
            [{"params": [deltas.data], "lr": float(lr), "name": "pose_deltas"}],
            eps=1e-15,
        )
        decay = float(self.config["pose_feedback"]["lr_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=decay ** (1.0 / float(max(int(steps), 1)))
        )
        return optimizer, scheduler

    def _pose_step(
        self,
        deltas: PoseDeltas,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        max_norm = float(self.config["pose_feedback"]["grad_max_norm"])
        if deltas.data.grad is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                deltas.data, max_norm=max_norm, norm_type=float("inf")
            )
        optimizer.step()
        scheduler.step()

    @torch.no_grad()
    def _bake_pose_deltas(
        self, deltas: PoseDeltas, views: Sequence[TrainingView], event: str
    ) -> dict[str, Any]:
        """Fold the deltas into the views' stored poses and log magnitudes."""

        T = deltas.matrices(
            torch.arange(len(views), device=self.device)
        ).detach().cpu().double().numpy()
        for index, view in enumerate(views):
            c2w = T[index] @ view.c2w_normalized.double().numpy()
            # Re-project the rotation onto SO(3) so float32 products cannot
            # drift past validate_c2w_cv's tolerance over many cycles.
            u, _, vt = np.linalg.svd(c2w[:3, :3])
            rotation = u @ vt
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1.0
                rotation = u @ vt
            c2w[:3, :3] = rotation
            view.c2w_normalized = torch.from_numpy(
                c2w.astype(np.float32)
            ).contiguous()
        rot, trans = deltas.magnitudes()
        rot_deg = torch.rad2deg(rot).cpu()
        trans_mm = (trans / float(self.normalization.scale) * 1000.0).cpu()
        record = {
            "event": event,
            "views": len(views),
            "max_rot_deg": float(rot_deg.max()),
            "mean_rot_deg": float(rot_deg.mean()),
            "max_trans_mm": float(trans_mm.max()),
            "mean_trans_mm": float(trans_mm.mean()),
            "per_view_rot_deg": [round(float(v), 4) for v in rot_deg],
            "per_view_trans_mm": [round(float(v), 4) for v in trans_mm],
        }
        self.feedback_log.append(record)
        return record

    def view_poses_metric(self) -> np.ndarray:
        """Current view poses in tracker order as metric OpenCV c2w, [N,4,4]."""

        if not self.views:
            return np.empty((0, 4, 4), dtype=np.float32)
        return np.stack(
            [
                self.normalization.metric_c2w(view.c2w_normalized.numpy())
                for view in self.views
            ]
        ).astype(np.float32)

    @torch.no_grad()
    def render(
        self, view_index: int, include_depth: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Render RGB, alpha, and optional metric camera-z depth.

        Depth is zero where alpha is zero; callers should use alpha as its
        validity mask.
        """
        if self.splats is None:
            raise RuntimeError("Runner is not initialized")
        view = self.views[view_index]
        K = view.K.to(self.device)[None]
        c2w = view.c2w_normalized.to(self.device)[None]
        mode = "RGB+ED" if include_depth else "RGB"
        result = self._rasterize(
            K=K,
            c2w=c2w,
            width=view.width,
            height=view.height,
            sh_degree=min(
                self.total_steps // int(self.config["sh_degree_interval"]),
                int(self.config["sh_degree"]),
            ),
            render_mode=mode,
            absgrad=False,
        )
        rendered = result.colors
        alpha = result.alpha
        depth_metric = None
        rgb = rendered[..., :3]
        if include_depth:
            depth_normalized = rendered[..., 3].squeeze(0)
            depth_metric = (
                depth_normalized / self.normalization.scale
            ).detach().cpu().numpy()
        return (
            rgb.squeeze(0).detach().cpu().numpy(),
            alpha.squeeze(0).squeeze(-1).detach().cpu().numpy(),
            depth_metric,
        )

    def _checkpoint_views(self) -> list[dict[str, Any]]:
        return [
            {
                "frame_id": view.frame_id,
                "rgb": view.rgb.cpu(),
                "mask": view.mask.cpu(),
                "K": view.K.cpu(),
                "c2w_normalized": view.c2w_normalized.cpu(),
                "crop_xyxy": list(view.crop_xyxy),
                "depth": None if view.depth is None else view.depth.cpu(),
            }
            for view in self.views
        ]

    def save_checkpoint(self, path: str | Path) -> None:
        if self.splats is None:
            raise RuntimeError("Runner is not initialized")
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        def to_cpu(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: to_cpu(item) for key, item in value.items()}
            if isinstance(value, list):
                return [to_cpu(item) for item in value]
            if isinstance(value, tuple):
                return tuple(to_cpu(item) for item in value)
            return value

        payload = {
            "format_version": 1,
            "gsplat_version": GSPLAT_VERSION,
            "config": copy.deepcopy(self.config),
            "normalization": {
                "scale": self.normalization.scale,
                "translation": torch.from_numpy(
                    self.normalization.translation.astype(np.float32)
                ),
            },
            "splats": {
                name: value.detach().cpu() for name, value in self.splats.items()
            },
            "optimizers": {
                name: to_cpu(optimizer.state_dict())
                for name, optimizer in self.optimizers.items()
            },
            "strategy_state": to_cpu(self.strategy_state),
            "views": self._checkpoint_views(),
            "lifecycle_fields": (
                {name: tensor.detach().cpu()
                 for name, tensor in vars(self.lifecycle_fields).items()}
                if self.lifecycle_fields is not None else None
            ),
            "observed_points_metric": torch.from_numpy(
                self.observed_points_metric
            ),
            "observed_colors": torch.from_numpy(self.observed_colors),
            "total_steps": self.total_steps,
            "update_index": self.update_index,
            "strategy_step": self.strategy_step,
            "generator_state": self._generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if self.device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
        }
        torch.save(payload, checkpoint_path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str | None = None,
    ) -> "GaussianRunner":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported Gaussian checkpoint format")
        if payload.get("gsplat_version") != GSPLAT_VERSION:
            raise ValueError("Checkpoint gsplat version does not match runner")
        norm_payload = payload["normalization"]
        normalization = SceneNormalization(
            scale=float(norm_payload["scale"]),
            translation=norm_payload["translation"].numpy(),
        )
        runner = cls(payload["config"], normalization, device=device)
        runner._set_splats(payload["splats"])
        runner.views = [
            TrainingView(
                frame_id=item["frame_id"],
                rgb=item["rgb"].contiguous(),
                mask=item["mask"].bool().contiguous(),
                K=item["K"].contiguous(),
                c2w_normalized=item["c2w_normalized"].contiguous(),
                crop_xyxy=tuple(int(value) for value in item["crop_xyxy"]),
                depth=(
                    item["depth"].contiguous()
                    if item.get("depth") is not None
                    else None
                ),
            )
            for item in payload["views"]
        ]
        fields_payload = payload.get("lifecycle_fields")
        if fields_payload is not None:
            runner.lifecycle_fields = LifecycleFields(
                **{name: tensor.to(runner.device)
                   for name, tensor in fields_payload.items()}
            ).validated()
            if len(runner.lifecycle_fields) != runner.num_gaussians:
                raise ValueError(
                    "Checkpoint lifecycle fields do not align with splats"
                )
        elif runner.lifecycle_enabled:
            raise ValueError(
                "prior_lifecycle.enabled but the checkpoint has no "
                "lifecycle fields"
            )
        runner.observed_points_metric = payload[
            "observed_points_metric"
        ].numpy().copy()
        runner.observed_colors = payload["observed_colors"].numpy().copy()
        runner.total_steps = int(payload["total_steps"])
        runner.update_index = int(payload["update_index"])
        lr_scale = (
            float(runner.config["initial_lr_scale"])
            if runner.update_index == 0
            else float(runner.config["update_lr_scale"])
        )
        runner._reset_optimization_state(
            lr_scale, initial=runner.update_index == 0
        )
        for name, optimizer_state in payload["optimizers"].items():
            runner.optimizers[name].load_state_dict(optimizer_state)
        runner.strategy_state = {
            key: value.to(runner.device) if isinstance(value, torch.Tensor) else value
            for key, value in payload["strategy_state"].items()
        }
        runner.strategy_step = int(payload["strategy_step"])
        runner._generator.set_state(payload["generator_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if payload["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        return runner

    @torch.no_grad()
    def export_ply(
        self,
        normalized_path: str | Path,
        metric_path: str | Path | None = None,
    ) -> None:
        if self.splats is None:
            raise RuntimeError("Runner is not initialized")
        gsplat, _, _, _ = self._require_gsplat()
        normalized_path = Path(normalized_path)
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        values = {name: value.detach() for name, value in self.splats.items()}
        for name, value in values.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Cannot export non-finite {name}")
        gsplat.export_splats(
            means=values["means"],
            scales=values["scales"],
            quats=values["quats"],
            opacities=values["opacities"],
            sh0=values["sh0"],
            shN=values["shN"],
            format="ply",
            save_to=str(normalized_path),
        )
        if metric_path is not None:
            metric_path = Path(metric_path)
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            means_metric = torch.from_numpy(
                self.normalization.metric_points(
                    values["means"].detach().cpu().numpy()
                )
            ).to(values["means"].device)
            scales_metric = values["scales"] - math.log(self.normalization.scale)
            gsplat.export_splats(
                means=means_metric,
                scales=scales_metric,
                quats=values["quats"],
                opacities=values["opacities"],
                sh0=values["sh0"],
                shN=values["shN"],
                format="ply",
                save_to=str(metric_path),
            )
