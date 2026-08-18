"""Persistent 3D Gaussian Splatting reconstruction primitives.

The runner deliberately keeps the public boundary in BundleTrack's native
coordinate convention: OpenCV camera-to-object poses in metric units.  Scene
normalization and conversion to gsplat world-to-camera matrices happen only
inside this module.

The first integration milestone uses fixed camera poses.  Learned Gaussian
parameters persist across reconstruction updates, while optimizers and the
densification strategy are recreated for every reconstruction update.  This
keeps the representation persistent without depending on private
optimizer-resize behavior in gsplat.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree


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

    def _require_gsplat(self) -> tuple[Any, Any, Any]:
        try:
            import gsplat
            from gsplat.rendering import rasterization
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
        return gsplat, rasterization, DefaultStrategy

    def _frames_to_cloud(
        self, frames: Sequence[GaussianFrame]
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        point_batches: list[np.ndarray] = []
        color_batches: list[np.ndarray] = []
        raw_count = 0
        for frame in frames:
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
        K_crop = crop_intrinsics(frame.K, x0=x0, y0=y0).astype(np.float32)
        c2w_normalized = self.normalization.normalize_c2w(frame.c2w_cv)
        return TrainingView(
            frame_id=frame.frame_id,
            rgb=rgb_tensor.contiguous(),
            mask=mask_tensor.contiguous(),
            K=torch.from_numpy(K_crop).contiguous(),
            c2w_normalized=torch.from_numpy(c2w_normalized).contiguous(),
            crop_xyxy=(x0, y0, x1, y1),
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
        _, _, DefaultStrategy = self._require_gsplat()
        base_lrs = self.config["learning_rates"]
        self.optimizers = {
            name: torch.optim.Adam(
                [
                    {
                        "params": [self.splats[name]],
                        "lr": float(base_lrs[name]) * lr_scale,
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
        }

        try:
            self._append_splats(novel_points, novel_colors)
            after_append = self.num_gaussians
            self.views.extend(self._prepare_view(frame) for frame in validated)
            self.update_index += 1
            self._reset_optimization_state(
                float(self.config["update_lr_scale"]), initial=False
            )
            steps = int(
                self.config["update_steps"] if train_steps is None else train_steps
            )
            first_loss, final_loss = self.train(steps)
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

    def train(
        self,
        steps: int,
        *,
        lr_decay_horizon_steps: int | None = None,
    ) -> tuple[float | None, float | None]:
        if self.splats is None or not self.views:
            raise RuntimeError("Runner requires splats and training views")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if lr_decay_horizon_steps is not None and lr_decay_horizon_steps <= 0:
            raise ValueError("lr_decay_horizon_steps must be positive")
        if steps == 0:
            return None, None
        _, rasterization, _ = self._require_gsplat()
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

            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            active_sh_degree = min(
                self.total_steps // int(self.config["sh_degree_interval"]),
                int(self.config["sh_degree"]),
            )
            colors = torch.cat((self.splats["sh0"], self.splats["shN"]), dim=1)
            rendered, _, info = rasterization(
                means=self.splats["means"],
                quats=self.splats["quats"],
                scales=torch.exp(self.splats["scales"]),
                opacities=torch.sigmoid(self.splats["opacities"]),
                colors=colors,
                viewmats=torch.linalg.inv(c2w),
                Ks=K,
                width=view.width,
                height=view.height,
                sh_degree=active_sh_degree,
                packed=bool(self.config["packed"]),
                sparse_grad=False,
                absgrad=bool(self.strategy.absgrad),
                render_mode="RGB",
                rasterize_mode=str(self.config["rasterize_mode"]),
                camera_model="pinhole",
            )
            self.strategy.step_pre_backward(
                self.splats,
                self.optimizers,
                self.strategy_state,
                strategy_step,
                info,
            )
            mask_channels = mask[..., None].to(rendered.dtype)
            denominator = mask_channels.sum().clamp_min(1.0) * 3.0
            l1 = (torch.abs(rendered - target) * mask_channels).sum() / denominator
            dssim = self._masked_dssim(rendered, target, mask)
            ssim_weight = float(self.config["ssim_weight"])
            loss = (1.0 - ssim_weight) * l1 + ssim_weight * dssim
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Gaussian training loss")
            loss.backward()
            for optimizer in self.optimizers.values():
                optimizer.step()
            means_scheduler.step()
            self._strategy_post_backward(strategy_step, info)
            self.strategy_step += 1
            self.total_steps += 1
            final_loss = float(loss.detach().cpu())
            if first_loss is None:
                first_loss = final_loss
        return first_loss, final_loss

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
        _, rasterization, _ = self._require_gsplat()
        view = self.views[view_index]
        K = view.K.to(self.device)[None]
        c2w = view.c2w_normalized.to(self.device)[None]
        colors = torch.cat((self.splats["sh0"], self.splats["shN"]), dim=1)
        mode = "RGB+ED" if include_depth else "RGB"
        rendered, alpha, _ = rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=torch.linalg.inv(c2w),
            Ks=K,
            width=view.width,
            height=view.height,
            sh_degree=min(
                self.total_steps // int(self.config["sh_degree_interval"]),
                int(self.config["sh_degree"]),
            ),
            packed=bool(self.config["packed"]),
            render_mode=mode,
            rasterize_mode=str(self.config["rasterize_mode"]),
            camera_model="pinhole",
        )
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
            )
            for item in payload["views"]
        ]
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
        gsplat, _, _ = self._require_gsplat()
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
