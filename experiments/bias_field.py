"""EXPERIMENT ③b: bias-extrapolation correction — pure field math.

Estimate the SAM3D shape-residual field measured on VERIFIED surfels (the
lifecycle's support-residual accumulator) and extrapolate it to UNSEEN
surfels:

  stage 1  global anisotropic residual  δ ≈ n · (diag(s)·x + t)
           (6 DoF, robust; Sim(3) alignment only fits an isotropic scale, so
           per-axis fatness/thinness of the SAM3D shape lands here)
  stage 2  local residual via k-NN-graph harmonic interpolation (Jacobi),
           confidence-damped by distance to the nearest measured anchor

Applied as means += δ · normal, capped. All in metric object coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BiasFieldConfig:
    min_residual_count: int = 3
    trim_sigma: float = 3.0
    knn: int = 8
    jacobi_iters: int = 200
    damping_sigma_m: float = 0.015  # local trust decays with anchor distance
    offset_cap_m: float = 0.015
    global_stage: bool = True


def robust_anchor_residuals(
    residual_sum: torch.Tensor,
    residual_count: torch.Tensor,
    min_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-splat mean residual where enough support observations exist."""

    count = residual_count.to(torch.float32)
    has = residual_count >= int(min_count)
    mean = torch.zeros_like(residual_sum)
    mean[has] = residual_sum[has] / count[has].clamp_min(1.0)
    return mean, has


def fit_global_affine_residual(
    means: torch.Tensor,
    normals: torch.Tensor,
    residuals: torch.Tensor,
    anchors: torch.Tensor,
    trim_sigma: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit of δ ≈ n·(diag(s)·x + t) on the anchors.

    Returns (params[6] = (s1,s2,s3,t1,t2,t3), predicted δ for ALL splats).
    One trimming round removes >trim_sigma outliers. Falls back to zeros if
    the system is under-constrained.
    """

    idx = torch.where(anchors)[0]
    if len(idx) < 24:
        zeros = torch.zeros(6, dtype=means.dtype)
        return zeros, torch.zeros(len(means), dtype=means.dtype)

    def design(rows):
        n = normals[rows]
        x = means[rows]
        return torch.cat((n * x, n), dim=1)  # [k, 6]

    def solve(rows):
        A = design(rows)
        b = residuals[rows]
        solution = torch.linalg.lstsq(A, b.unsqueeze(-1))
        return solution.solution.squeeze(-1)

    params = solve(idx)
    residual_fit = design(idx) @ params - residuals[idx]
    sigma = residual_fit.std().clamp_min(1e-9)
    keep = residual_fit.abs() <= float(trim_sigma) * sigma
    if int(keep.sum()) >= 24 and int(keep.sum()) < len(idx):
        params = solve(idx[keep])
    predicted = design(torch.arange(len(means))) @ params
    return params, predicted


def knn_graph(means: torch.Tensor, k: int) -> torch.Tensor:
    """Symmetric-enough kNN neighbor index table [N, k] (self excluded)."""

    from scipy.spatial import cKDTree

    pts = means.detach().cpu().numpy().astype(np.float64)
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k + 1, workers=-1)
    return torch.from_numpy(indices[:, 1:].astype(np.int64))


def harmonic_interpolate(
    neighbor_idx: torch.Tensor,
    anchor_mask: torch.Tensor,
    anchor_values: torch.Tensor,
    iters: int,
) -> torch.Tensor:
    """Jacobi relaxation: anchors fixed, the rest converge to neighbor means."""

    values = torch.zeros(len(anchor_mask), dtype=anchor_values.dtype)
    values[anchor_mask] = anchor_values[anchor_mask]
    free = ~anchor_mask
    for _ in range(int(iters)):
        neighbor_mean = values[neighbor_idx].mean(dim=1)
        values = torch.where(free, neighbor_mean, values)
    return values


def anchor_distance_damping(
    means: torch.Tensor,
    anchor_mask: torch.Tensor,
    sigma_m: float,
) -> torch.Tensor:
    """exp(−d²/2σ²) weight from each splat to its nearest anchor."""

    from scipy.spatial import cKDTree

    anchors = means[anchor_mask].detach().cpu().numpy().astype(np.float64)
    if len(anchors) == 0:
        return torch.zeros(len(means), dtype=means.dtype)
    tree = cKDTree(anchors)
    d, _ = tree.query(means.detach().cpu().numpy().astype(np.float64),
                      k=1, workers=-1)
    d = torch.from_numpy(d).to(means.dtype)
    return torch.exp(-0.5 * (d / float(sigma_m)) ** 2)


def estimate_bias_offsets(
    means: torch.Tensor,
    normals: torch.Tensor,
    residual_sum: torch.Tensor,
    residual_count: torch.Tensor,
    config: BiasFieldConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full ③b pipeline → per-splat target offset δ̂ (metric, capped).

    All inputs are prior-lineage splats in metric object coordinates.
    """

    anchors_value, anchors = robust_anchor_residuals(
        residual_sum, residual_count, config.min_residual_count
    )
    info: dict[str, float] = {
        "anchors": float(anchors.sum()),
        "anchor_residual_mean_mm": (
            float(anchors_value[anchors].mean() * 1000) if anchors.any()
            else 0.0
        ),
    }
    if not bool(anchors.any()):
        return torch.zeros(len(means)), info

    predicted_global = torch.zeros(len(means), dtype=means.dtype)
    if config.global_stage:
        params, predicted_global = fit_global_affine_residual(
            means, normals, anchors_value, anchors, config.trim_sigma
        )
        info["global_params"] = [round(float(v), 6) for v in params]

    local_anchor_values = anchors_value - predicted_global
    neighbor_idx = knn_graph(means, config.knn)
    local = harmonic_interpolate(
        neighbor_idx, anchors, local_anchor_values, config.jacobi_iters
    )
    damping = anchor_distance_damping(means, anchors, config.damping_sigma_m)
    offsets = predicted_global + damping * local
    # Anchored splats know their own residual exactly.
    offsets[anchors] = anchors_value[anchors]
    offsets = offsets.clamp(-config.offset_cap_m, config.offset_cap_m)
    info["offset_mean_mm"] = float(offsets.abs().mean() * 1000)
    info["offset_p95_mm"] = float(offsets.abs().quantile(0.95) * 1000)
    info["damping_mean"] = float(damping.mean())
    return offsets, info
