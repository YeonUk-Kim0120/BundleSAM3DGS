"""First-frame SAM3D mesh-prior R/T/s alignment with the 2DGS renderer.

Freezes the surfel prior (geometry/colors/opacity) and optimizes only a
Sim(3) correction (rotation/translation/log-scale deltas) on top of SAM3D's
initial canonical→camera pose, by render-and-compare against the first
frame's RGB-D + mask.

The optimization recipe is a faithful port of the validated BundleGS
implementation in the sibling checkout
(/home/kist/Desktop/BundleSDF/gaussian_runner.py: ``_rts_parameters``,
``_render_sam3d_rts``, ``_sam3d_refine_loss``, ``_refine_sam3d_rts``,
``_compose_sam3d_refined_pose``), with the 3DGS rasterizer replaced by
gsplat's 2DGS surfel rasterizer.

Example (the adopted default recipe: raw depth + transferred gaussian
colors + SSIM w1.0):
    python3 run_sam3d_alignment.py \
      --track-dir logs/mustard0_bundlesam3dgs_..._20260811 \
      --depth-dir datasets/YCBInEOAT/mustard0/depth \
      --mesh-npz .../mustard0_mesh_depth.npz \
      --pose-json .../mustard0_mesh_depth.json \
      --gaussian-ply .../mustard0_splat_depth.ply \
      --output-dir logs/mustard0_sam3d_alignment_<date>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from gaussian_runner import crop_intrinsics, resize_intrinsics, validate_intrinsics
from sam3d_prior import (
    Sim3Pose,
    load_mesh_prior,
    load_sam3d_gaussian_ply,
    load_sam3d_pose,
    quats_from_normals,
    sample_surfels,
    transfer_gaussian_colors,
)


DEFAULT_CONFIG: dict[str, Any] = {
    # Surfel prior (radius 0.75 adopted from the 2026-08-31 radius ablation)
    "surfel_count": 20000,
    "surfel_seed": 0,
    "surfel_radius_multiplier": 0.75,
    "surfel_opacity": 0.9,
    # Optimization (sibling gs_sam3d_refine_* defaults)
    "steps": 400,
    "lr": 5.0e-3,
    "end_lr": 0.0,
    "warmup": 10,
    "grad_clip": 1.0,
    "max_rot_deg": 60.0,
    "max_translation_m": 0.24,
    "max_scale_delta": 2.7,
    # Loss weights.  Photo defaults are OFF and are enabled (SSIM only,
    # w_photo 1.0) when --gaussian-ply supplies transferred colors — the "C"
    # recipe adopted 2026-08-31 (22-seq YCB+HO3D A/C study, see
    # SAM3D_ALIGNMENT_RESULTS.md).  Mesh-color+SSIM was ineffective (arm D).
    "use_ssim": False,
    "use_ms_ssim": False,
    "w_depth": 80.0,
    "w_photo": 1.0,
    "w_coverage": 2.0,
    "w_visibility": 50.0,
    "w_outside_alpha": 0.2,
    "w_reg_rot": 0.02,
    "w_reg_t": 1.0,
    "w_reg_scale": 0.1,
    # Loss thresholds
    "min_depth_m": 0.05,
    "max_depth_m": 5.0,
    "depth_alpha_threshold": 0.03,
    "depth_mask_threshold": 0.25,
    "depth_weight_min": 0.05,
    "depth_huber_delta_m": 0.03,
    "depth_residual_clip_m": 0.15,
    "coverage_alpha_target": 0.6,
    "coverage_mask_threshold": 0.25,
    "outside_mask_threshold": 0.1,
    "min_visible_ratio": 0.6,
    "best_min_visible_ratio": 0.5,
    "best_min_depth_valid_ratio": 0.3,
    # Rendering
    "render_size": 224,
    "roi_padding_px": 24,
    "near_plane": 0.01,
    "far_plane": 10.0,
    "eps2d": 0.3,
    "log_every": 100,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track-dir", type=Path, required=True,
                        help="Tracking output with color/depth_filtered/mask/cam_K.txt")
    parser.add_argument("--depth-dir", type=Path, default=None,
                        help="Override depth directory (e.g. the dataset's raw "
                             "depth); defaults to <track-dir>/depth_filtered. "
                             "BundleTrack's filtering can erase large depth "
                             "patches on some first frames.")
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, default=None,
                        help="SAM3D <seq>_splat_depth.ply (same canonical "
                             "frame). When given, gaussian colors are "
                             "transferred onto the surfels and the SSIM photo "
                             "term is enabled (the adopted default recipe).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; it must not already exist")
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional YAML overriding DEFAULT_CONFIG keys")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        from ruamel.yaml import YAML

        with path.open("r", encoding="utf-8") as f:
            loaded = YAML(typ="safe").load(f) or {}
        unknown = set(loaded) - set(config)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        config.update(loaded)
    return config


def load_first_frame(
    track_dir: Path, depth_dir: Path | None = None
) -> dict[str, Any]:
    """First frame in the saved-tracking-log layout used by the GS replay."""

    from PIL import Image

    color_dir = track_dir / "color"
    names = sorted(p.name for p in color_dir.iterdir())
    if not names:
        raise FileNotFoundError(f"No frames under {color_dir}")
    name = names[0]
    rgb = np.array(Image.open(color_dir / name).convert("RGB"))
    depth_path = (
        (depth_dir / name) if depth_dir is not None
        else track_dir / "depth_filtered" / name
    )
    depth_raw = np.array(Image.open(depth_path))
    if depth_raw.dtype.kind not in "iu":
        raise ValueError("Saved depth must be integer millimeters")
    depth_m = depth_raw.astype(np.float32) / 1000.0
    mask = np.array(Image.open(track_dir / "mask" / name))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 0
    K = validate_intrinsics(np.loadtxt(track_dir / "cam_K.txt").reshape(3, 3))
    return {"frame_id": Path(name).stem, "rgb": rgb, "depth": depth_m,
            "mask": mask, "K": K}


def prepare_target(frame: Mapping[str, Any], config: Mapping[str, Any],
                   device: torch.device) -> dict[str, Any]:
    """Square mask-centered crop, resized to render_size, K adjusted."""

    mask = frame["mask"]
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("First-frame mask is empty")
    height, width = mask.shape
    padding = int(config["roi_padding_px"])
    x0, x1 = int(xs.min()) - padding, int(xs.max()) + 1 + padding
    y0, y1 = int(ys.min()) - padding, int(ys.max()) + 1 + padding
    side = max(x1 - x0, y1 - y0)
    cx_box, cy_box = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = max(0, min(width - side, cx_box - side // 2))
    y0 = max(0, min(height - side, cy_box - side // 2))
    side = min(side, width - x0, height - y0)
    x1, y1 = x0 + side, y0 + side

    size = int(config["render_size"])
    K_crop = crop_intrinsics(frame["K"], x0=x0, y0=y0)
    K_render = resize_intrinsics(K_crop, (side, side), (size, size)).astype(np.float32)

    def crop_resize(image: np.ndarray, mode: str) -> torch.Tensor:
        patch = torch.from_numpy(
            np.ascontiguousarray(image[y0:y1, x0:x1]).astype(np.float32)
        )
        patch = patch[None, None] if patch.ndim == 2 else patch.permute(2, 0, 1)[None]
        resized = F.interpolate(
            patch, size=(size, size), mode=mode,
            **({"align_corners": False} if mode == "bilinear" else {}),
        )
        return resized[0]

    rgb = crop_resize(frame["rgb"] / 255.0, "bilinear")  # [3, S, S]
    mask_t = (crop_resize(mask.astype(np.float32), "nearest")[0] > 0.5).float()
    depth = crop_resize(frame["depth"], "nearest")[0]
    depth_valid = (
        torch.isfinite(depth)
        & (depth > float(config["min_depth_m"]))
        & (depth < float(config["max_depth_m"]))
    )
    return {
        "frame_id": frame["frame_id"],
        "image": (rgb * mask_t[None]).to(device),
        "mask": mask_t.to(device),
        "depth": depth.to(device),
        "depth_valid": depth_valid.to(device),
        "K": torch.from_numpy(K_render).to(device),
        "crop_xyxy": (int(x0), int(y0), int(x1), int(y1)),
    }


def clipped_vector(raw: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(raw)
    factor = torch.clamp(max_norm / norm.clamp_min(1e-12), max=1.0)
    return raw * factor


def axis_angle_to_matrix(rot_vec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rot_vec).clamp_min(1e-12)
    axis = rot_vec / theta
    K = torch.zeros(3, 3, dtype=rot_vec.dtype, device=rot_vec.device)
    K[0, 1], K[0, 2] = -axis[2], axis[1]
    K[1, 0], K[1, 2] = axis[2], -axis[0]
    K[2, 0], K[2, 1] = axis[0], -axis[1]
    eye = torch.eye(3, dtype=rot_vec.dtype, device=rot_vec.device)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


class RtsParameters(torch.nn.Module):
    def __init__(self, config: Mapping[str, Any], device: torch.device) -> None:
        super().__init__()
        self.raw_rot = torch.nn.Parameter(torch.zeros(3, device=device))
        self.raw_t = torch.nn.Parameter(torch.zeros(3, device=device))
        self.raw_log_s = torch.nn.Parameter(torch.zeros((), device=device))
        self.max_rot = math.radians(float(config["max_rot_deg"]))
        self.max_trans = float(config["max_translation_m"])
        self.max_log_scale = math.log(float(config["max_scale_delta"]))

    def current(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rot_vec = clipped_vector(self.raw_rot, self.max_rot)
        trans_delta = clipped_vector(self.raw_t, self.max_trans)
        log_scale_delta = self.max_log_scale * torch.tanh(self.raw_log_s)
        return rot_vec, trans_delta, log_scale_delta, torch.exp(log_scale_delta)


def render_surfels(surfels_canonical, pose: Sim3Pose, rot_vec, trans_delta,
                   scale_delta, target, config, device) -> dict[str, torch.Tensor]:
    """Canonical surfels + Sim(3) delta → 2DGS render in the CV camera.

    Mirrors sibling ``_render_sam3d_rts`` (delta applied between the
    canonical scaling and R_row), with rasterization_2dgs instead of 3DGS.
    """

    from gsplat import rasterization_2dgs

    scale = pose.scale.to(device)
    xyz0 = surfels_canonical.means.to(device) * scale[None, :]
    delta_r = axis_angle_to_matrix(rot_vec)
    xyz_delta = (xyz0 * scale_delta) @ delta_r.T + trans_delta[None, :]
    xyz_p3d = xyz_delta @ pose.R_row.to(device) + pose.T.to(device)[None, :]
    normals_p3d = (surfels_canonical.normals.to(device) @ delta_r.T) @ pose.R_row.to(device)

    xyz_cv = xyz_p3d.clone()
    xyz_cv[:, 0] *= -1.0
    xyz_cv[:, 1] *= -1.0
    normals_cv = normals_p3d.clone()
    normals_cv[:, 0] *= -1.0
    normals_cv[:, 1] *= -1.0

    radii = (
        surfels_canonical.radii.to(device) * float(scale.mean())
    ) * scale_delta
    scales = torch.stack((radii, radii, radii * 0.1), dim=-1)
    size = int(config["render_size"])
    # Degree-0 SH colors: the same rasterization_2dgs path the GS runner
    # trains through (the raw per-camera RGB path crashed in backward).
    sh0 = (surfels_canonical.colors.to(device) - 0.5) / 0.28209479177387814
    renders, alphas, _, _, _, _, _ = rasterization_2dgs(
        means=xyz_cv,
        quats=quats_from_normals(normals_cv),
        scales=scales,
        opacities=surfels_canonical.opacities.to(device),
        colors=sh0[:, None, :],  # [N, 1, 3]
        sh_degree=0,
        viewmats=torch.eye(4, dtype=torch.float32, device=device)[None],
        Ks=target["K"][None],
        width=size,
        height=size,
        render_mode="RGB+ED",
        near_plane=float(config["near_plane"]),
        far_plane=float(config["far_plane"]),
        eps2d=float(config["eps2d"]),
    )
    image = renders[0]
    return {
        "rgb": image[..., :3].permute(2, 0, 1),
        "depth": image[..., 3],
        "alpha": alphas[0, ..., 0],
    }


def refine_loss(rendered, target, rot_vec, trans_delta, log_scale_delta,
                config, ssim_metric, ms_ssim_metric):
    """Faithful port of sibling ``_sam3d_refine_loss`` (same terms/guards)."""

    device = target["image"].device
    trunc_mask = (target["image"].sum(dim=0, keepdim=True) > 0).float()
    target_img = target["image"] * target["mask"]
    photo = torch.tensor(0.0, device=device)
    if ssim_metric is not None:
        photo = photo + ssim_metric((rendered["rgb"] * trunc_mask)[None], target_img[None])
    if ms_ssim_metric is not None:
        photo = photo + ms_ssim_metric((rendered["rgb"] * trunc_mask)[None], target_img[None])

    depth = torch.nan_to_num(rendered["depth"], nan=0.0, posinf=0.0, neginf=0.0)
    alpha = torch.nan_to_num(rendered["alpha"], nan=0.0, posinf=1.0, neginf=0.0)
    target_mask = target["mask"].clamp(0.0, 1.0)
    inside = target_mask > float(config["coverage_mask_threshold"])
    valid = (
        (target_mask > float(config["depth_mask_threshold"]))
        & target["depth_valid"]
        & torch.isfinite(rendered["depth"])
        & (depth > float(config["min_depth_m"]))
        & (alpha.detach() > float(config["depth_alpha_threshold"]))
    )
    if int(valid.sum()) > 0:
        clip = float(config["depth_residual_clip_m"])
        residual = (depth[valid] - target["depth"][valid]).clamp(-clip, clip)
        per_pixel = F.smooth_l1_loss(
            residual, torch.zeros_like(residual),
            beta=float(config["depth_huber_delta_m"]), reduction="none",
        )
        weights = alpha.detach()[valid].clamp(float(config["depth_weight_min"]), 1.0)
        depth_loss = (per_pixel * weights).sum() / weights.sum().clamp_min(1e-6)
    else:
        depth_loss = torch.tensor(0.0, device=device)

    if int(inside.sum()) > 0:
        alpha_target = float(config["coverage_alpha_target"])
        missed = F.relu(alpha_target - alpha[inside])
        coverage_loss = torch.mean(missed * missed)
        visible_ratio = (alpha[inside] / max(alpha_target, 1e-6)).clamp(0.0, 1.0).mean()
        hard_visible = (alpha[inside] > float(config["depth_alpha_threshold"])).sum()
        hard_visible_ratio = hard_visible.float() / inside.sum().float().clamp_min(1.0)
    else:
        coverage_loss = torch.tensor(0.0, device=device)
        visible_ratio = torch.tensor(1.0, device=device)
        hard_visible_ratio = torch.tensor(1.0, device=device)

    visible_gap = F.relu(float(config["min_visible_ratio"]) - visible_ratio)
    visibility_loss = visible_gap * visible_gap
    target_depth_inside = inside & target["depth_valid"]
    if int(target_depth_inside.sum()) > 0:
        depth_valid_ratio = valid.sum().float() / target_depth_inside.sum().float()
    else:
        depth_valid_ratio = torch.tensor(1.0, device=device)

    outside = target_mask < float(config["outside_mask_threshold"])
    outside_loss = (
        torch.mean(alpha[outside] * alpha[outside])
        if int(outside.sum()) > 0
        else torch.tensor(0.0, device=device)
    )
    rot_prior = torch.sum(rot_vec * rot_vec)
    t_prior = torch.sum(trans_delta * trans_delta)
    scale_prior = log_scale_delta * log_scale_delta
    total = (
        float(config["w_depth"]) * depth_loss
        + float(config["w_photo"]) * photo
        + float(config["w_coverage"]) * coverage_loss
        + float(config["w_visibility"]) * visibility_loss
        + float(config["w_outside_alpha"]) * outside_loss
        + float(config["w_reg_rot"]) * rot_prior
        + float(config["w_reg_t"]) * t_prior
        + float(config["w_reg_scale"]) * scale_prior
    )
    parts = {
        "total": float(total.detach().cpu()),
        "depth": float(depth_loss.detach().cpu()),
        "photo": float(photo.detach().cpu()),
        "coverage": float(coverage_loss.detach().cpu()),
        "visibility": float(visibility_loss.detach().cpu()),
        "outside_alpha": float(outside_loss.detach().cpu()),
        "visible_ratio": float(visible_ratio.detach().cpu()),
        "hard_visible_ratio": float(hard_visible_ratio.detach().cpu()),
        "depth_valid_ratio": float(depth_valid_ratio.detach().cpu()),
        "depth_pixels": int(valid.sum().detach().cpu()),
    }
    return total, parts


def make_photo_metrics(config, device):
    """SSIM / MS-SSIM losses (1 - metric); kornia-based with graceful MS skip."""

    ssim_metric = None
    ms_ssim_metric = None
    if bool(config["use_ssim"]):
        from kornia.losses import ssim_loss

        def ssim_metric(pred, tgt):  # noqa: F811 - intentional closure
            return 2.0 * ssim_loss(pred, tgt, window_size=11)

    if bool(config["use_ms_ssim"]):
        try:
            from kornia.losses import MS_SSIMLoss

            ms_module = MS_SSIMLoss().to(device)

            def ms_ssim_metric(pred, tgt):  # noqa: F811
                return ms_module(pred, tgt)

        except ImportError:
            print("[warn] kornia MS_SSIMLoss unavailable; continuing with SSIM only")
    return ssim_metric, ms_ssim_metric


def evaluate(rendered, target, config) -> dict[str, float]:
    alpha = rendered["alpha"]
    mask = target["mask"] > 0.5
    pred = alpha > 0.5
    inter = (pred & mask).sum()
    union = (pred | mask).sum()
    valid = mask & target["depth_valid"] & (alpha > 0.5) & torch.isfinite(rendered["depth"])
    depth_mae_mm = (
        float((rendered["depth"] - target["depth"])[valid].abs().mean() * 1000.0)
        if int(valid.sum()) > 0
        else float("nan")
    )
    outside = ~mask
    return {
        "mask_iou_at_0.5": float(inter / union.clamp_min(1)),
        "depth_mae_mm": depth_mae_mm,
        "depth_pixels": int(valid.sum()),
        "outside_alpha_mean": float(alpha[outside].mean()) if int(outside.sum()) else 0.0,
    }


def save_visuals(tag: str, rendered, target, out_dir: Path) -> None:
    from PIL import Image

    def to_u8(x: torch.Tensor) -> np.ndarray:
        return (x.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    rgb = to_u8(rendered["rgb"].permute(1, 2, 0))
    Image.fromarray(rgb).save(out_dir / f"{tag}_render_rgb.png")
    Image.fromarray(to_u8(rendered["alpha"])).save(out_dir / f"{tag}_render_alpha.png")

    mask = target["mask"] > 0.5
    pred = rendered["alpha"] > 0.5
    overlay = torch.zeros(*mask.shape, 3)
    overlay[mask.cpu() & ~pred.cpu()] = torch.tensor([0.0, 0.8, 0.0])   # miss: green
    overlay[~mask.cpu() & pred.cpu()] = torch.tensor([0.9, 0.0, 0.0])   # false: red
    overlay[mask.cpu() & pred.cpu()] = torch.tensor([0.9, 0.9, 0.0])    # hit: yellow
    Image.fromarray(to_u8(overlay)).save(out_dir / f"{tag}_alpha_mask_overlay.png")

    residual = (rendered["depth"] - target["depth"]).detach().cpu()
    valid = (mask & target["depth_valid"] & (rendered["alpha"] > 0.05)).cpu()
    clip = float(DEFAULT_CONFIG["depth_residual_clip_m"])
    heat = torch.zeros(*mask.shape, 3)
    pos = (residual.clamp(0, clip) / clip)
    neg = ((-residual).clamp(0, clip) / clip)
    heat[..., 0] = torch.where(valid, pos, torch.zeros(()))
    heat[..., 2] = torch.where(valid, neg, torch.zeros(()))
    Image.fromarray(to_u8(heat)).save(out_dir / f"{tag}_depth_residual.png")


def align_prior_sim3(
    surfels,
    init_pose,
    target: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    verbose: bool = True,
):
    """Optimize the rigid Sim(3) correction of an aligned surfel prior.

    Callable core of this script (used online by the ④ backend as well):
    freezes the surfels and refines rotation/translation/log-scale deltas by
    2DGS render-and-compare against ``target`` (see ``prepare_target``).
    Returns ``(refined_pose: Sim3Pose, best, history, status)`` where the
    refined pose composes the best deltas onto ``init_pose`` (sibling
    ``_compose_sam3d_refined_pose``).
    """

    from sam3d_prior import Sim3Pose

    torch.manual_seed(int(config["surfel_seed"]))
    params = RtsParameters(config, device)
    optimizer = torch.optim.AdamW(params.parameters(), lr=float(config["lr"]))
    steps = int(config["steps"])
    warmup = int(config["warmup"])
    lr_max, lr_end = float(config["lr"]), float(config["end_lr"])

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr_max * (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(steps - warmup, 1)
        return lr_end + 0.5 * (lr_max - lr_end) * (
            1.0 + math.cos(math.pi * progress)
        )

    ssim_metric, ms_ssim_metric = make_photo_metrics(config, device)
    best: dict[str, Any] = {"loss": float("inf"), "step": -1}
    history: list[dict[str, Any]] = []
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        rot_vec, trans_delta, log_scale_delta, scale_delta = params.current()
        rendered = render_surfels(surfels, init_pose, rot_vec, trans_delta,
                                  scale_delta, target, config, device)
        loss, parts = refine_loss(rendered, target, rot_vec, trans_delta,
                                  log_scale_delta, config, ssim_metric,
                                  ms_ssim_metric)
        is_candidate = (
            np.isfinite(parts["total"])
            and parts["visible_ratio"]
            >= float(config["best_min_visible_ratio"])
            and parts["depth_valid_ratio"]
            >= float(config["best_min_depth_valid_ratio"])
        )
        if is_candidate and parts["total"] < best["loss"]:
            best = {
                "loss": parts["total"],
                "step": step,
                "rot_vec": rot_vec.detach().clone(),
                "trans_delta": trans_delta.detach().clone(),
                "log_scale_delta": log_scale_delta.detach().clone(),
                "scale_delta": scale_delta.detach().clone(),
            }
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(),
                                       float(config["grad_clip"]))
        optimizer.step()
        log_every = int(config["log_every"])
        if (step == 0 or step == steps - 1
                or (log_every > 0 and (step + 1) % log_every == 0)):
            row = {"step": step, **parts,
                   "rot_deg": float(torch.rad2deg(torch.linalg.norm(rot_vec))),
                   "trans_cm": float(torch.linalg.norm(trans_delta) * 100.0),
                   "scale_delta": float(scale_delta), "lr": lr_at(step)}
            history.append(row)
            if verbose:
                print(f"step {step:04d} total={parts['total']:.5f} "
                      f"depth={parts['depth']:.5f} "
                      f"photo={parts['photo']:.4f} "
                      f"vis={parts['visible_ratio']:.2f} "
                      f"rot={row['rot_deg']:.2f}° t={row['trans_cm']:.2f}cm "
                      f"s={row['scale_delta']:.4f}")

    if best["step"] < 0:
        if verbose:
            print("[warn] no candidate passed the guards; "
                  "keeping the initial pose")
        best.update({
            "rot_vec": torch.zeros(3, device=device),
            "trans_delta": torch.zeros(3, device=device),
            "log_scale_delta": torch.zeros((), device=device),
            "scale_delta": torch.ones((), device=device),
        })
        status = "no_improvement"
    else:
        status = "refined"

    delta_r = axis_angle_to_matrix(best["rot_vec"]).cpu()
    refined_pose = Sim3Pose(
        scale=init_pose.scale * float(best["scale_delta"]),
        R_row=delta_r.T @ init_pose.R_row,
        T=best["trans_delta"].cpu() @ init_pose.R_row + init_pose.T,
    )
    return refined_pose, best, history, status


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda":
        # gsplat 1.5.3 rasterization_2dgs backward hits an illegal memory
        # access whenever the current CUDA device differs from the tensor
        # device (its 2DGS kernels lack a device guard; the 3DGS path has
        # one).  run_gaussian_incremental.py:373 does the same.
        torch.cuda.set_device(device)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    frame = load_first_frame(args.track_dir, depth_dir=args.depth_dir)
    target = prepare_target(frame, config, device)
    prior = load_mesh_prior(args.mesh_npz)
    init_pose = load_sam3d_pose(args.pose_json)
    surfels = sample_surfels(
        prior,
        int(config["surfel_count"]),
        seed=int(config["surfel_seed"]),
        radius_multiplier=float(config["surfel_radius_multiplier"]),
        opacity=float(config["surfel_opacity"]),
    )
    transfer_info: dict[str, float] | None = None
    if args.gaussian_ply is not None:
        gaussian = load_sam3d_gaussian_ply(args.gaussian_ply)
        surfels, transfer_info = transfer_gaussian_colors(surfels, gaussian)
        config["use_ssim"] = True
        config["use_ms_ssim"] = False
        print(f"color transfer: Δmean {transfer_info['color_delta_mean']:.4f} "
              f"fallback {transfer_info['fallback_fraction'] * 100:.1f}% "
              f"| SSIM w={config['w_photo']}")
    elif bool(config["use_ssim"]) or bool(config["use_ms_ssim"]):
        raise SystemExit(
            "Photo terms require --gaussian-ply (transferred colors); the "
            "mesh-color+SSIM combination was ineffective (arm D, 2026-08-31)."
        )
    print(f"frame {target['frame_id']} | surfels {len(surfels)} "
          f"| crop {target['crop_xyxy']} → {config['render_size']}²")

    with torch.no_grad():
        zero = torch.zeros(3, device=device)
        one = torch.ones((), device=device)
        ssim_metric, ms_ssim_metric = make_photo_metrics(config, device)
        rendered0 = render_surfels(surfels, init_pose, zero, zero, one,
                                   target, config, device)
        metrics_before = evaluate(rendered0, target, config)
        _, parts_before = refine_loss(
            rendered0, target, zero, zero, torch.zeros((), device=device),
            config, ssim_metric, ms_ssim_metric,
        )
    save_visuals("before", rendered0, target, out_dir)
    print(f"before: {metrics_before} | parts {parts_before}")

    refined_pose, best, history, status = align_prior_sim3(
        surfels, init_pose, target, config, device
    )

    with torch.no_grad():
        rendered1 = render_surfels(
            surfels, init_pose, best["rot_vec"], best["trans_delta"],
            best["scale_delta"], target, config, device,
        )
        metrics_after = evaluate(rendered1, target, config)
        _, parts_after = refine_loss(
            rendered1, target, best["rot_vec"], best["trans_delta"],
            best["log_scale_delta"], config, ssim_metric, ms_ssim_metric,
        )
    save_visuals("after", rendered1, target, out_dir)
    print(f"after ({status}, best step {best['step']}): {metrics_after}")

    from sam3d_prior import matrix_to_quat_wxyz

    r_row = refined_pose.R_row
    t_row = refined_pose.T
    scale_refined = refined_pose.scale
    steps = int(config["steps"])
    refined = {
        "mapping": "sam3d_canonical_to_pytorch3d_camera",
        "status": status,
        "refined": {
            "sam3d_row_pose": {
                "rotation": matrix_to_quat_wxyz(r_row[None])[0].tolist(),
                "translation": t_row.tolist(),
                "scale": scale_refined.tolist(),
            }
        },
        "delta": {
            "rotation_axis_angle": best["rot_vec"].cpu().tolist(),
            "rotation_deg": float(torch.rad2deg(torch.linalg.norm(best["rot_vec"]))),
            "translation_object_frame_m": best["trans_delta"].cpu().tolist(),
            "log_scale_delta": float(best["log_scale_delta"]),
            "scale_delta": float(best["scale_delta"]),
        },
        "optimization": {
            "best_loss": best["loss"], "best_step": best["step"], "steps": steps,
            "renderer": "gsplat-1.5.3 rasterization_2dgs",
        },
        "metrics": {"before": {**metrics_before, "loss_parts": parts_before},
                    "after": {**metrics_after, "loss_parts": parts_after}},
        "inputs": {
            "track_dir": str(args.track_dir), "frame_id": target["frame_id"],
            "mesh_npz": str(args.mesh_npz), "pose_json": str(args.pose_json),
            "gaussian_ply": (
                str(args.gaussian_ply) if args.gaussian_ply is not None else None
            ),
            "color_transfer": transfer_info,
            "depth_source": (
                str(args.depth_dir) if args.depth_dir is not None
                else "track_dir/depth_filtered"
            ),
        },
        "config": config,
    }
    with (out_dir / "sam3d_rts_refined.json").open("w", encoding="utf-8") as f:
        json.dump(refined, f, indent=2)
    with (out_dir / "loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"saved: {out_dir / 'sam3d_rts_refined.json'}")


if __name__ == "__main__":
    main()
