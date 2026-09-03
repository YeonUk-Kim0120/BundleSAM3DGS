"""EXPERIMENT ⑤-probe: frozen-map pose-only recovery capability (offline).

Milestone-5 attempt 1 (branch ``milestone5-feedback-attempt1``) failed to
recover translation during JOINT training. Two suspected causes: the map
absorbing pose error, and rotation/translation coupling (1.5 deg at 0.75 m
object distance ~ 19.5 mm of image shift, 8x the 2.5 mm translation).

This probe removes the map from the equation entirely: the map is FROZEN
after prior initialization, and only a per-view SE(3) delta is optimized by
render-and-compare. Injections are swept per axis and per size so the two
causes separate:

  pure-trans 2.5/5/10 mm  -> at what size does translation signal beat noise?
  pure-rot   1.5/3 deg    -> how much rotation leaks into translation?
  mixed      (realistic)  -> does freezing the map alone fix the old setup?

Goal reframing (2026-09-02, user-approved): the feedback target is OUTLIER
correction (pull >=5 mm excursions toward the ~2-3 mm floor), not precision
recovery of 2.5 mm errors.

  python3 experiments/exp_pose_only_probe.py --device cuda:1 \
    --output-dir logs/mustard0_poseonly_probe_<date>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

TRACK = Path("/home/kist/Desktop/BundleSAM3DGS/logs/"
             "mustard0_bundlesam3dgs_exact_baseline_rematch_backprojection_20260811")
MESH = Path("/home/kist/Desktop/sam-3d-objects/output_YCBInEOAT_sam2mask_mesh")
POSE_JSON = Path("/home/kist/Desktop/BundleSAM3DGS/logs/"
                 "mustard0_sam3d_align_sam2mask_rawdepth_20260831/"
                 "sam3d_rts_refined.json")

# (name, trans_mm, rot_deg)
INJECTIONS = [
    ("trans2.5", 2.5, 0.0),
    ("trans5", 5.0, 0.0),
    ("trans10", 10.0, 0.0),
    ("rot1.5", 0.0, 1.5),
    ("rot3", 0.0, 3.0),
    ("mix2.5_1.5", 2.5, 1.5),
    ("mix5_3", 5.0, 3.0),
    ("mix10_5", 10.0, 5.0),
]


def se3_exp(delta: torch.Tensor) -> torch.Tensor:
    """Differentiable SE(3)-style exponential of (rotvec[3], trans[3]) -> 4x4.

    Translation applies directly (no V-matrix), exact at delta=0. The
    clamped-theta form keeps gradients alive at the zero initialization.
    NOTE: the attempt-1 branch version of this function had swapped
    K[2,0]/K[2,1] skew entries (shear for non-z axes); this is the
    corrected antisymmetric generator.
    """
    rotvec, translation = delta[:3], delta[3:]
    theta = torch.linalg.norm(rotvec).clamp_min(1e-12)
    axis = rotvec / theta
    x, y, z = axis[0], axis[1], axis[2]
    zero = torch.zeros((), dtype=delta.dtype, device=delta.device)
    K = torch.stack((
        torch.stack((zero, -z, y)),
        torch.stack((z, zero, -x)),
        torch.stack((-y, x, zero)),
    ))
    eye = torch.eye(3, dtype=delta.dtype, device=delta.device)
    R = eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
    top = torch.cat((R, translation[:, None]), dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=delta.dtype,
                          device=delta.device)
    return torch.cat((top, bottom), dim=0)


def perturbation(rng: np.random.Generator, trans_mm: float,
                 rot_deg: float) -> np.ndarray:
    """Camera-frame perturbation T (right-multiplied onto c2w)."""
    T = np.eye(4, dtype=np.float64)
    if rot_deg > 0:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.radians(rot_deg)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        T[:3, :3] = (np.eye(3) + np.sin(angle) * K
                     + (1 - np.cos(angle)) * (K @ K))
    if trans_mm > 0:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        T[:3, 3] = direction * trans_mm / 1000.0
    return T


def pose_errors_normalized(a: torch.Tensor, b: torch.Tensor,
                           scale: float) -> tuple[float, float]:
    """(mm, deg) between two normalized c2w matrices."""
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    trans_mm = float(np.linalg.norm(a_np[:3, 3] - b_np[:3, 3]) / scale * 1000)
    cos = np.clip((np.trace(a_np[:3, :3].T @ b_np[:3, :3]) - 1) / 2, -1, 1)
    return trans_mm, float(np.degrees(np.arccos(cos)))


def run_probe(runner: GaussianRunner, view, bad_n: torch.Tensor,
              steps: int, lr_rot: float, lr_trans: float) -> dict:
    device = runner.device
    rot = torch.zeros(3, device=device, requires_grad=True)
    trans = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([
        {"params": [rot], "lr": lr_rot},
        {"params": [trans], "lr": lr_trans},
    ])
    target = view.rgb.to(device)[None]
    mask = view.mask.to(device)[None]
    K = view.K.to(device)[None]
    bad = bad_n.to(device)
    sh_degree = min(
        runner.total_steps // int(runner.config["sh_degree_interval"]),
        int(runner.config["sh_degree"]),
    )
    depth_weight = float(runner.config["depth_loss_weight"])
    ssim_weight = float(runner.config["ssim_weight"])
    final_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        c2w = (bad @ se3_exp(torch.cat((rot, trans))))[None]
        result = runner._rasterize(
            K=K,
            c2w=c2w,
            width=view.width,
            height=view.height,
            sh_degree=sh_degree,
            render_mode="RGB+ED" if depth_weight > 0 else "RGB",
            absgrad=False,
        )
        rendered = result.colors[..., :3]
        mask_channels = mask[..., None].to(rendered.dtype)
        denominator = mask_channels.sum().clamp_min(1.0) * 3.0
        l1 = (torch.abs(rendered - target) * mask_channels).sum() / denominator
        dssim = runner._masked_dssim(rendered, target, mask)
        loss = (1.0 - ssim_weight) * l1 + ssim_weight * dssim
        if depth_weight > 0:
            loss = loss + depth_weight * runner._masked_depth_loss(
                result, view, mask
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite probe loss")
        loss.backward()
        optimizer.step()
        for parameter in runner.splats.values():
            parameter.grad = None
        final_loss = float(loss.detach().cpu())
    with torch.no_grad():
        corrected = bad @ se3_exp(torch.cat((rot, trans)))
    scale = float(runner.normalization.scale)
    return {
        "corrected_n": corrected,
        "rot": rot.detach().cpu(),
        "trans": trans.detach().cpu(),
        "delta_rot_deg": float(np.degrees(torch.linalg.norm(rot).item())),
        "delta_trans_mm": float(torch.linalg.norm(trans).item() / scale * 1000),
        "final_loss": final_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--initial-steps", type=int, default=4000)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--lr-rot", type=float, default=2e-3)
    parser.add_argument("--lr-trans", type=float, default=5e-4)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    config = load_gaussian_config(
        "/home/kist/Desktop/BundleSAM3DGS/config_gs_2dgs_1mm_lifecycle.yml"
    )
    config["device"] = args.device

    frame_ids, poses = load_fixed_poses(find_latest_keyframe_snapshot(TRACK))
    K = np.loadtxt(TRACK / "cam_K.txt", dtype=np.float32).reshape(3, 3)
    n_init = 5
    frame_ids, poses = frame_ids[: n_init + 1], poses[: n_init + 1]

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
    normalization = SceneNormalization(
        scale=1.0 / max(radius * 1.2, 1e-6), translation=-center
    )
    runner = GaussianRunner(config, normalization, device=args.device)

    initial_frames = [
        load_frame(TRACK, frame_ids[i], K, poses[i]) for i in range(n_init)
    ]
    runner.initialize_from_prior(
        surfels_cv, poses[0], initial_frames, train_steps=args.initial_steps
    )
    # The map stays frozen by never stepping its optimizers. Splats keep
    # requires_grad=True on purpose: gsplat's backward with camera-only
    # gradients is an unexercised path (suspected cause of the first
    # full-run segfault); their grads are computed and discarded per step.

    scale = float(runner.normalization.scale)
    # in-map: last initialization view; held-out: the next keyframe (never
    # trained on), i.e. the online condition of a newly arriving frame.
    held_out_frame = load_frame(TRACK, frame_ids[n_init], K, poses[n_init])
    targets = {
        "in_map": (runner.views[n_init - 1], poses[n_init - 1]),
        "held_out": (runner._prepare_view(held_out_frame), poses[n_init]),
    }

    rows = []
    for target_name, (view, true_pose) in targets.items():
        true_n = torch.from_numpy(
            runner.normalization.normalize_c2w(true_pose)
        ).to(torch.float32)
        for config_index, (config_name, trans_mm, rot_deg) in enumerate(
            INJECTIONS
        ):
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * seed + config_index)
                T_metric = perturbation(rng, trans_mm, rot_deg)
                T_norm = T_metric.copy()
                T_norm[:3, 3] *= scale
                bad_n = true_n.to(torch.float64) @ torch.from_numpy(T_norm)
                bad_n = bad_n.to(torch.float32)
                injected_t, injected_r = pose_errors_normalized(
                    bad_n, true_n, scale
                )
                probe = run_probe(
                    runner, view, bad_n,
                    steps=args.probe_steps,
                    lr_rot=args.lr_rot, lr_trans=args.lr_trans,
                )
                residual_t, residual_r = pose_errors_normalized(
                    probe["corrected_n"], true_n, scale
                )
                row = {
                    "target": target_name,
                    "config": config_name,
                    "seed": seed,
                    "injected_mm": injected_t,
                    "injected_deg": injected_r,
                    "residual_mm": residual_t,
                    "residual_deg": residual_r,
                    "recovery_mm_pct": (
                        100 * (1 - residual_t / injected_t)
                        if injected_t > 0.1 else None
                    ),
                    "recovery_deg_pct": (
                        100 * (1 - residual_r / injected_r)
                        if injected_r > 0.1 else None
                    ),
                    "delta_rot_deg": probe["delta_rot_deg"],
                    "delta_trans_mm": probe["delta_trans_mm"],
                    "final_loss": probe["final_loss"],
                }
                rows.append(row)
                print(json.dumps({
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in row.items()
                }), flush=True)

    aggregates = []
    for target_name in targets:
        for config_name, _, _ in INJECTIONS:
            group = [r for r in rows
                     if r["target"] == target_name
                     and r["config"] == config_name]
            def _mean(key):
                values = [g[key] for g in group if g[key] is not None]
                return float(np.mean(values)) if values else None
            aggregates.append({
                "target": target_name,
                "config": config_name,
                "recovery_mm_pct": _mean("recovery_mm_pct"),
                "recovery_deg_pct": _mean("recovery_deg_pct"),
                "residual_mm": _mean("residual_mm"),
                "residual_deg": _mean("residual_deg"),
                "leak_rot_deg": _mean("delta_rot_deg"),
                "leak_trans_mm": _mean("delta_trans_mm"),
            })
    with (out_dir / "result.json").open("w") as f:
        json.dump({
            "rows": rows,
            "aggregates": aggregates,
            "settings": {
                "probe_steps": args.probe_steps,
                "lr_rot": args.lr_rot,
                "lr_trans": args.lr_trans,
                "initial_steps": args.initial_steps,
                "seeds": args.seeds,
                "loss": "masked L1 + DSSIM + depth Huber (map frozen)",
            },
        }, f, indent=2)
    for aggregate in aggregates:
        print("AGG " + json.dumps({
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in aggregate.items()
        }), flush=True)


if __name__ == "__main__":
    main()
