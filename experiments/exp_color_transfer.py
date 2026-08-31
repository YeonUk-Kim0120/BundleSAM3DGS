"""EXPERIMENT: SAM3D-gaussian color transfer onto mesh surfels + SSIM photo term.

Isolated from the main pipeline on purpose — imports the alignment building
blocks from `run_sam3d_alignment` / `sam3d_prior` and modifies ONLY the surfel
colors and the photo-loss configuration. If the results justify it, an
approved change lands in the main code afterwards.

Arms (driven by CLI):
  A  --color-source mesh     --ssim-weight 0     (control = current best)
  B  --color-source gaussian --ssim-weight 0.5
  C  --color-source gaussian --ssim-weight 1.0
  D  --color-source mesh     --ssim-weight 0.5   (term-vs-color control)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_sam3d_alignment import (  # noqa: E402
    DEFAULT_CONFIG,
    RtsParameters,
    evaluate,
    load_first_frame,
    make_photo_metrics,
    prepare_target,
    refine_loss,
    render_surfels,
    save_visuals,
)
from sam3d_prior import load_mesh_prior, load_sam3d_pose, sample_surfels  # noqa: E402

SH_C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True,
                        help="SAM3D <seq>_splat_depth.ply in the same canonical frame")
    parser.add_argument("--color-source", choices=("mesh", "gaussian"), required=True)
    parser.add_argument("--ssim-weight", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_sam3d_gaussian_ply(path: Path) -> dict[str, np.ndarray]:
    """Parse SAM3D's gaussian PLY (binary_little_endian, float32 properties)."""

    raw = path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii").splitlines()
    count = None
    props: list[str] = []
    for line in header:
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        elif line.startswith("property float"):
            props.append(line.split()[-1])
        elif line.startswith("property") and count is not None:
            raise ValueError(f"Non-float property unsupported: {line}")
    if count is None:
        raise ValueError("PLY has no vertex element")
    for needed in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"):
        if needed not in props:
            raise ValueError(f"PLY missing property {needed}")
    data = np.frombuffer(raw, dtype="<f4", offset=header_end,
                         count=count * len(props)).reshape(count, len(props))
    idx = {name: i for i, name in enumerate(props)}
    positions = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float64)
    f_dc = data[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]]
    colors = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-data[:, idx["opacity"]]))
    return {"positions": positions, "colors": colors, "opacities": opacities}


def transfer_colors(surfels, gaussian: dict[str, np.ndarray], k: int = 8):
    """k-NN inverse-distance color transfer in the shared canonical frame."""

    from scipy.spatial import cKDTree

    tree = cKDTree(gaussian["positions"])
    # Typical gaussian spacing → outlier cutoff at 3× the median NN distance.
    sample = gaussian["positions"][:: max(1, len(gaussian["positions"]) // 20000)]
    spacing, _ = tree.query(sample, k=2, workers=-1)
    cutoff = 3.0 * float(np.median(spacing[:, 1]))

    query = surfels.means.numpy().astype(np.float64)
    dists, indices = tree.query(query, k=k, workers=-1)
    weights = 1.0 / (dists + 1e-6)
    weights[dists > cutoff] = 0.0
    weight_sum = weights.sum(axis=1)
    neighbor_colors = gaussian["colors"][indices]  # [N, k, 3]
    blended = (neighbor_colors * weights[..., None]).sum(axis=1) / np.maximum(
        weight_sum, 1e-9
    )[:, None]
    fallback = weight_sum <= 1e-9
    mesh_colors = surfels.colors.numpy()
    blended[fallback] = mesh_colors[fallback]
    delta = float(np.abs(blended - mesh_colors).mean())
    colors = torch.from_numpy(np.clip(blended, 0.0, 1.0).astype(np.float32))
    return replace(surfels, colors=colors), {
        "color_delta_mean": delta,
        "fallback_fraction": float(fallback.mean()),
        "nn_cutoff_canonical": cutoff,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)  # gsplat 1.5.3 2DGS device-guard bug
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    config = dict(DEFAULT_CONFIG)
    config.update({
        "surfel_radius_multiplier": 0.75,
        "use_ssim": args.ssim_weight > 0,
        "use_ms_ssim": False,
        "w_photo": float(args.ssim_weight),
    })

    frame = load_first_frame(args.track_dir, depth_dir=args.depth_dir)
    target = prepare_target(frame, config, device)
    prior = load_mesh_prior(args.mesh_npz)
    init_pose = load_sam3d_pose(args.pose_json)
    surfels = sample_surfels(
        prior, int(config["surfel_count"]), seed=int(config["surfel_seed"]),
        radius_multiplier=float(config["surfel_radius_multiplier"]),
        opacity=float(config["surfel_opacity"]),
    )
    transfer_info = {"color_delta_mean": 0.0, "fallback_fraction": 0.0}
    if args.color_source == "gaussian":
        gaussian = load_sam3d_gaussian_ply(args.gaussian_ply)
        surfels, transfer_info = transfer_colors(surfels, gaussian)
        print(f"color transfer: Δmean {transfer_info['color_delta_mean']:.4f} "
              f"fallback {transfer_info['fallback_fraction'] * 100:.1f}%")

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
        return lr_end + 0.5 * (lr_max - lr_end) * (1.0 + math.cos(math.pi * progress))

    ssim_metric, ms_ssim_metric = make_photo_metrics(config, device)

    def render_current(rot_vec, trans_delta, scale_delta):
        return render_surfels(surfels, init_pose, rot_vec, trans_delta,
                              scale_delta, target, config, device)

    with torch.no_grad():
        zero = torch.zeros(3, device=device)
        one = torch.ones((), device=device)
        rendered0 = render_current(zero, zero, one)
        metrics_before = evaluate(rendered0, target, config)
    save_visuals("before", rendered0, target, out_dir)

    best: dict = {"loss": float("inf"), "step": -1}
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        rot_vec, trans_delta, log_scale_delta, scale_delta = params.current()
        rendered = render_current(rot_vec, trans_delta, scale_delta)
        loss, parts = refine_loss(rendered, target, rot_vec, trans_delta,
                                  log_scale_delta, config, ssim_metric, ms_ssim_metric)
        is_candidate = (
            np.isfinite(parts["total"])
            and parts["visible_ratio"] >= float(config["best_min_visible_ratio"])
            and parts["depth_valid_ratio"] >= float(config["best_min_depth_valid_ratio"])
        )
        if is_candidate and parts["total"] < best["loss"]:
            best = {"loss": parts["total"], "step": step,
                    "rot_vec": rot_vec.detach().clone(),
                    "trans_delta": trans_delta.detach().clone(),
                    "log_scale_delta": log_scale_delta.detach().clone(),
                    "scale_delta": scale_delta.detach().clone()}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(), float(config["grad_clip"]))
        optimizer.step()

    if best["step"] < 0:
        best.update({"rot_vec": torch.zeros(3, device=device),
                     "trans_delta": torch.zeros(3, device=device),
                     "log_scale_delta": torch.zeros((), device=device),
                     "scale_delta": torch.ones((), device=device)})
        status = "no_improvement"
    else:
        status = "refined"

    with torch.no_grad():
        rendered1 = render_current(best["rot_vec"], best["trans_delta"],
                                   best["scale_delta"])
        metrics_after = evaluate(rendered1, target, config)
    save_visuals("after", rendered1, target, out_dir)

    from run_sam3d_alignment import axis_angle_to_matrix  # noqa: E402
    from sam3d_prior import matrix_to_quat_wxyz  # noqa: E402

    delta_r = axis_angle_to_matrix(best["rot_vec"]).cpu()
    r_row = delta_r.T @ init_pose.R_row
    t_row = best["trans_delta"].cpu() @ init_pose.R_row + init_pose.T
    scale_refined = init_pose.scale * float(best["scale_delta"])
    result = {
        "mapping": "sam3d_canonical_to_pytorch3d_camera",
        "status": status,
        "experiment": {
            "name": "color_transfer",
            "color_source": args.color_source,
            "ssim_weight": float(args.ssim_weight),
            **transfer_info,
        },
        "refined": {"sam3d_row_pose": {
            "rotation": matrix_to_quat_wxyz(r_row[None])[0].tolist(),
            "translation": t_row.tolist(),
            "scale": scale_refined.tolist(),
        }},
        "delta": {
            "rotation_axis_angle": best["rot_vec"].cpu().tolist(),
            "rotation_deg": float(torch.rad2deg(torch.linalg.norm(best["rot_vec"]))),
            "translation_object_frame_m": best["trans_delta"].cpu().tolist(),
            "log_scale_delta": float(best["log_scale_delta"]),
            "scale_delta": float(best["scale_delta"]),
        },
        "metrics": {"before": metrics_before, "after": metrics_after},
        "inputs": {"track_dir": str(args.track_dir),
                   "depth_source": str(args.depth_dir) if args.depth_dir else
                   "track_dir/depth_filtered",
                   "mesh_npz": str(args.mesh_npz),
                   "gaussian_ply": str(args.gaussian_ply)},
        "config": config,
    }
    with (out_dir / "sam3d_rts_refined.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.color_source}/ssim{args.ssim_weight}] {status} "
          f"IoU {metrics_before['mask_iou_at_0.5']:.3f}→{metrics_after['mask_iou_at_0.5']:.3f} "
          f"rot {result['delta']['rotation_deg']:.1f}° scale {result['delta']['scale_delta']:.3f}")


if __name__ == "__main__":
    main()
