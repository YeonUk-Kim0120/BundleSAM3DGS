"""Global refinement and mesh extraction for the Gaussian backend.

Counterpart of BundleSDF's ``run_global_nerf`` for ``bundlesdf.run_gaussian``.  Input is the online map the
backend saved at the end of tracking (``<out_folder>/gs_online/checkpoint_final.pt``: Gaussians, lifecycle state,
keyframe views with their final poses, normalization).  Steps:

1. continue training the online map for ``steps`` iterations (same loss and view sampling as the online updates;
   keyframe poses fixed by default — the poses written back during tracking are not changed, so ADD stays an
   online metric as in the original);
2. extract the primary mesh with screened Poisson reconstruction from the Gaussian centres and surfel normals
   (opacity >= ``opacity_min``, in-plane radius <= ``max_scale_mm``, CONTRADICTED excluded; UNSEEN prior surfels
   included — they are what completes never-observed surfaces);
3. extract the control mesh by TSDF fusion of the 2DGS median depth rendered at the training views (the 6DOPE-GS
   recipe: voxel 2 mm, truncation 2 cm) — seen surfaces only, like-for-like with BundleSDF / 6DOPE-GS;
4. keep the largest connected component of each mesh and write it in the tracker's metric object frame.

Outputs in ``out_dir`` (default ``<out_folder>/final/gs``): ``mesh_real_world.obj`` (= the Poisson mesh; the name
the original benchmark looks for), ``mesh_poisson.obj``, ``mesh_tsdf_train.obj``, optional ``mesh_tsdf_virtual.obj``,
``surfels_metric.ply``, ``checkpoint_global.pt``, ``poses_before_global.txt`` / ``poses_after_global.txt``,
``view_ids.txt``, ``global_manifest.json``.

Setting confirmed on AP12 / mustard0 (GLOBAL_REFINE_RESULTS.md, 2026-09-11): online map continued, pose refinement
off, 2000 steps, Poisson with opacity >= 0.1 and radius <= 10 mm.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import trimesh

from gaussian_runner import GaussianRunner
from prior_lifecycle import STATE_CONTRADICTED, STATE_SUSPECT
from sam3d_prior import quat_wxyz_to_matrix

DEFAULT_GLOBAL_CONFIG: dict[str, Any] = {
    "steps": 2000,
    "pose_refine": False,
    "opacity_min": 0.1,
    "max_scale_mm": 10.0,
    "include_suspect": True,
    "poisson_depth": 9,
    "poisson_density_quantile": 0.05,
    "tsdf_voxel": 0.002,
    "tsdf_trunc": 0.02,
    "depth_mode": "median",
    "alpha_threshold": None,          # None = runner's depth_alpha_threshold
    "tsdf_virtual": False,
    "virtual_views": 120,
    "virtual_radius_factor": 2.5,
    "virtual_size": (640, 480),
    "seed": 0,
}


# ----------------------------------------------------------------------------- checkpoint
def load_online_checkpoint(path: str | Path, device: str) -> GaussianRunner:
    """``GaussianRunner.load_checkpoint`` with the ruamel YAML scalar types allow-listed (older checkpoints carry
    them inside the stored config; torch>=2.6 loads weights-only by default)."""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    from ruamel.yaml.scalarbool import ScalarBoolean
    from ruamel.yaml.scalarfloat import ScalarFloat
    from ruamel.yaml.scalarint import ScalarInt

    with torch.serialization.safe_globals([ScalarInt, ScalarFloat, ScalarBoolean, CommentedMap, CommentedSeq]):
        return GaussianRunner.load_checkpoint(path, device=device)


# ----------------------------------------------------------------------------- rendering
def render_depth(runner: GaussianRunner, K: np.ndarray, c2w_metric: np.ndarray, width: int, height: int,
                 depth_mode: str = "median") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB (H,W,3 float), alpha (H,W), metric camera-z depth (H,W) at an arbitrary camera (2DGS renderer).

    ``median``: depth where the accumulated transmittance crosses 0.5 (what the 2DGS paper fuses with TSDF);
    ``expected``: alpha-weighted mean depth (what the online depth loss uses).  Mirrors
    ``GaussianRunner._rasterize`` — that method discards the median-depth output, so the rasterizer is called here.
    """
    if runner.config["renderer"] != "2dgs":
        raise RuntimeError("render_depth requires the 2dgs renderer")
    if runner.splats is None:
        raise RuntimeError("Runner is not initialized")
    device = runner.device
    c2w_norm = torch.from_numpy(runner.normalization.normalize_c2w(c2w_metric)).float().to(device)[None]
    sh_degree = min(runner.total_steps // int(runner.config["sh_degree_interval"]), int(runner.config["sh_degree"]))
    _, _, rasterization_2dgs, _ = runner._require_gsplat()
    with torch.no_grad():
        colors, alpha, _, _, _, median, _ = rasterization_2dgs(
            means=runner.splats["means"], quats=runner.splats["quats"], scales=torch.exp(runner.splats["scales"]),
            opacities=torch.sigmoid(runner.splats["opacities"]),
            colors=torch.cat((runner.splats["sh0"], runner.splats["shN"]), dim=1),
            viewmats=torch.linalg.inv(c2w_norm), Ks=torch.from_numpy(K.astype(np.float32)).to(device)[None],
            width=int(width), height=int(height), sh_degree=sh_degree, packed=bool(runner.config["packed"]),
            sparse_grad=False, render_mode="RGB+ED", absgrad=False, distloss=False, depth_mode="expected")
        rgb = colors[0, ..., :3].clamp(0, 1).cpu().numpy()
        alpha_np = alpha[0, ..., 0].cpu().numpy()
        depth_t = median[0, ..., 0] if depth_mode == "median" else colors[0, ..., 3]
        depth = (depth_t / runner.normalization.scale).cpu().numpy()
    return rgb, alpha_np, depth


class HideBigGaussians:
    """Temporarily make over-sized Gaussians invisible (opacity logit -30) for the mesh-extraction renders.

    Online training let some prior-lineage surfels grow to 1-29 cm radius (AP12: 2391 of 170 k); rendered from
    arbitrary directions they produce spurious planes.  Restored on exit.
    """

    def __init__(self, runner: GaussianRunner, max_scale_mm: float) -> None:
        self.runner, self.saved = runner, None
        with torch.no_grad():
            radius_mm = torch.exp(runner.splats["scales"][:, :2]).max(1).values / runner.normalization.scale * 1000.0
            self.rows = radius_mm > float(max_scale_mm)
        self.count = int(self.rows.sum())

    def __enter__(self) -> "HideBigGaussians":
        opac = self.runner.splats["opacities"]
        self.saved = opac.data.clone()
        with torch.no_grad():
            opac.data[self.rows] = -30.0
        return self

    def __exit__(self, *exc: Any) -> bool:
        with torch.no_grad():
            self.runner.splats["opacities"].data.copy_(self.saved)
        return False


def look_at_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV camera-to-world (x right, y down, z forward) looking from ``eye`` at ``target``."""
    z = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    z /= np.linalg.norm(z)
    up = np.array([0.0, -1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], -1)


# ----------------------------------------------------------------------------- meshes
def tsdf_fuse(frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], voxel: float,
              trunc: float) -> trimesh.Trimesh:
    """Open3D scalable TSDF fusion.  ``frames``: (rgb uint8 HxWx3, depth float32 metres with 0 = invalid,
    K 3x3, metric OpenCV c2w 4x4)."""
    import open3d as o3d

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel), sdf_trunc=float(trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    n = 0
    for rgb, depth, K, c2w in frames:
        h, w = depth.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32))),
            depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intrinsic, np.linalg.inv(np.asarray(c2w, dtype=np.float64)))
        n += 1
    m = vol.extract_triangle_mesh()
    mesh = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False)
    mesh.metadata["n_frames"] = n
    return mesh


def gaussian_surfels(runner: GaussianRunner, opacity_min: float, max_scale_mm: float,
                     include_suspect: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Metric centres, outward-oriented surfel normals (local +z axis) and colours of the Gaussians kept for Poisson."""
    with torch.no_grad():
        means = runner.normalization.metric_points(runner.splats["means"].detach().cpu().numpy()).astype(np.float64)
        opac = torch.sigmoid(runner.splats["opacities"].detach()).cpu().numpy()
        R = quat_wxyz_to_matrix(runner.splats["quats"].detach().cpu()).numpy()
        normals = R[:, :, 2].astype(np.float64)
        colors = np.clip(runner.splats["sh0"].detach().cpu().numpy()[:, 0, :] * 0.28209479177387814 + 0.5, 0, 1)
        radius_mm = (torch.exp(runner.splats["scales"][:, :2]).max(1).values / runner.normalization.scale * 1000.0).cpu().numpy()
    keep = (opac > float(opacity_min)) & (radius_mm <= float(max_scale_mm))
    if runner.lifecycle_fields is not None:
        state = runner.lifecycle_fields.state.cpu().numpy()
        keep &= state != STATE_CONTRADICTED
        if not include_suspect:
            keep &= state != STATE_SUSPECT
    means, normals, colors = means[keep], normals[keep], colors[keep]
    centre = -runner.normalization.translation.astype(np.float64)
    flip = ((means - centre) * normals).sum(1) < 0
    normals[flip] *= -1.0
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    return means, normals, colors.astype(np.float64)


def poisson_mesh(points: np.ndarray, normals: np.ndarray, colors: np.ndarray | None, depth: int,
                 density_quantile: float) -> trimesh.Trimesh:
    """Screened Poisson surface reconstruction (Open3D); vertices below the density quantile are removed."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
    dens = np.asarray(dens)
    if len(dens) and density_quantile > 0:
        m.remove_vertices_by_mask(dens < np.quantile(dens, float(density_quantile)))
    return trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False)


def largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        return mesh
    return max(parts, key=lambda p: len(p.vertices))


# ----------------------------------------------------------------------------- driver
def extract_meshes(runner: GaussianRunner, out_dir: Path, cfg: dict[str, Any], K_seq: np.ndarray | None = None) -> dict[str, Any]:
    """Poisson (+ TSDF from training views, optional TSDF from virtual views) from the runner's current map."""
    alpha_thr = float(cfg["alpha_threshold"] if cfg["alpha_threshold"] is not None else runner.config["depth_alpha_threshold"])
    manifest: dict[str, Any] = {}

    t1 = time.time()
    means, normals, colors = gaussian_surfels(runner, cfg["opacity_min"], cfg["max_scale_mm"], cfg["include_suspect"])
    trimesh.PointCloud(means, colors=(colors * 255).astype(np.uint8)).export(out_dir / "surfels_metric.ply")
    mesh = poisson_mesh(means, normals, colors, cfg["poisson_depth"], cfg["poisson_density_quantile"])
    manifest["poisson"] = {"points": int(len(means)), "vertices_raw": len(mesh.vertices)}
    mesh = largest_component(mesh)
    mesh.export(out_dir / "mesh_poisson.obj")
    mesh.export(out_dir / "mesh_real_world.obj")
    manifest["poisson"].update({"vertices": len(mesh.vertices), "seconds": time.time() - t1})
    logging.info(f"[GS global] poisson: {manifest['poisson']}")

    def training_frames():
        for view in runner.views:
            c2w = runner.normalization.metric_c2w(view.c2w_normalized.numpy()).astype(np.float64)
            rgb, alpha, depth = render_depth(runner, view.K.numpy(), c2w, view.width, view.height, cfg["depth_mode"])
            valid = (alpha > alpha_thr) & view.mask.numpy()
            yield (np.clip(rgb * 255, 0, 255).astype(np.uint8), np.where(valid, depth, 0.0).astype(np.float32),
                   view.K.numpy(), c2w)

    t1 = time.time()
    with HideBigGaussians(runner, cfg["max_scale_mm"]) as hidden:
        mesh = tsdf_fuse(training_frames(), cfg["tsdf_voxel"], cfg["tsdf_trunc"])
    manifest["hidden_big_gaussians"] = hidden.count
    manifest["tsdf_train"] = {"frames": mesh.metadata["n_frames"], "vertices_raw": len(mesh.vertices)}
    mesh = largest_component(mesh)
    mesh.export(out_dir / "mesh_tsdf_train.obj")
    manifest["tsdf_train"].update({"vertices": len(mesh.vertices), "seconds": time.time() - t1})
    logging.info(f"[GS global] tsdf_train: {manifest['tsdf_train']}")

    if cfg["tsdf_virtual"]:
        if K_seq is None:
            raise ValueError("tsdf_virtual needs the sequence intrinsics K_seq")
        centre = -runner.normalization.translation.astype(np.float64)
        radius_obj = 1.0 / (runner.normalization.scale * 1.2)      # normalization: scale = 1 / (radius * 1.2)
        dist = float(cfg["virtual_radius_factor"]) * radius_obj
        W, H = cfg["virtual_size"]

        def virtual_frames():
            for d in fibonacci_sphere(int(cfg["virtual_views"])):
                c2w = look_at_c2w(centre + dist * d, centre)
                rgb, alpha, depth = render_depth(runner, K_seq, c2w, W, H, cfg["depth_mode"])
                yield (np.clip(rgb * 255, 0, 255).astype(np.uint8), np.where(alpha > alpha_thr, depth, 0.0).astype(np.float32),
                       K_seq, c2w)

        t1 = time.time()
        with HideBigGaussians(runner, cfg["max_scale_mm"]):
            mesh = tsdf_fuse(virtual_frames(), cfg["tsdf_voxel"], cfg["tsdf_trunc"])
        manifest["tsdf_virtual"] = {"frames": mesh.metadata["n_frames"], "camera_distance_m": dist,
                                    "object_radius_m": radius_obj, "vertices_raw": len(mesh.vertices)}
        mesh = largest_component(mesh)
        mesh.export(out_dir / "mesh_tsdf_virtual.obj")
        manifest["tsdf_virtual"].update({"vertices": len(mesh.vertices), "seconds": time.time() - t1})
        logging.info(f"[GS global] tsdf_virtual: {manifest['tsdf_virtual']}")
    return manifest


def resolve_out_dir(out_folder: str | Path) -> Path:
    """``<out_folder>/final/gs``; if it already exists (previous global run) a time-stamped sibling is used so that
    earlier results are never overwritten."""
    base = Path(out_folder) / "final" / "gs"
    if not base.exists():
        return base
    stamped = base.parent / f"gs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.warning(f"[GS global] {base} exists; writing to {stamped}")
    return stamped


def run_global_refine(out_folder: str | Path, out_dir: str | Path | None = None, device: str = "cuda:0",
                      config: dict[str, Any] | None = None, checkpoint: str | Path | None = None) -> dict[str, Any]:
    """Global stage for a finished online run directory (see module docstring).  Returns the manifest."""
    cfg = copy.deepcopy(DEFAULT_GLOBAL_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"unknown global config keys: {sorted(unknown)}")
        cfg.update(config)
    # The BundleSDF entry points set torch's default tensor type to CUDA; GaussianRunner (like the online
    # run_gaussian process) expects the CPU default and moves tensors explicitly.  Restore afterwards.
    default_dtype = torch.get_default_dtype()
    was_cuda_default = torch.tensor([0.0]).is_cuda
    if was_cuda_default:
        torch.set_default_tensor_type("torch.FloatTensor")
    try:
        return _run_global_refine(out_folder, out_dir, device, cfg, checkpoint)
    finally:
        if was_cuda_default:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            torch.set_default_dtype(default_dtype)


def _run_global_refine(out_folder: str | Path, out_dir: str | Path | None, device: str, cfg: dict[str, Any],
                       checkpoint: str | Path | None) -> dict[str, Any]:
    out_folder = Path(out_folder)
    ckpt = Path(checkpoint) if checkpoint else out_folder / "gs_online" / "checkpoint_final.pt"
    out_dir = Path(out_dir) if out_dir else resolve_out_dir(out_folder)
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(cfg["seed"]))
    t0 = time.time()

    runner = load_online_checkpoint(ckpt, device=device)
    if "pose_feedback" not in runner.config:
        raise RuntimeError("checkpoint config has no pose_feedback block (pre-v1 runner?)")
    runner.config["pose_feedback"]["enabled"] = bool(cfg["pose_refine"])
    logging.info(f"[GS global] loaded {ckpt}: {runner.num_gaussians} gaussians, {len(runner.views)} views, "
                 f"online steps {runner.total_steps}")
    manifest: dict[str, Any] = {"out_folder": str(out_folder), "checkpoint": str(ckpt), "config": copy.deepcopy(cfg),
                                "views": len(runner.views), "gaussians_before": runner.num_gaussians}
    np.savetxt(out_dir / "poses_before_global.txt", runner.view_poses_metric().reshape(-1, 4))
    with open(out_dir / "view_ids.txt", "w") as fh:
        fh.write("\n".join(v.frame_id for v in runner.views) + "\n")

    if int(cfg["steps"]) > 0:
        first_loss, final_loss = runner.train(int(cfg["steps"]), optimize_poses=bool(cfg["pose_refine"]))
        manifest.update({"first_loss": first_loss, "final_loss": final_loss})
        if runner.feedback_log:
            manifest["pose_record"] = {k: v for k, v in runner.feedback_log[-1].items() if not k.startswith("per_view")}
        logging.info(f"[GS global] trained {cfg['steps']} steps: loss {first_loss:.4f} -> {final_loss:.4f} ({time.time() - t0:.0f}s)")
    np.savetxt(out_dir / "poses_after_global.txt", runner.view_poses_metric().reshape(-1, 4))
    manifest["gaussians_after"] = runner.num_gaussians
    runner.save_checkpoint(out_dir / "checkpoint_global.pt")

    K_seq = None
    if cfg["tsdf_virtual"] and (out_folder / "cam_K.txt").exists():
        K_seq = np.loadtxt(out_folder / "cam_K.txt").reshape(3, 3)
    manifest.update(extract_meshes(runner, out_dir, cfg, K_seq))
    manifest["seconds_total"] = time.time() - t0
    with open(out_dir / "global_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    logging.info(f"[GS global] done in {manifest['seconds_total']:.0f}s -> {out_dir}")
    return manifest
