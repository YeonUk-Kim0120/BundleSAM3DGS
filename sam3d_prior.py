"""SAM3D mesh prior loading and 2DGS surfel initialization.

Conventions mirror the validated BundleGS implementation in the sibling
checkout (/home/kist/Desktop/BundleSDF/gaussian_runner.py):

- The mesh npz stores SAM3D's *raw* MeshExtractResult in the canonical
  [-0.5, 0.5]^3 frame — the same frame as the Gaussian PLY — so the pose
  JSON (rotation/translation/scale) applies to it directly.  The processed
  GLB must NOT be used here: to_glb() rotates vertices z-up→y-up.
- Pose application (sibling ``_render_sam3d_rts``):
      x_p3d = (x_can * scale) @ R_row + T
  with R_row = quaternion_to_matrix(rotation), quaternion real-first
  (w, x, y, z), row-vector convention.  The result lives in the PyTorch3D
  camera frame of the input image.
- PyTorch3D camera → OpenCV camera: negate x and y (sibling
  ``gaussian_runner.py:1986-1990``).  This is a 180-degree rotation about
  z (det = +1), so triangle winding and normals stay consistent.

Surfels keep their unit normals; disk orientation quaternions are derived
from the normals only when rendering, because a 2DGS disk is isotropic
in-plane and the in-plane angle is arbitrary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import json

import numpy as np
import torch


SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class MeshPrior:
    vertices: torch.Tensor  # [V, 3] float32, canonical frame
    faces: torch.Tensor  # [F, 3] int64
    vertex_colors: torch.Tensor  # [V, 3] float32 in [0, 1]

    def validated(self) -> "MeshPrior":
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be [V, 3]")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be [F, 3]")
        if self.vertex_colors.shape != self.vertices.shape:
            raise ValueError("vertex_colors must match vertices")
        if len(self.vertices) == 0 or len(self.faces) == 0:
            raise ValueError("mesh prior is empty")
        if not torch.isfinite(self.vertices).all():
            raise ValueError("vertices must be finite")
        if int(self.faces.min()) < 0 or int(self.faces.max()) >= len(self.vertices):
            raise ValueError("faces index out of range")
        colors = self.vertex_colors
        if not torch.isfinite(colors).all() or colors.min() < 0 or colors.max() > 1:
            raise ValueError("vertex_colors must be finite in [0, 1]")
        return self


@dataclass(frozen=True)
class Sim3Pose:
    """SAM3D canonical → PyTorch3D-camera Sim(3), row-vector convention."""

    scale: torch.Tensor  # [3] float32 (SAM3D emits an isotropic triple)
    R_row: torch.Tensor  # [3, 3] float32, applied as x @ R_row
    T: torch.Tensor  # [3] float32


@dataclass(frozen=True)
class SurfelSet:
    """Oriented disks; frame is whatever the producing function documents."""

    means: torch.Tensor  # [N, 3]
    normals: torch.Tensor  # [N, 3] unit
    radii: torch.Tensor  # [N] linear disk radius
    colors: torch.Tensor  # [N, 3] in [0, 1]
    opacities: torch.Tensor  # [N] in (0, 1)

    def __len__(self) -> int:
        return int(self.means.shape[0])


def quat_wxyz_to_matrix(quats: torch.Tensor) -> torch.Tensor:
    """Batch quaternion (w, x, y, z) → rotation matrix (column convention)."""

    q = quats / torch.linalg.norm(quats, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        -2,
    )


def matrix_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Batch rotation matrix → quaternion (w, x, y, z), Shepperd's method."""

    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    q = torch.empty(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)

    c0 = trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-12)) * 2
    q0 = torch.stack((0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s), -1)
    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    s = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12)) * 2
    q1 = torch.stack(((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s), -1)
    c2 = (~c0) & (~c1) & (m11 > m22)
    s = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-12)) * 2
    q2 = torch.stack(((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s), -1)
    s = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-12)) * 2
    q3 = torch.stack(((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s), -1)

    q = torch.where(c0[..., None], q0, torch.where(c1[..., None], q1,
                    torch.where(c2[..., None], q2, q3)))
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def quats_from_normals(normals: torch.Tensor) -> torch.Tensor:
    """wxyz quaternions whose local +z axis maps onto each unit normal.

    The in-plane angle is chosen deterministically; 2DGS disks with equal
    in-plane radii are invariant to it.
    """

    n = normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp_min(1e-12)
    ref = torch.zeros_like(n)
    use_z = n[:, 2].abs() < 0.9
    ref[use_z, 2] = 1.0
    ref[~use_z, 0] = 1.0
    t1 = torch.linalg.cross(ref, n)
    t1 = t1 / torch.linalg.norm(t1, dim=-1, keepdim=True).clamp_min(1e-12)
    t2 = torch.linalg.cross(n, t1)
    R = torch.stack((t1, t2, n), dim=-1)  # columns: local x, y, z axes
    return matrix_to_quat_wxyz(R)


def load_mesh_prior(path: str | Path) -> MeshPrior:
    with np.load(path) as data:
        if "success" in data and not bool(np.asarray(data["success"])):
            raise ValueError(f"SAM3D mesh prior marked success=False: {path}")
        prior = MeshPrior(
            vertices=torch.from_numpy(np.asarray(data["vertices"], dtype=np.float32)),
            faces=torch.from_numpy(np.asarray(data["faces"], dtype=np.int64)),
            vertex_colors=torch.from_numpy(
                np.asarray(data["vertex_colors"], dtype=np.float32)
            ),
        )
    return prior.validated()


def load_sam3d_pose(path: str | Path) -> Sim3Pose:
    with Path(path).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    pose: dict[str, Any] = meta.get("pose", meta)
    quat = torch.tensor(pose["rotation"], dtype=torch.float32).reshape(-1, 4)[0]
    translation = torch.tensor(pose["translation"], dtype=torch.float32).reshape(-1, 3)[0]
    scale = torch.tensor(pose["scale"], dtype=torch.float32).reshape(-1, 3)[0]
    if not torch.isfinite(quat).all() or torch.linalg.norm(quat) < 1e-8:
        raise ValueError(f"Invalid rotation quaternion in {path}")
    if not torch.isfinite(translation).all() or not torch.isfinite(scale).all():
        raise ValueError(f"Non-finite pose values in {path}")
    if scale.min() <= 0:
        raise ValueError(f"Pose scale must be positive in {path}")
    R_row = quat_wxyz_to_matrix(quat[None])[0]
    return Sim3Pose(scale=scale, R_row=R_row, T=translation)


def face_areas_and_normals(
    vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = torch.linalg.cross(v1 - v0, v2 - v0)
    double_area = torch.linalg.norm(cross, dim=-1)
    areas = 0.5 * double_area
    normals = cross / double_area[:, None].clamp_min(1e-12)
    return areas, normals


def sample_surfels(
    prior: MeshPrior,
    count: int,
    *,
    seed: int = 0,
    radius_multiplier: float = 1.5,
    opacity: float = 0.9,
) -> SurfelSet:
    """Area-weighted surfel sampling on the mesh, in the canonical frame."""

    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 < opacity < 1.0:
        raise ValueError("opacity must be in (0, 1)")
    if radius_multiplier <= 0:
        raise ValueError("radius_multiplier must be positive")
    prior = prior.validated()
    areas, normals = face_areas_and_normals(prior.vertices, prior.faces)
    valid = areas > 0
    if not bool(valid.any()):
        raise ValueError("mesh prior has no non-degenerate faces")
    areas = areas[valid]
    normals = normals[valid]
    faces = prior.faces[valid]

    generator = torch.Generator().manual_seed(seed)
    face_idx = torch.multinomial(
        areas / areas.sum(), count, replacement=True, generator=generator
    )
    # Uniform barycentric sampling via the square-root trick.
    r1 = torch.sqrt(torch.rand(count, generator=generator))
    r2 = torch.rand(count, generator=generator)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    tri = faces[face_idx]
    weights = torch.stack((w0, w1, w2), dim=-1)[:, :, None]
    means = (prior.vertices[tri] * weights).sum(dim=1)
    colors = (prior.vertex_colors[tri] * weights).sum(dim=1).clamp(0.0, 1.0)

    total_area = float(areas.sum())
    radius = radius_multiplier * float(np.sqrt(total_area / count))
    return SurfelSet(
        means=means,
        normals=normals[face_idx],
        radii=torch.full((count,), radius, dtype=torch.float32),
        colors=colors,
        opacities=torch.full((count,), float(opacity), dtype=torch.float32),
    )


def transform_surfels_canonical_to_cv_camera(
    surfels: SurfelSet, pose: Sim3Pose
) -> SurfelSet:
    """Canonical → PyTorch3D camera (Sim(3)) → OpenCV camera (x, y negated)."""

    scale = pose.scale.to(surfels.means.dtype)
    means_p3d = (surfels.means * scale[None, :]) @ pose.R_row + pose.T[None, :]
    normals_p3d = surfels.normals @ pose.R_row
    means_cv = means_p3d.clone()
    means_cv[:, 0] *= -1.0
    means_cv[:, 1] *= -1.0
    normals_cv = normals_p3d.clone()
    normals_cv[:, 0] *= -1.0
    normals_cv[:, 1] *= -1.0
    normals_cv = normals_cv / torch.linalg.norm(
        normals_cv, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    return replace(
        surfels,
        means=means_cv,
        normals=normals_cv,
        radii=surfels.radii * float(scale.mean()),
    )


def load_sam3d_gaussian_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Parse SAM3D's gaussian PLY (binary_little_endian, float32 properties).

    Returns canonical-frame positions, activated RGB colors (from the SH DC
    band), and sigmoid-activated opacities.
    """

    raw = Path(path).read_bytes()
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
    index = {name: i for i, name in enumerate(props)}
    positions = data[:, [index["x"], index["y"], index["z"]]].astype(np.float64)
    f_dc = data[:, [index["f_dc_0"], index["f_dc_1"], index["f_dc_2"]]]
    colors = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-data[:, index["opacity"]]))
    return {"positions": positions, "colors": colors,
            "opacities": opacities.astype(np.float32)}


def transfer_gaussian_colors(
    surfels: SurfelSet,
    gaussian: Mapping[str, np.ndarray],
    k: int = 8,
) -> tuple[SurfelSet, dict[str, float]]:
    """k-NN inverse-distance color transfer in the shared canonical frame.

    Adopted as the default appearance source for the Sim(3) alignment
    (2026-08-31, 22-sequence A/C study): SAM3D's gaussian colors are
    appearance-trained, unlike the coarse mesh vertex colors.  Surfels with
    no gaussian neighbor within 3x the median gaussian spacing keep their
    mesh colors.
    """

    from scipy.spatial import cKDTree

    tree = cKDTree(gaussian["positions"])
    sample = gaussian["positions"][:: max(1, len(gaussian["positions"]) // 20000)]
    spacing, _ = tree.query(sample, k=2, workers=-1)
    cutoff = 3.0 * float(np.median(spacing[:, 1]))

    query = surfels.means.numpy().astype(np.float64)
    dists, indices = tree.query(query, k=k, workers=-1)
    weights = 1.0 / (dists + 1e-6)
    weights[dists > cutoff] = 0.0
    weight_sum = weights.sum(axis=1)
    neighbor_colors = gaussian["colors"][indices]
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


def surfels_to_gsplat_inputs(
    surfels: SurfelSet, *, flat_axis_ratio: float = 0.1
) -> dict[str, torch.Tensor]:
    """SurfelSet → gsplat rasterization_2dgs tensor dict.

    The third scale component is ignored by the 2DGS rasterizer; it is kept
    small but positive so PLY exports stay viewable.
    """

    radii = surfels.radii.clamp_min(1e-8)
    scales = torch.stack(
        (radii, radii, radii * float(flat_axis_ratio)), dim=-1
    )
    return {
        "means": surfels.means,
        "quats": quats_from_normals(surfels.normals),
        "scales": scales,
        "opacities": surfels.opacities,
        "colors": surfels.colors,
    }
