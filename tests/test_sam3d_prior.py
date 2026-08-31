"""CPU tests for SAM3D mesh-prior loading, surfel sampling, and pose math."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sam3d_prior import (
    SH_C0,
    MeshPrior,
    Sim3Pose,
    face_areas_and_normals,
    load_mesh_prior,
    load_sam3d_gaussian_ply,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    quats_from_normals,
    sample_surfels,
    surfels_to_gsplat_inputs,
    transfer_gaussian_colors,
    transform_surfels_canonical_to_cv_camera,
)


def two_triangle_prior(area_ratio=4.0):
    """Two right triangles in z=0 planes with area 0.5 and 0.5*area_ratio."""

    s = math.sqrt(area_ratio)
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0], [2.0 + s, 0.0, 0.0], [2.0, s, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
    colors = torch.tensor(
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
        dtype=torch.float32,
    )
    return MeshPrior(vertices, faces, colors)


class QuaternionMathTest(unittest.TestCase):
    def test_matrix_quat_round_trip(self):
        generator = torch.Generator().manual_seed(3)
        axis = torch.nn.functional.normalize(torch.randn(64, 3, generator=generator), dim=-1)
        # Build rotations from random quaternions and round-trip them.
        quats = torch.nn.functional.normalize(torch.randn(64, 4, generator=generator), dim=-1)
        R = quat_wxyz_to_matrix(quats)
        back = quat_wxyz_to_matrix(matrix_to_quat_wxyz(R))
        torch.testing.assert_close(back, R, atol=1e-5, rtol=1e-5)
        del axis

    def test_quats_from_normals_map_z_to_normal(self):
        generator = torch.Generator().manual_seed(1)
        normals = torch.nn.functional.normalize(
            torch.randn(128, 3, generator=generator), dim=-1
        )
        # Include the branch-switch cases.
        normals = torch.cat(
            (normals, torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]))
        )
        R = quat_wxyz_to_matrix(quats_from_normals(normals))
        z_axis = R[..., :, 2]
        torch.testing.assert_close(z_axis, normals, atol=1e-5, rtol=1e-5)
        det = torch.linalg.det(R)
        torch.testing.assert_close(det, torch.ones_like(det), atol=1e-4, rtol=1e-4)


class SurfelSamplingTest(unittest.TestCase):
    def test_area_weighted_face_selection_and_colors(self):
        prior = two_triangle_prior(area_ratio=4.0)
        surfels = sample_surfels(prior, 5000, seed=0)
        blue = surfels.colors[:, 2] > 0.5
        fraction_blue = float(blue.float().mean())
        self.assertAlmostEqual(fraction_blue, 0.8, delta=0.03)  # 4/(1+4)
        self.assertTrue(torch.isfinite(surfels.means).all())
        # Samples must lie inside their triangles (z == 0 planes here).
        self.assertTrue(torch.allclose(surfels.means[:, 2], torch.zeros(5000)))

    def test_radius_scales_with_count(self):
        prior = two_triangle_prior()
        few = sample_surfels(prior, 100, seed=0)
        many = sample_surfels(prior, 400, seed=0)
        self.assertAlmostEqual(
            float(few.radii[0]) / float(many.radii[0]), 2.0, delta=1e-4
        )

    def test_degenerate_faces_are_ignored(self):
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 2.0]],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [3, 3, 3]], dtype=torch.int64)
        colors = torch.zeros(4, 3)
        surfels = sample_surfels(MeshPrior(vertices, faces, colors), 64, seed=0)
        self.assertEqual(len(surfels), 64)
        self.assertTrue(torch.allclose(surfels.means[:, 2], torch.zeros(64)))


class PoseTransformTest(unittest.TestCase):
    def test_known_sim3_and_cv_flip(self):
        prior = two_triangle_prior()
        surfels = sample_surfels(prior, 16, seed=0)
        # 90° about z (row convention), scale 2, translation (0.1, -0.2, 0.5).
        angle = math.pi / 2
        R_row = torch.tensor(
            [
                [math.cos(angle), math.sin(angle), 0.0],
                [-math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        pose = Sim3Pose(
            scale=torch.full((3,), 2.0),
            R_row=R_row,
            T=torch.tensor([0.1, -0.2, 0.5]),
        )
        out = transform_surfels_canonical_to_cv_camera(surfels, pose)
        x = surfels.means[0]
        expected_p3d = (x * 2.0) @ R_row + pose.T
        expected_cv = expected_p3d * torch.tensor([-1.0, -1.0, 1.0])
        torch.testing.assert_close(out.means[0], expected_cv, atol=1e-5, rtol=1e-5)
        # Normals: rotated, flipped, and still unit.
        n_expected = (surfels.normals[0] @ R_row) * torch.tensor([-1.0, -1.0, 1.0])
        torch.testing.assert_close(out.normals[0], n_expected, atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(float(torch.linalg.norm(out.normals[0])), 1.0, places=5)
        # Radii scale by the mean of the (isotropic) pose scale.
        self.assertAlmostEqual(
            float(out.radii[0] / surfels.radii[0]), 2.0, places=5
        )

    def test_gsplat_inputs_shapes(self):
        prior = two_triangle_prior()
        surfels = sample_surfels(prior, 8, seed=0)
        inputs = surfels_to_gsplat_inputs(surfels)
        self.assertEqual(inputs["means"].shape, (8, 3))
        self.assertEqual(inputs["quats"].shape, (8, 4))
        self.assertEqual(inputs["scales"].shape, (8, 3))
        self.assertTrue((inputs["scales"] > 0).all())
        self.assertTrue((inputs["scales"][:, 2] < inputs["scales"][:, 0]).all())


class LoaderTest(unittest.TestCase):
    def test_npz_round_trip_and_success_gate(self):
        prior = two_triangle_prior()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mesh.npz"
            np.savez_compressed(
                path,
                vertices=prior.vertices.numpy(),
                faces=prior.faces.numpy(),
                vertex_colors=prior.vertex_colors.numpy(),
                success=np.array(True),
            )
            loaded = load_mesh_prior(path)
            torch.testing.assert_close(loaded.vertices, prior.vertices)
            np.savez_compressed(
                path,
                vertices=prior.vertices.numpy(),
                faces=prior.faces.numpy(),
                vertex_colors=prior.vertex_colors.numpy(),
                success=np.array(False),
            )
            with self.assertRaisesRegex(ValueError, "success=False"):
                load_mesh_prior(path)

    def test_face_areas_and_normals(self):
        prior = two_triangle_prior(area_ratio=4.0)
        areas, normals = face_areas_and_normals(prior.vertices, prior.faces)
        torch.testing.assert_close(
            areas, torch.tensor([0.5, 2.0]), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            normals[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=1e-6
        )


def write_gaussian_ply(path, positions, colors, opacity_logit=2.0):
    """Minimal SAM3D-style gaussian PLY (17 float32 properties per vertex)."""

    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    count = len(positions)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    )
    data = np.zeros((count, len(props)), dtype="<f4")
    data[:, 0:3] = positions
    data[:, 6:9] = (np.asarray(colors, dtype=np.float32) - 0.5) / SH_C0
    data[:, 9] = opacity_logit
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


class GaussianColorTransferTest(unittest.TestCase):
    def test_ply_round_trip(self):
        positions = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "g.ply"
            write_gaussian_ply(path, positions, colors)
            gaussian = load_sam3d_gaussian_ply(path)
        np.testing.assert_allclose(gaussian["positions"], positions, atol=1e-6)
        np.testing.assert_allclose(gaussian["colors"], colors, atol=1e-6)
        self.assertTrue((gaussian["opacities"] > 0.8).all())

    def test_transfer_picks_nearest_colors_and_falls_back(self):
        prior = two_triangle_prior()
        surfels = sample_surfels(prior, 64, seed=0)
        # Gaussians: dense red cloud on triangle 1's region (x<2), none near
        # triangle 2 (x>2) → those surfels must keep their mesh (blue) color.
        rng = np.random.default_rng(0)
        positions = np.concatenate([
            rng.uniform([0.0, 0.0, -0.001], [1.0, 1.0, 0.001], (500, 3)),
        ])
        colors = np.tile([[1.0, 0.0, 0.0]], (500, 1))
        gaussian = {
            "positions": positions,
            "colors": colors.astype(np.float32),
            "opacities": np.full(500, 0.9, dtype=np.float32),
        }
        transferred, info = transfer_gaussian_colors(surfels, gaussian)
        near = surfels.means[:, 0] < 1.5
        far = surfels.means[:, 0] > 2.0
        self.assertTrue(bool(near.any()) and bool(far.any()))
        self.assertTrue((transferred.colors[near, 0] > 0.9).all())  # red
        torch.testing.assert_close(  # fallback keeps mesh blue
            transferred.colors[far], surfels.colors[far]
        )
        self.assertGreater(info["fallback_fraction"], 0.0)
        self.assertGreater(info["color_delta_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
