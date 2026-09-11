"""CPU tests for gaussian_global (mesh extraction helpers on synthetic data; no gsplat needed)."""
import math
import unittest

import numpy as np
import trimesh

from gaussian_global import DEFAULT_GLOBAL_CONFIG, fibonacci_sphere, largest_component, look_at_c2w, poisson_mesh, tsdf_fuse


def sphere_points(n=4000, r=0.05, seed=0):
    d = fibonacci_sphere(n)
    return d * r, d


class GeometryHelpersTest(unittest.TestCase):
    def test_look_at_is_a_valid_opencv_c2w(self):
        eye, target = np.array([0.3, -0.1, 0.2]), np.zeros(3)
        c2w = look_at_c2w(eye, target)
        R = c2w[:3, :3]
        self.assertTrue(np.allclose(R.T @ R, np.eye(3), atol=1e-9))
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)
        self.assertTrue(np.allclose(R[:, 2], (target - eye) / np.linalg.norm(target - eye)))   # z axis looks at target

    def test_fibonacci_sphere_unit_and_spread(self):
        d = fibonacci_sphere(200)
        self.assertTrue(np.allclose(np.linalg.norm(d, axis=1), 1.0))
        self.assertLess(np.abs(d.mean(0)).max(), 0.05)


class PoissonTest(unittest.TestCase):
    def test_sphere_from_points_and_normals(self):
        pts, normals = sphere_points()
        mesh = poisson_mesh(pts, normals, None, depth=6, density_quantile=0.05)
        mesh = largest_component(mesh)
        self.assertGreater(len(mesh.vertices), 100)
        radii = np.linalg.norm(mesh.vertices, axis=1)
        self.assertLess(np.abs(radii - 0.05).mean(), 0.003)      # surface within 3 mm of the true sphere


class TsdfTest(unittest.TestCase):
    def test_sphere_from_rendered_depth(self):
        # analytic depth images of a sphere (r = 5 cm at the origin) from 20 cameras on a 30 cm shell
        r, K, W, H = 0.05, np.array([[300.0, 0, 64.0], [0, 300.0, 48.0], [0, 0, 1.0]]), 128, 96
        frames = []
        for d in fibonacci_sphere(20):
            c2w = look_at_c2w(0.3 * d, np.zeros(3))
            w2c = np.linalg.inv(c2w)
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            dirs = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=float)], -1)
            dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
            o = w2c[:3, :3] @ np.zeros(3) + w2c[:3, 3]                 # sphere centre in camera coords
            b = (dirs * o).sum(-1); c = o @ o - r * r
            disc = b * b - c
            t = np.where(disc > 0, b - np.sqrt(np.maximum(disc, 0)), 0.0)
            depth = (t * dirs[..., 2]).astype(np.float32)               # camera-z depth, 0 = miss
            rgb = np.full((H, W, 3), 128, np.uint8)
            frames.append((rgb, depth, K, c2w))
        mesh = largest_component(tsdf_fuse(frames, voxel=0.002, trunc=0.01))
        self.assertGreater(len(mesh.vertices), 500)
        radii = np.linalg.norm(mesh.vertices, axis=1)
        self.assertLess(np.abs(radii - r).mean(), 0.003)


class ComponentTest(unittest.TestCase):
    def test_largest_component_keeps_the_big_piece(self):
        big = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
        small = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
        small.apply_translation([0.2, 0, 0])
        both = trimesh.util.concatenate([big, small])
        kept = largest_component(both)
        self.assertEqual(len(kept.vertices), len(big.vertices))

    def test_default_config_is_the_confirmed_setting(self):
        self.assertEqual(DEFAULT_GLOBAL_CONFIG["steps"], 2000)
        self.assertFalse(DEFAULT_GLOBAL_CONFIG["pose_refine"])
        self.assertAlmostEqual(DEFAULT_GLOBAL_CONFIG["opacity_min"], 0.1)
        self.assertAlmostEqual(DEFAULT_GLOBAL_CONFIG["max_scale_mm"], 10.0)


if __name__ == "__main__":
    unittest.main()
