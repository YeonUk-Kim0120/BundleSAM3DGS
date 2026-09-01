"""CPU tests for the ③b bias-field math.

Run: python3 -m unittest experiments.test_bias_field
"""

import math
import unittest

import numpy as np
import torch

from experiments.bias_field import (
    BiasFieldConfig,
    anchor_distance_damping,
    estimate_bias_offsets,
    fit_global_affine_residual,
    harmonic_interpolate,
    knn_graph,
    robust_anchor_residuals,
)


def sphere_points(n=600, radius=0.08, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    means = torch.from_numpy((v * radius).astype(np.float32))
    normals = torch.from_numpy(v.astype(np.float32))
    return means, normals


class AnchorResidualTest(unittest.TestCase):
    def test_mean_and_min_count(self):
        s = torch.tensor([0.006, 0.010, 0.0])
        c = torch.tensor([3, 2, 0], dtype=torch.int32)
        mean, has = robust_anchor_residuals(s, c, min_count=3)
        self.assertTrue(bool(has[0]))
        self.assertFalse(bool(has[1]))  # below min_count
        self.assertAlmostEqual(float(mean[0]), 0.002, places=6)


class GlobalAffineTest(unittest.TestCase):
    def test_recovers_anisotropic_scale(self):
        # True residual: surface inflated 5% along z only →
        # δ = n·(diag(0,0,0.05)·x). Anchors on the +z hemisphere only.
        means, normals = sphere_points()
        true_s = torch.tensor([0.0, 0.0, 0.05])
        residuals = (normals * (means * true_s[None, :])).sum(-1)
        anchors = means[:, 2] > 0.01
        params, predicted = fit_global_affine_residual(
            means, normals, residuals, anchors
        )
        torch.testing.assert_close(params[:3], true_s, atol=2e-3, rtol=0)
        # Extrapolation: the unanchored −z hemisphere is predicted too.
        err = (predicted - residuals)[~anchors].abs().max()
        self.assertLess(float(err), 5e-4)


class HarmonicTest(unittest.TestCase):
    def test_hole_surrounded_by_anchors_interpolates(self):
        means, _ = sphere_points(n=800, seed=1)
        hole = means[:, 2] > 0.06  # small cap = "hand patch"
        anchors = ~hole
        anchor_values = torch.full((len(means),), -0.005)
        idx = knn_graph(means, k=8)
        filled = harmonic_interpolate(idx, anchors, anchor_values, iters=300)
        self.assertAlmostEqual(float(filled[hole].mean()), -0.005, places=3)

    def test_damping_decays_with_distance(self):
        means = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0],
                              [0.10, 0.0, 0.0]])
        anchors = torch.tensor([True, False, False])
        w = anchor_distance_damping(means, anchors, sigma_m=0.015)
        self.assertAlmostEqual(float(w[0]), 1.0, places=5)
        self.assertGreater(float(w[1]), 0.7)
        self.assertLess(float(w[2]), 1e-6)


class EndToEndTest(unittest.TestCase):
    def test_bulge_band_pulled_inward_and_capped(self):
        # Sphere with a verified band (z∈[0, 0.05]) that measured a −6mm
        # inward residual; the unverified band just above should inherit
        # ≈−6mm (interpolation), the far bottom decays toward global-only.
        means, normals = sphere_points(n=1000, seed=2)
        n = len(means)
        residual_sum = torch.zeros(n)
        residual_count = torch.zeros(n, dtype=torch.int32)
        band = (means[:, 2] > 0.0) & (means[:, 2] < 0.05)
        residual_sum[band] = -0.006 * 3
        residual_count[band] = 3
        config = BiasFieldConfig(global_stage=False, offset_cap_m=0.004)
        offsets, info = estimate_bias_offsets(
            means, normals, residual_sum, residual_count, config
        )
        near_band = (means[:, 2] >= 0.05) & (means[:, 2] < 0.07)
        self.assertLess(float(offsets[near_band].mean()), -0.002)
        # cap respected everywhere (−6mm anchors clamped to −4mm)
        self.assertLessEqual(float(offsets.abs().max()), 0.004 + 1e-9)
        self.assertGreater(info["anchors"], 0)


if __name__ == "__main__":
    unittest.main()
