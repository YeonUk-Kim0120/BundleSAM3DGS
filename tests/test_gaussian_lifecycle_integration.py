"""Lifecycle integration tests for GaussianRunner (milestone ④ port).

CPU tests cover config guards and pose refresh; the CUDA class mirrors the
existing GPU gating and exercises prior init → update → checkpoint.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gaussian_runner import GaussianFrame, GaussianRunner, SceneNormalization
from prior_lifecycle import STATE_UNSEEN, STATE_VERIFIED
from sam3d_prior import MeshPrior, sample_surfels

RUN_GPU_INTEGRATION = os.environ.get("RUN_GSPLAT_GPU_INTEGRATION") == "1"
TEST_DEVICE = os.environ.get("GAUSSIAN_TEST_DEVICE", "cuda:0")

LIFECYCLE_ON = {
    "renderer": "2dgs",
    "prior_lifecycle": {"enabled": True},
    "strategy": {
        "refine_start_iter": 100_000_000,
        "refine_stop_iter": 100_000_001,
        "reset_every": 100_000_000,
    },
    "update_strategy": {
        "refine_start_iter": 100_000_000,
        "refine_stop_iter": 100_000_001,
    },
}


def make_plane_frame(frame_id, camera_translation_x=0.0, depth_value=1.0):
    height = width = 64
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[24:40, 24:40] = [220, 80, 30]
    depth = np.zeros((height, width), dtype=np.float32)
    depth[24:40, 24:40] = depth_value
    mask = depth > 0
    K = np.array(
        [[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = camera_translation_x
    return GaussianFrame(frame_id, rgb, depth, mask, K, c2w)


def plane_prior_surfels(z=1.0, extent=0.24, n=400):
    """A flat square mesh at depth ``z`` in the first-camera frame."""

    half = extent / 2
    vertices = torch.tensor(
        [[-half, -half, z], [half, -half, z], [half, half, z],
         [-half, half, z]], dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    colors = torch.full((4, 3), 0.6)
    prior = MeshPrior(vertices, faces, colors)
    return sample_surfels(prior, n, seed=0, radius_multiplier=0.75)


class LifecycleConfigGuardTest(unittest.TestCase):
    def make(self, **overrides):
        return GaussianRunner(
            overrides, SceneNormalization(1.0, np.zeros(3)), device="cpu"
        )

    def test_valid_lifecycle_config(self):
        runner = self.make(**LIFECYCLE_ON)
        self.assertTrue(runner.lifecycle_enabled)

    def test_requires_2dgs(self):
        bad = {**LIFECYCLE_ON, "renderer": "3dgs"}
        with self.assertRaisesRegex(ValueError, "renderer '2dgs'"):
            self.make(**bad)

    def test_requires_densify_off(self):
        bad = {**LIFECYCLE_ON,
               "strategy": {**LIFECYCLE_ON["strategy"],
                            "refine_start_iter": 500}}
        with self.assertRaisesRegex(ValueError, "densification disabled"):
            self.make(**bad)

    def test_requires_reset_off(self):
        bad = {**LIFECYCLE_ON,
               "strategy": {**LIFECYCLE_ON["strategy"],
                            "reset_every": 3000}}
        with self.assertRaisesRegex(ValueError, "opacity reset disabled"):
            self.make(**bad)

    def test_initialize_from_prior_requires_lifecycle(self):
        runner = self.make(renderer="2dgs")
        with self.assertRaisesRegex(RuntimeError, "prior_lifecycle"):
            runner.initialize_from_prior(
                plane_prior_surfels(), np.eye(4), [make_plane_frame("f0")]
            )


class PoseRefreshTest(unittest.TestCase):
    def test_refresh_updates_matching_views(self):
        runner = GaussianRunner(
            {"roi_padding": 4},
            SceneNormalization(2.0, np.array([0.1, 0.0, 0.0])),
            device="cpu",
        )
        frame = make_plane_frame("f0").validated()
        runner.views = [runner._prepare_view(frame)]
        new_pose = np.eye(4, dtype=np.float32)
        new_pose[0, 3] = 0.05
        refreshed = runner.refresh_view_poses({"f0": new_pose, "zz": new_pose})
        self.assertEqual(refreshed, 1)
        expected = runner.normalization.normalize_c2w(new_pose)
        np.testing.assert_allclose(
            runner.views[0].c2w_normalized.numpy(), expected, atol=1e-6
        )


@unittest.skipUnless(RUN_GPU_INTEGRATION, "set RUN_GSPLAT_GPU_INTEGRATION=1")
class LifecycleGpuIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        import gsplat

        if gsplat.__version__ != "1.5.3":
            raise unittest.SkipTest(f"gsplat {gsplat.__version__}")

    def make_runner(self):
        config = {
            **LIFECYCLE_ON,
            "device": TEST_DEVICE,
            "voxel_size": 0.02,
            "novelty_distance": 0.01,
            "roi_padding": 4,
            "initial_steps": 5,
            "update_steps": 2,
            "sh_degree": 0,
            "depth_loss_weight": 1.0,
        }
        return GaussianRunner(
            config, SceneNormalization(1.0, np.zeros(3)), device=TEST_DEVICE
        )

    def test_prior_init_update_checkpoint(self):
        runner = self.make_runner()
        surfels = plane_prior_surfels()
        stats = runner.initialize_from_prior(
            surfels, np.eye(4), [make_plane_frame("initial")]
        )
        self.assertTrue(np.isfinite(stats.final_loss))
        fields = runner.lifecycle_fields
        self.assertIsNotNone(fields)
        summary = fields.summary()
        # The frame observes the plane at its true depth → most surfels
        # inside the (eroded) mask verify; off-mask ones stay UNSEEN.
        self.assertGreater(summary["verified"], 0)
        self.assertGreater(summary["unseen"], 0)
        self.assertEqual(summary["prior_lineage"], summary["total"])

        unseen_before = runner.splats["means"].detach()[
            fields.state == STATE_UNSEEN
        ].clone()
        update_stats = runner.update(
            [make_plane_frame("update", camera_translation_x=0.05)]
        )
        self.assertTrue(np.isfinite(update_stats.final_loss))
        fields = runner.lifecycle_fields
        appended = ~fields.lineage
        if bool(appended.any()):
            self.assertTrue(
                (fields.state[appended] == STATE_VERIFIED).all()
            )
        # Frozen UNSEEN must not move during training: every splat that is
        # still UNSEEN must sit exactly at one of the pre-update UNSEEN
        # positions (indices shift with appends/removals, so match by value).
        still_unseen_rows = torch.where(
            fields.lineage & (fields.state == STATE_UNSEEN)
        )[0]
        unseen_after = runner.splats["means"].detach()[still_unseen_rows]
        if len(unseen_after):
            before_set = {
                tuple(np.round(row, 6))
                for row in unseen_before.cpu().numpy()
            }
            for row in unseen_after.cpu().numpy():
                self.assertIn(tuple(np.round(row, 6)), before_set)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            runner.save_checkpoint(path)
            restored = GaussianRunner.load_checkpoint(path, device=TEST_DEVICE)
            self.assertIsNotNone(restored.lifecycle_fields)
            self.assertEqual(
                restored.lifecycle_fields.summary(), fields.summary()
            )
            restored.train(1)


if __name__ == "__main__":
    unittest.main()
