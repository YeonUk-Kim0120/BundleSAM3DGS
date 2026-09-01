"""Milestone-⑤ pose-feedback tests (tracker-authority delta variant)."""

import os
import unittest

import numpy as np
import torch

from gaussian_runner import (
    GaussianFrame,
    GaussianRunner,
    SceneNormalization,
    se3_exp,
)
from sam3d_prior import MeshPrior, sample_surfels

RUN_GPU_INTEGRATION = os.environ.get("RUN_GSPLAT_GPU_INTEGRATION") == "1"
TEST_DEVICE = os.environ.get("GAUSSIAN_TEST_DEVICE", "cuda:0")

FEEDBACK_ON = {
    "renderer": "2dgs",
    "pose_feedback": {"enabled": True},
}


def make_plane_frame(frame_id, camera_translation_x=0.0):
    height = width = 64
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[24:40, 24:40] = [220, 80, 30]
    depth = np.zeros((height, width), dtype=np.float32)
    depth[24:40, 24:40] = 1.0
    mask = depth > 0
    K = np.array(
        [[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = camera_translation_x
    return GaussianFrame(frame_id, rgb, depth, mask, K, c2w)


class Se3ExpTest(unittest.TestCase):
    def test_zero_delta_is_identity(self):
        T = se3_exp(torch.zeros(6))
        torch.testing.assert_close(T, torch.eye(4), atol=1e-6, rtol=0)

    def test_known_rotation_and_translation(self):
        delta = torch.tensor([0.0, 0.0, np.pi / 2, 0.01, 0.0, 0.0])
        T = se3_exp(delta)
        expected_R = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        torch.testing.assert_close(T[:3, :3], expected_R, atol=1e-6, rtol=0)
        torch.testing.assert_close(
            T[:3, 3], torch.tensor([0.01, 0.0, 0.0]), atol=1e-8, rtol=0
        )

    def test_differentiable(self):
        delta = torch.zeros(6, requires_grad=True)
        T = se3_exp(delta)
        T.sum().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())


class FeedbackClampTest(unittest.TestCase):
    def make_runner(self):
        return GaussianRunner(
            {**FEEDBACK_ON, "roi_padding": 4},
            SceneNormalization(2.0, np.zeros(3)),
            device="cpu",
        )

    def test_disabled_returns_none(self):
        runner = GaussianRunner(
            {"renderer": "2dgs"}, SceneNormalization(1.0, np.zeros(3)),
            device="cpu",
        )
        self.assertIsNone(runner.get_feedback_poses())

    def test_zero_delta_returns_tracker_pose_and_clamp_limits(self):
        runner = self.make_runner()
        frame = make_plane_frame("f0", camera_translation_x=0.02).validated()
        runner.views = [runner._prepare_view(frame)]
        runner._pose_deltas = torch.zeros((1, 6))
        poses, stats = runner.get_feedback_poses()
        np.testing.assert_allclose(poses["f0"], frame.c2w_cv, atol=1e-5)
        self.assertEqual(stats["clipped"], 0)

        # Oversized delta must be clamped to the configured limits
        # (3 mm translation, 3° rotation by default).
        runner._pose_deltas = torch.tensor(
            [[0.0, 0.0, 1.0, 0.5, 0.0, 0.0]]  # 57° rotation, 0.25 m metric
        )
        poses, stats = runner.get_feedback_poses()
        self.assertEqual(stats["clipped"], 1)
        moved = poses["f0"]
        trans_mm = np.linalg.norm(moved[:3, 3] - frame.c2w_cv[:3, 3]) * 1000
        self.assertLessEqual(trans_mm, 3.0 + 1e-3)
        rotation_error = np.arccos(
            np.clip((np.trace(moved[:3, :3].T @ frame.c2w_cv[:3, :3]) - 1) / 2,
                    -1, 1)
        )
        self.assertLessEqual(np.degrees(rotation_error), 3.0 + 1e-3)

    def test_config_guard(self):
        with self.assertRaisesRegex(ValueError, "pose_feedback.lr"):
            GaussianRunner(
                {"renderer": "2dgs",
                 "pose_feedback": {"enabled": True, "lr": 0.0}},
                SceneNormalization(1.0, np.zeros(3)), device="cpu",
            )


@unittest.skipUnless(RUN_GPU_INTEGRATION, "set RUN_GSPLAT_GPU_INTEGRATION=1")
class FeedbackGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")

    def test_training_produces_finite_nonzero_deltas(self):
        config = {
            **FEEDBACK_ON,
            "device": TEST_DEVICE,
            "voxel_size": 0.02,
            "novelty_distance": 0.01,
            "roi_padding": 4,
            "initial_steps": 20,
            "sh_degree": 0,
            "depth_loss_weight": 1.0,
            "strategy": {"refine_start_iter": 1000},
        }
        runner = GaussianRunner(
            config, SceneNormalization(1.0, np.zeros(3)), device=TEST_DEVICE
        )
        runner.initialize([make_plane_frame("f0")])
        self.assertIsNotNone(runner._pose_deltas)
        self.assertTrue(torch.isfinite(runner._pose_deltas).all())
        self.assertGreater(float(runner._pose_deltas.abs().sum()), 0.0)
        poses, stats = runner.get_feedback_poses()
        self.assertIn("f0", poses)
        self.assertLessEqual(stats["trans_mm_max"], 3.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
