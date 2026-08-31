"""Renderer-switch (3DGS/2DGS) config, view-depth, and checkpoint tests.

CPU tests cover config validation and depth plumbing.  The CUDA class mirrors
``test_gaussian_gpu_integration`` gating and exercises the 2DGS training path.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gaussian_runner import (
    DEFAULT_CONFIG,
    GaussianFrame,
    GaussianRunner,
    SceneNormalization,
    TrainingView,
)


RUN_GPU_INTEGRATION = os.environ.get("RUN_GSPLAT_GPU_INTEGRATION") == "1"
TEST_DEVICE = os.environ.get("GAUSSIAN_TEST_DEVICE", "cuda:0")


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


class RendererConfigValidationTest(unittest.TestCase):
    def make_runner(self, **overrides):
        return GaussianRunner(
            dict(overrides), SceneNormalization(1.0, np.zeros(3)), device="cpu"
        )

    def test_defaults_keep_previous_behavior(self):
        self.assertEqual(DEFAULT_CONFIG["renderer"], "3dgs")
        self.assertEqual(float(DEFAULT_CONFIG["depth_loss_weight"]), 0.0)
        self.assertEqual(float(DEFAULT_CONFIG["normal_consistency_weight"]), 0.0)
        self.assertEqual(float(DEFAULT_CONFIG["distortion_weight"]), 0.0)
        self.make_runner()  # default config must validate

    def test_unknown_renderer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "renderer"):
            self.make_runner(renderer="4dgs")

    def test_2dgs_regularizers_require_2dgs_renderer(self):
        with self.assertRaisesRegex(ValueError, "normal_consistency_weight"):
            self.make_runner(renderer="3dgs", normal_consistency_weight=0.05)
        with self.assertRaisesRegex(ValueError, "distortion_weight"):
            self.make_runner(renderer="3dgs", distortion_weight=100.0)
        self.make_runner(
            renderer="2dgs",
            normal_consistency_weight=0.05,
            distortion_weight=100.0,
        )

    def test_2dgs_rejects_absgrad(self):
        with self.assertRaisesRegex(ValueError, "absgrad"):
            self.make_runner(renderer="2dgs", strategy={"absgrad": True})

    def test_depth_loss_on_3dgs_is_allowed_for_ablation(self):
        self.make_runner(renderer="3dgs", depth_loss_weight=1.0)

    def test_negative_or_nonfinite_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "depth_loss_weight"):
            self.make_runner(depth_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "depth_huber_delta_m"):
            self.make_runner(depth_huber_delta_m=0.0)
        with self.assertRaisesRegex(ValueError, "depth_alpha_threshold"):
            self.make_runner(depth_alpha_threshold=float("nan"))


class ViewDepthPlumbingTest(unittest.TestCase):
    def setUp(self):
        self.runner = GaussianRunner(
            {"roi_padding": 4},
            SceneNormalization(2.0, np.zeros(3)),
            device="cpu",
        )

    def test_prepare_view_stores_cropped_metric_depth(self):
        frame = make_plane_frame("f0").validated()
        view = self.runner._prepare_view(frame)
        self.assertIsNotNone(view.depth)
        x0, y0, x1, y1 = view.crop_xyxy
        np.testing.assert_array_equal(
            view.depth.numpy(), frame.depth[y0:y1, x0:x1]
        )
        self.assertEqual(view.depth.shape, view.mask.shape)
        self.assertEqual(view.depth.dtype, torch.float32)

    def test_checkpoint_views_round_trip_depth(self):
        frame = make_plane_frame("f0").validated()
        self.runner.views = [self.runner._prepare_view(frame)]
        item = self.runner._checkpoint_views()[0]
        self.assertIn("depth", item)
        torch.testing.assert_close(item["depth"], self.runner.views[0].depth)

    def test_training_view_depth_defaults_to_none(self):
        view = TrainingView(
            frame_id="legacy",
            rgb=torch.zeros(4, 4, 3),
            mask=torch.ones(4, 4, dtype=torch.bool),
            K=torch.eye(3),
            c2w_normalized=torch.eye(4),
            crop_xyxy=(0, 0, 4, 4),
        )
        self.assertIsNone(view.depth)


@unittest.skipUnless(RUN_GPU_INTEGRATION, "set RUN_GSPLAT_GPU_INTEGRATION=1")
class Gaussian2dgsGpuIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        import gsplat

        if gsplat.__version__ != "1.5.3":
            raise unittest.SkipTest(
                f"expected gsplat 1.5.3, got {gsplat.__version__}"
            )

    def make_config(self):
        return {
            "device": TEST_DEVICE,
            "voxel_size": 0.02,
            "novelty_distance": 0.01,
            "roi_padding": 4,
            "initial_steps": 5,
            "update_steps": 2,
            "sh_degree": 0,
            "renderer": "2dgs",
            "depth_loss_weight": 1.0,
            "normal_consistency_weight": 0.05,
            "normal_consistency_start_step": 0,
            "distortion_weight": 10.0,
            "distortion_start_step": 0,
            "strategy": {"refine_start_iter": 1000},
            "update_strategy": {
                "refine_start_iter": 1000,
                "refine_stop_iter": 1500,
            },
        }

    def test_2dgs_train_render_checkpoint(self):
        runner = GaussianRunner(
            self.make_config(),
            SceneNormalization(1.0, np.zeros(3)),
            device=TEST_DEVICE,
        )
        stats = runner.initialize([make_plane_frame("initial")])
        self.assertTrue(np.isfinite(stats.first_loss))
        self.assertTrue(np.isfinite(stats.final_loss))
        update_stats = runner.update(
            [make_plane_frame("update", camera_translation_x=0.08)]
        )
        self.assertTrue(np.isfinite(update_stats.final_loss))

        rgb, alpha, depth = runner.render(0, include_depth=True)
        self.assertEqual(rgb.shape[-1], 3)
        self.assertTrue(np.isfinite(rgb).all())
        self.assertTrue(np.isfinite(alpha).all())
        self.assertIsNotNone(depth)
        covered = alpha > 0.5
        if covered.any():
            self.assertGreater(float(np.median(depth[covered])), 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            runner.save_checkpoint(path)
            restored = GaussianRunner.load_checkpoint(path, device=TEST_DEVICE)
            self.assertEqual(restored.config["renderer"], "2dgs")
            for view in restored.views:
                self.assertIsNotNone(view.depth)
            restored.train(1)

    def test_legacy_checkpoint_without_depth_fails_loudly(self):
        runner = GaussianRunner(
            self.make_config(),
            SceneNormalization(1.0, np.zeros(3)),
            device=TEST_DEVICE,
        )
        runner.initialize([make_plane_frame("initial")])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            runner.save_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for item in payload["views"]:
                del item["depth"]
            torch.save(payload, path)
            restored = GaussianRunner.load_checkpoint(path, device=TEST_DEVICE)
            for view in restored.views:
                self.assertIsNone(view.depth)
            with self.assertRaisesRegex(RuntimeError, "depth"):
                restored.train(1)


if __name__ == "__main__":
    unittest.main()
