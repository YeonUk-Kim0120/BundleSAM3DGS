"""Optional gsplat CUDA integration test for append, render, and persistence."""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from gaussian_runner import GaussianFrame, GaussianRunner, SceneNormalization
from gaussian_budget_strategy import apply_budgeted_topology


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


@unittest.skipUnless(RUN_GPU_INTEGRATION, "set RUN_GSPLAT_GPU_INTEGRATION=1")
class GaussianGpuIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        import gsplat

        if gsplat.__version__ != "1.5.3":
            raise unittest.SkipTest(f"expected gsplat 1.5.3, got {gsplat.__version__}")

    def test_initialize_append_checkpoint_render_and_export(self):
        config = {
            "device": TEST_DEVICE,
            "voxel_size": 0.02,
            "novelty_distance": 0.01,
            "roi_padding": 4,
            "initial_steps": 1,
            "update_steps": 0,
            "sh_degree": 0,
            "strategy": {"refine_start_iter": 1000},
            "update_strategy": {
                "refine_start_iter": 1000,
                "refine_stop_iter": 1500,
            },
        }
        runner = GaussianRunner(
            config, SceneNormalization(1.0, np.zeros(3)), device=TEST_DEVICE
        )
        initial_stats = runner.initialize([make_plane_frame("initial")])
        self.assertTrue(np.isfinite(initial_stats.first_loss))
        self.assertTrue(np.isfinite(initial_stats.final_loss))
        gaussians_before = runner.num_gaussians
        values_before = {
            name: value.detach().clone() for name, value in runner.splats.items()
        }

        update_stats = runner.update(
            [make_plane_frame("update", camera_translation_x=0.08)]
        )
        self.assertGreater(update_stats.novel_points, 0)
        self.assertGreater(runner.num_gaussians, gaussians_before)
        for name, value in runner.splats.items():
            self.assertTrue(
                torch.equal(value[:gaussians_before], values_before[name])
            )

        with tempfile.TemporaryDirectory(prefix="gaussian-runner-test-") as directory:
            directory = Path(directory)
            checkpoint = directory / "checkpoint.pt"
            runner.save_checkpoint(checkpoint)
            loaded = GaussianRunner.load_checkpoint(checkpoint, device=TEST_DEVICE)
            self.assertEqual(loaded.num_gaussians, runner.num_gaussians)
            self.assertEqual(len(loaded.views), 2)
            for name in runner.splats.keys():
                self.assertTrue(torch.equal(loaded.splats[name], runner.splats[name]))

            first_loss, final_loss = loaded.train(1)
            self.assertTrue(np.isfinite(first_loss))
            self.assertTrue(np.isfinite(final_loss))
            rgb, alpha, depth_metric = loaded.render(1, include_depth=True)
            self.assertEqual(rgb.shape, (*alpha.shape, 3))
            self.assertEqual(depth_metric.shape, alpha.shape)
            self.assertTrue(np.isfinite(rgb).all())
            self.assertTrue(np.isfinite(alpha).all())
            self.assertTrue(np.isfinite(depth_metric).all())
            max_alpha_pixel = np.unravel_index(np.argmax(alpha), alpha.shape)
            self.assertGreater(alpha[max_alpha_pixel], 0.0)
            self.assertAlmostEqual(float(depth_metric[max_alpha_pixel]), 1.0, delta=0.2)

            normalized_ply = directory / "normalized.ply"
            metric_ply = directory / "metric.ply"
            loaded.export_ply(normalized_ply, metric_ply)
            self.assertGreater(normalized_ply.stat().st_size, 0)
            self.assertGreater(metric_ply.stat().st_size, 0)

    def test_budget_topology_preserves_count_and_lr_horizon(self):
        config = {
            "device": TEST_DEVICE,
            "voxel_size": 0.02,
            "novelty_distance": 0.01,
            "roi_padding": 4,
            "initial_steps": 1,
            "update_steps": 0,
            "sh_degree": 0,
            "strategy": {"refine_start_iter": 1000},
            "update_strategy": {
                "refine_start_iter": 1000,
                "refine_stop_iter": 1500,
            },
        }
        runner = GaussianRunner(
            config, SceneNormalization(1.0, np.zeros(3)), device=TEST_DEVICE
        )
        runner.initialize([make_plane_frame("initial")])
        update_stats = runner.update(
            [make_plane_frame("update", camera_translation_x=0.08)]
        )
        base_update_lr = float(runner.optimizers["means"].param_groups[0]["lr"])
        runner.train(2, lr_decay_horizon_steps=4)
        self.assertAlmostEqual(
            float(runner.optimizers["means"].param_groups[0]["lr"]),
            base_update_lr * 0.1,
            places=12,
        )

        with tempfile.TemporaryDirectory(prefix="gaussian-budget-test-") as directory:
            checkpoint = Path(directory) / "common.pt"
            runner.save_checkpoint(checkpoint)
            branch = GaussianRunner.load_checkpoint(checkpoint, device=TEST_DEVICE)
            count_before = branch.num_gaussians
            event = apply_budgeted_topology(
                branch.splats,
                branch.optimizers,
                branch.strategy_state,
                fraction=0.05,
                protected_start=update_stats.gaussians_before,
            )
            self.assertGreater(event.prune_count, 0)
            self.assertEqual(event.duplicate_count + event.split_count, event.prune_count)
            self.assertEqual(branch.num_gaussians, count_before)
            self.assertTrue(
                all(
                    index < update_stats.gaussians_before
                    for index in event.prune_indices
                )
            )
            for parameter in branch.splats.values():
                self.assertTrue(torch.isfinite(parameter).all())

            branch.train(2, lr_decay_horizon_steps=4)
            self.assertAlmostEqual(
                float(branch.optimizers["means"].param_groups[0]["lr"]),
                base_update_lr * 0.01,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
