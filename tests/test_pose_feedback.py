"""Milestone ⑤ pose-feedback tests: SE(3) deltas, clamps, baking, write-back.

CPU tests need only torch/scipy; the gsplat checks are gated like
``tests/test_gaussian_gpu_integration.py`` (``RUN_GSPLAT_GPU_INTEGRATION=1``).
"""

import math
import os
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gaussian_runner import (
    GaussianFrame,
    GaussianRunner,
    PoseDeltas,
    SceneNormalization,
    se3_exp_batch,
)

RUN_GPU_INTEGRATION = os.environ.get("RUN_GSPLAT_GPU_INTEGRATION") == "1"
TEST_DEVICE = os.environ.get("GAUSSIAN_TEST_DEVICE", "cuda:0")


def random_pose(rng):
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
    c2w[:3, 3] = rng.normal(size=3)
    return c2w


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def make_plane_frame(frame_id, camera_translation_x=0.0):
    height = width = 64
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[24:40, 24:40] = [220, 80, 30]
    rgb[28:36, 28:36] = [30, 200, 220]
    depth = np.zeros((height, width), dtype=np.float32)
    depth[24:40, 24:40] = 1.0
    mask = depth > 0
    K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]],
                 dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = camera_translation_x
    return GaussianFrame(frame_id, rgb, depth, mask, K, c2w)


class Se3ExpTest(unittest.TestCase):
    def test_zero_is_identity(self):
        T = se3_exp_batch(torch.zeros((3, 6)))
        self.assertTrue(torch.allclose(T, torch.eye(4).expand(3, 4, 4)))

    def test_matches_scipy(self):
        rng = np.random.default_rng(0)
        delta = rng.normal(size=(8, 6)) * np.array([0.05] * 3 + [0.8] * 3)
        T = se3_exp_batch(torch.from_numpy(delta)).numpy()
        for row, matrix in zip(delta, T):
            expected = Rotation.from_rotvec(row[3:]).as_matrix()
            self.assertTrue(np.allclose(matrix[:3, :3], expected, atol=1e-9))
            self.assertTrue(np.allclose(matrix[:3, 3], row[:3]))
            self.assertTrue(np.allclose(matrix[3], [0, 0, 0, 1]))
            self.assertTrue(np.allclose(matrix[:3, :3].T @ matrix[:3, :3],
                                        np.eye(3), atol=1e-9))

    def test_gradient_alive_at_zero(self):
        delta = torch.zeros((1, 6), dtype=torch.float64, requires_grad=True)
        point = torch.tensor([0.3, -0.2, 0.9, 1.0], dtype=torch.float64)
        target = torch.tensor([0.35, -0.1, 0.8, 1.0], dtype=torch.float64)
        loss = ((se3_exp_batch(delta)[0] @ point - target) ** 2).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertGreater(float(delta.grad[0, 3:].abs().max()), 0.0)
        self.assertGreater(float(delta.grad[0, :3].abs().max()), 0.0)


class PoseDeltasTest(unittest.TestCase):
    def test_zero_is_identity_everywhere(self):
        deltas = PoseDeltas(4, max_trans_norm=0.1, max_rot_rad=0.3)
        T = deltas.matrices([0, 1, 2, 3])
        self.assertTrue(torch.allclose(T, torch.eye(4).expand(4, 4, 4)))
        rot, trans = deltas.magnitudes()
        self.assertEqual(float(rot.abs().max()), 0.0)
        self.assertEqual(float(trans.abs().max()), 0.0)

    def test_first_view_fixed_and_clamped(self):
        deltas = PoseDeltas(3, max_trans_norm=0.02, max_rot_rad=math.radians(20.0))
        with torch.no_grad():
            deltas.data.fill_(50.0)   # tanh saturates -> component clamps
        T = deltas.matrices([0, 1, 2])
        self.assertTrue(torch.allclose(T[0], torch.eye(4)))
        self.assertFalse(torch.allclose(T[1], torch.eye(4)))
        rows = deltas.deltas()
        self.assertLessEqual(float(rows[:, :3].abs().max()), 0.02 + 1e-6)
        self.assertLessEqual(float(rows[:, 3:].abs().max()),
                             math.radians(20.0) + 1e-6)
        rot, trans = deltas.magnitudes()
        self.assertEqual(float(rot[0]), 0.0)
        self.assertLessEqual(float(rot[1]), math.sqrt(3.0) * math.radians(20.0) + 1e-6)
        self.assertLessEqual(float(trans[1]), math.sqrt(3.0) * 0.02 + 1e-6)

    def test_unfixed_first_view_moves(self):
        deltas = PoseDeltas(2, max_trans_norm=0.02, max_rot_rad=0.3, fix_first=False)
        with torch.no_grad():
            deltas.data[0, 3] = 0.5
        self.assertFalse(torch.allclose(deltas.matrices([0])[0], torch.eye(4)))


class NormalizationConsistencyTest(unittest.TestCase):
    def test_normalized_delta_is_a_metric_rigid_motion(self):
        """metric_c2w(T_n @ normalize(P)) == T_m @ P with T_m = (R, R t - t + d/s)."""
        rng = np.random.default_rng(1)
        scale, translation = 2.5, np.array([0.1, -0.2, 0.3])
        normalization = SceneNormalization(scale=scale, translation=translation)
        delta = torch.from_numpy(rng.normal(size=(1, 6)) * np.array([0.05] * 3 + [0.3] * 3))
        T_n = se3_exp_batch(delta)[0].numpy()
        R, d = T_n[:3, :3], T_n[:3, 3]
        T_m = np.eye(4)
        T_m[:3, :3] = R
        T_m[:3, 3] = R @ translation - translation + d / scale
        for _ in range(5):
            P = random_pose(rng)
            via_normalized = normalization.metric_c2w(T_n @ normalization.normalize_c2w(P))
            self.assertTrue(np.allclose(via_normalized, T_m @ P, atol=1e-5))


class RunnerCpuTest(unittest.TestCase):
    def _runner(self):
        normalization = SceneNormalization(scale=2.0, translation=np.array([0.1, -0.2, 0.3]))
        runner = GaussianRunner({"device": "cpu", "roi_padding": 4}, normalization, device="cpu")
        frames = [make_plane_frame("a"), make_plane_frame("b", camera_translation_x=0.05)]
        runner.views = [runner._prepare_view(frame) for frame in frames]
        return runner, frames

    def test_view_poses_metric_round_trip(self):
        runner, frames = self._runner()
        poses = runner.view_poses_metric()
        self.assertEqual(poses.shape, (2, 4, 4))
        for pose, frame in zip(poses, frames):
            self.assertTrue(np.allclose(pose, frame.c2w_cv, atol=1e-5))

    def test_bake_applies_delta_and_logs(self):
        runner, frames = self._runner()
        deltas = runner._new_pose_deltas(2, fix_first=True)
        with torch.no_grad():
            deltas.data[1] = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.1, 0.0])
        before = [view.c2w_normalized.clone() for view in runner.views]
        T1 = deltas.matrices([1])[0].detach()
        record = runner._bake_pose_deltas(deltas, runner.views, "test")
        self.assertTrue(torch.allclose(runner.views[0].c2w_normalized, before[0]))
        self.assertTrue(torch.allclose(runner.views[1].c2w_normalized, T1 @ before[1], atol=1e-5))
        self.assertEqual(len(runner.feedback_log), 1)
        self.assertEqual(record["views"], 2)
        self.assertEqual(record["per_view_rot_deg"][0], 0.0)
        self.assertGreater(record["max_rot_deg"], 0.0)
        self.assertGreater(record["max_trans_mm"], 0.0)
        # clamp scale: max_trans_m * scale in normalized units
        self.assertAlmostEqual(deltas.max_trans_norm, 0.02 * 2.0)
        self.assertAlmostEqual(deltas.max_rot_rad, math.radians(20.0))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            GaussianRunner({"device": "cpu", "pose_feedback": {"lr_decay": 0.0}},
                           SceneNormalization(1.0, np.zeros(3)), device="cpu")
        with self.assertRaises(ValueError):
            GaussianRunner({"device": "cpu", "pose_feedback": {"max_rot_deg": -1.0}},
                           SceneNormalization(1.0, np.zeros(3)), device="cpu")


@unittest.skipUnless(RUN_GPU_INTEGRATION, "set RUN_GSPLAT_GPU_INTEGRATION=1")
class PoseFeedbackGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        import gsplat

        if gsplat.__version__ != "1.5.3":
            raise unittest.SkipTest(f"expected gsplat 1.5.3, got {gsplat.__version__}")

    def _fitted_runner(self, renderer):
        config = {
            "device": TEST_DEVICE,
            "voxel_size": 0.02,
            "novelty_distance": 0.01,
            "roi_padding": 4,
            "initial_steps": 1,
            "update_steps": 0,
            "sh_degree": 0,
            "renderer": renderer,
            "depth_loss_weight": 0.5 if renderer == "2dgs" else 0.0,
            "strategy": {"refine_start_iter": 1000},
            "update_strategy": {"refine_start_iter": 1000, "refine_stop_iter": 1500},
            "pose_feedback": {"enabled": True, "in_initial": False},
        }
        runner = GaussianRunner(config, SceneNormalization(1.0, np.zeros(3)),
                                device=TEST_DEVICE)
        runner.initialize([make_plane_frame("a"), make_plane_frame("b", 0.03)],
                          train_steps=150)
        return runner

    def _check(self, renderer):
        # v1 joint training: a 2 deg error injected into the movable view is reduced by
        # pose-only-free joint steps (map + deltas), the record is logged, poses round-trip.
        runner = self._fitted_runner(renderer)
        view = runner.views[1]
        clean = view.c2w_normalized.clone()
        injected = se3_exp_batch(torch.tensor([[0.0, 0.0, 0.0, 0.0, math.radians(2.0), 0.0]]))[0]
        view.c2w_normalized = (injected @ clean).contiguous()
        before = rotation_angle_deg(view.c2w_normalized[:3, :3].numpy().T @ clean[:3, :3].numpy())
        n_log = len(runner.feedback_log)
        first_loss, final_loss = runner.train(60, optimize_poses=True)
        self.assertTrue(np.isfinite(first_loss) and np.isfinite(final_loss))
        self.assertEqual(len(runner.feedback_log), n_log + 1)
        record = runner.feedback_log[-1]
        self.assertTrue(record["event"].startswith("train_"))
        self.assertGreater(record["max_rot_deg"], 0.0)
        self.assertEqual(record["per_view_rot_deg"][0], 0.0)          # first view fixed
        residual = rotation_angle_deg(view.c2w_normalized[:3, :3].numpy().T @ clean[:3, :3].numpy())
        self.assertLess(residual, before, f"{renderer}: joint training did not reduce the injected error")
        self.assertEqual(runner.view_poses_metric().shape, (2, 4, 4))
        # v1 path through update(): one joint-training record per cycle, poses stay valid.
        stats = runner.update([make_plane_frame("c", 0.06)], train_steps=3)
        self.assertEqual(stats.train_steps, 3)
        self.assertEqual(len(runner.feedback_log), n_log + 2)
        self.assertTrue(runner.feedback_log[-1]["event"].startswith("train_"))
        self.assertEqual(runner.view_poses_metric().shape, (3, 4, 4))
        # disabled: no deltas, no record
        runner.config["pose_feedback"]["enabled"] = False
        runner.update([make_plane_frame("d", 0.09)], train_steps=3)
        self.assertEqual(len(runner.feedback_log), n_log + 2)

    def test_joint_training_2dgs(self):
        self._check("2dgs")

    def test_joint_training_3dgs(self):
        self._check("3dgs")


if __name__ == "__main__":
    unittest.main()
