"""CPU regressions for the Gaussian runner's coordinate and update contracts."""

import unittest
from unittest import mock

import numpy as np

from gaussian_runner import (
    GaussianFrame,
    GaussianRunner,
    SceneNormalization,
    c2w_cv_to_viewmat,
    crop_intrinsics,
    resize_intrinsics,
    rgbd_to_point_cloud,
)


def make_pose(rotation, translation):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def project(K, point_camera):
    pixel_h = K @ point_camera
    return pixel_h[:2] / pixel_h[2]


def make_small_frame(frame_id="frame0", translation=(0.0, 0.0, 0.0)):
    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb[2, 3] = [255, 64, 0]
    depth = np.zeros((5, 6), dtype=np.float32)
    depth[2, 3] = 1.0
    mask = np.zeros((5, 6), dtype=bool)
    mask[2, 3] = True
    K = np.array([[2.0, 0.0, 2.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = translation
    return GaussianFrame(frame_id, rgb, depth, mask, K, c2w)


class GaussianGeometryTest(unittest.TestCase):
    def setUp(self):
        self.rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.c2w = make_pose(self.rotation, [0.3, -0.2, 0.7])
        self.K = np.array(
            [[400.0, 0.0, 320.0], [0.0, 420.0, 240.0], [0.0, 0.0, 1.0]]
        )

    def test_direct_cv_viewmat_and_gl_compatibility(self):
        normalization = SceneNormalization(1.0, np.zeros(3))
        direct = c2w_cv_to_viewmat(self.c2w, normalization)
        np.testing.assert_allclose(direct @ self.c2w, np.eye(4), atol=1e-6)

        point_camera = np.array([0.2, -0.1, 2.0, 1.0])
        point_object = self.c2w @ point_camera
        np.testing.assert_allclose(direct @ point_object, point_camera, atol=1e-6)

        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
        c2w_gl = self.c2w @ cv_to_gl
        via_legacy_gl = cv_to_gl @ np.linalg.inv(c2w_gl)
        np.testing.assert_allclose(direct, via_legacy_gl, atol=1e-6)
        self.assertFalse(
            np.allclose(direct, np.linalg.inv(c2w_gl) @ cv_to_gl, atol=1e-6)
        )

    def test_normalization_preserves_pixel_and_scales_camera_depth(self):
        normalization = SceneNormalization(2.5, [-0.15, 0.08, -0.30])
        point_camera = np.array([0.2, -0.1, 2.0, 1.0])
        point_object = (self.c2w @ point_camera)[:3]
        point_object_normalized = normalization.normalize_points(point_object)
        c2w_normalized = normalization.normalize_c2w(self.c2w)
        point_camera_normalized = (
            np.linalg.inv(c2w_normalized)
            @ np.append(point_object_normalized, 1.0)
        )[:3]

        np.testing.assert_allclose(
            point_camera_normalized, 2.5 * point_camera[:3], atol=1e-6
        )
        np.testing.assert_allclose(
            project(self.K, point_camera_normalized), [360.0, 219.0], atol=1e-6
        )
        np.testing.assert_allclose(
            normalization.metric_c2w(c2w_normalized), self.c2w, atol=1e-6
        )

    def test_crop_then_resize_intrinsics_matches_pixel_affine(self):
        cropped = crop_intrinsics(self.K, x0=80, y0=40)
        resized = resize_intrinsics(cropped, (480, 400), (240, 100))
        expected = np.array(
            [[200.0, 0.0, 120.0], [0.0, 105.0, 50.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(resized, expected)
        point_camera = np.array([0.2, -0.1, 2.0])
        np.testing.assert_allclose(
            project(resized, point_camera), [140.0, 44.75], atol=1e-6
        )
        np.testing.assert_allclose(
            self.K,
            [[400.0, 0.0, 320.0], [0.0, 420.0, 240.0], [0.0, 0.0, 1.0]],
        )

    def test_rgbd_backprojection_is_metric_cv_camera_to_object(self):
        frame = make_small_frame(translation=(0.1, -0.2, 0.3))
        cloud = rgbd_to_point_cloud(
            frame, voxel_size=0.001, min_depth=0.1, max_depth=2.0
        )
        self.assertEqual(cloud.raw_point_count, 1)
        np.testing.assert_allclose(
            cloud.points_metric[0], [0.6, 0.05, 1.3], atol=1e-6
        )
        np.testing.assert_allclose(
            cloud.colors[0], [1.0, 64.0 / 255.0, 0.0], atol=1e-6
        )

    def test_initialize_failure_returns_to_clean_state(self):
        runner = GaussianRunner(
            {"device": "cpu", "append_mask_erode_px": 0}, SceneNormalization(1.0, np.zeros(3))
        )
        with mock.patch.object(
            runner,
            "_reset_optimization_state",
            side_effect=RuntimeError("forced failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced failure"):
                runner.initialize([make_small_frame()], train_steps=0)
        self.assertFalse(runner.is_initialized)
        self.assertEqual(runner.num_gaussians, 0)
        self.assertEqual(runner.views, [])
        self.assertEqual(runner.update_index, -1)
        self.assertEqual(len(runner.observed_points_metric), 0)

    def test_update_failure_restores_splats_views_and_observed_cloud(self):
        runner = GaussianRunner(
            {"device": "cpu", "novelty_distance": 0.01, "append_mask_erode_px": 0},
            SceneNormalization(1.0, np.zeros(3)),
        )
        with mock.patch.object(runner, "_reset_optimization_state"):
            runner.initialize([make_small_frame()], train_steps=0)
        splats_before = {
            name: value.detach().clone() for name, value in runner.splats.items()
        }
        observed_before = runner.observed_points_metric.copy()

        with mock.patch.object(runner, "_reset_optimization_state"), mock.patch.object(
            runner, "train", side_effect=RuntimeError("forced train failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "forced train failure"):
                runner.update(
                    [make_small_frame("frame1", translation=(0.1, 0.0, 0.0))],
                    train_steps=1,
                )

        self.assertEqual(len(runner.views), 1)
        self.assertEqual(runner.update_index, 0)
        np.testing.assert_array_equal(
            runner.observed_points_metric, observed_before
        )
        for name, value in runner.splats.items():
            self.assertTrue(
                np.array_equal(value.detach().numpy(), splats_before[name].numpy())
            )

    def test_duplicate_frame_id_is_rejected_before_update(self):
        runner = GaussianRunner(
            {"device": "cpu", "append_mask_erode_px": 0}, SceneNormalization(1.0, np.zeros(3))
        )
        with mock.patch.object(runner, "_reset_optimization_state"):
            runner.initialize([make_small_frame()], train_steps=0)
        with self.assertRaisesRegex(ValueError, "already processed"):
            runner.update([make_small_frame()], train_steps=0)


if __name__ == "__main__":
    unittest.main()
