"""Optional GPU integration checks for matching and native optimization.

Run with ``RUN_GPU_INTEGRATION=1`` inside the project container. These tests
require the built native module, LoFTR weights, and YCB mustard0 dataset.
"""

import copy
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import ruamel.yaml

from bundlesdf import BundleSdf, my_cpp
from loftr_wrapper import LoftrRunner


RUN_GPU_INTEGRATION = os.environ.get("RUN_GPU_INTEGRATION") == "1"
REPO_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_DIR = REPO_ROOT / "datasets" / "YCBInEOAT" / "mustard0"
# run_custom.py uses this complete tracking config for custom RGB-D input.
TRACK_CONFIG = REPO_ROOT / "BundleTrack" / "config_ho3d.yml"


class LoFTRMustNotRun:
    def predict(self, rgbAs, rgbBs):
        del rgbAs, rgbBs
        raise AssertionError("cached raw matches must bypass LoFTR")


@unittest.skipUnless(RUN_GPU_INTEGRATION, "set RUN_GPU_INTEGRATION=1")
class MatchingGpuIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loftr = LoftrRunner()
        rgb_names = {path.stem for path in (SEQUENCE_DIR / "rgb").glob("*.png")}
        depth_names = {path.stem for path in (SEQUENCE_DIR / "depth").glob("*.png")}
        mask_names = {path.stem for path in (SEQUENCE_DIR / "masks_sam2").glob("*.png")}
        cls.frame_names = sorted(rgb_names & depth_names & mask_names)[:2]
        if len(cls.frame_names) != 2:
            raise unittest.SkipTest("two aligned RGB/depth/SAM2 frames are required")
        cls.K = np.loadtxt(SEQUENCE_DIR / "cam_K.txt").reshape(3, 3).astype(np.float32)

    def make_tracker(self, min_match_with_ref):
        yaml_loader = ruamel.yaml.YAML()
        with TRACK_CONFIG.open() as config_file:
            config = yaml_loader.load(config_file)

        temp_dir = tempfile.TemporaryDirectory(prefix="bundlesam3dgs-test-")
        config = copy.deepcopy(config)
        config["debug_dir"] = temp_dir.name
        config["SPDLOG"] = 0
        config["feature_corres"]["min_match_with_ref"] = min_match_with_ref
        config_path = Path(temp_dir.name) / "track.yml"
        with config_path.open("w") as config_file:
            yaml_loader.dump(config, config_file)

        tracker = BundleSdf.__new__(BundleSdf)
        tracker.cfg_track = config
        tracker.debug_dir = temp_dir.name
        tracker.bundler = my_cpp.Bundler(my_cpp.YamlLoadFile(str(config_path)))
        tracker.loftr = self.loftr
        return tracker, temp_dir

    def make_frame(self, tracker, index):
        frame_name = self.frame_names[index]
        color = cv2.imread(str(SEQUENCE_DIR / "rgb" / f"{frame_name}.png"))[..., ::-1].copy()
        depth = cv2.imread(
            str(SEQUENCE_DIR / "depth" / f"{frame_name}.png"), cv2.IMREAD_UNCHANGED
        ).astype(np.float32) / 1000.0
        mask = cv2.imread(
            str(SEQUENCE_DIR / "masks_sam2" / f"{frame_name}.png"), cv2.IMREAD_UNCHANGED
        )
        if mask.ndim == 3:
            mask = (mask.sum(axis=-1) > 0).astype(np.uint8)
        tracker.cnt = index
        return tracker.make_frame(color, depth, self.K, frame_name, mask=mask)

    def test_all_reference_candidates_fail_only_after_matching(self):
        tracker, temp_dir = self.make_tracker(min_match_with_ref=1_000_000)
        self.addCleanup(temp_dir.cleanup)
        first_frame = self.make_frame(tracker, 0)
        second_frame = self.make_frame(tracker, 1)

        tracker.process_new_frame(first_frame)
        tracker.process_new_frame(second_frame)

        self.assertEqual(second_frame._status, my_cpp.Frame.FAIL)
        self.assertNotIn(second_frame._id, tracker.bundler._frames)
        self.assertIn(first_frame._id, tracker.bundler._frames)

    def test_two_frames_reach_native_optimizer(self):
        tracker, temp_dir = self.make_tracker(min_match_with_ref=5)
        self.addCleanup(temp_dir.cleanup)
        first_frame = self.make_frame(tracker, 0)
        second_frame = self.make_frame(tracker, 1)

        tracker.process_new_frame(first_frame)
        tracker.process_new_frame(second_frame)

        self.assertNotEqual(second_frame._status, my_cpp.Frame.FAIL)
        self.assertIn(second_frame._id, tracker.bundler._frames)

        pair = (second_frame, first_frame)
        raw_matches = np.array(tracker.bundler._fm._raw_matches[pair]).copy()
        del tracker.bundler._fm._matches[pair]
        tracker.loftr = LoFTRMustNotRun()

        rebuilt_counts = tracker.find_corres([pair])

        self.assertGreater(rebuilt_counts[0], 0)
        self.assertIn(pair, tracker.bundler._fm._matches)
        np.testing.assert_array_equal(
            np.array(tracker.bundler._fm._raw_matches[pair]), raw_matches
        )


if __name__ == "__main__":
    unittest.main()
