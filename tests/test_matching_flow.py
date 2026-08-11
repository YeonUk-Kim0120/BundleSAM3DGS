"""Matching-flow regressions.

Run inside the project container after building ``BundleTrack/build``. The
tests replace LoFTR and the native feature manager with small fakes, so no
checkpoint, dataset, or GPU inference is required.
"""

import unittest

import numpy as np

from bundlesdf import BundleSdf


class FakeFrame:
    pass


class FakeFeatureManager:
    def __init__(self):
        self._matches = {}
        self._raw_matches = {}
        self.converted_pairs = []
        self.ransac_pairs = []

    def getProcessedImagePairs(self, pairs):
        query_pairs = [pair for pair in pairs if pair not in self._raw_matches]
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        images = [image for _ in range(2 * len(query_pairs))]
        transforms = [np.eye(3, dtype=np.float32) for _ in images]
        return images, transforms, query_pairs

    def rawMatchesToCorres(self, pairs):
        self.converted_pairs.extend(pairs)
        for pair in pairs:
            self._matches[pair] = [object() for _ in self._raw_matches[pair]]

    def runRansacMultiPairGPU(self, pairs):
        self.ransac_pairs.extend(pairs)

    def vizCorresBetween(self, _frame_a, _frame_b, _stage):
        pass


class FakeLoFTR:
    def __init__(self, correspondences=None):
        self.correspondences = correspondences or []
        self.calls = 0

    def predict(self, rgbAs, rgbBs):
        del rgbAs, rgbBs
        self.calls += 1
        return self.correspondences


def make_tracker(feature_manager, loftr):
    tracker = BundleSdf.__new__(BundleSdf)
    tracker.bundler = type("FakeBundler", (), {"_fm": feature_manager})()
    tracker.loftr = loftr
    tracker.cfg_track = {
        "ransac": {"num_sample": 3, "min_match_after_ransac": 5}
    }
    return tracker


class MatchingFlowTest(unittest.TestCase):
    def setUp(self):
        self.pair = (FakeFrame(), FakeFrame())

    def test_cached_raw_matches_rebuild_filtered_matches_without_loftr(self):
        feature_manager = FakeFeatureManager()
        feature_manager._raw_matches[self.pair] = np.zeros((6, 4), dtype=np.uint16)
        loftr = FakeLoFTR()
        tracker = make_tracker(feature_manager, loftr)

        counts = tracker.find_corres([self.pair])

        self.assertEqual(counts, [6])
        self.assertEqual(loftr.calls, 0)
        self.assertEqual(feature_manager.converted_pairs, [self.pair])
        self.assertEqual(feature_manager.ransac_pairs, [self.pair])

    def test_empty_candidate_is_cached_with_four_columns_and_not_ransaced(self):
        feature_manager = FakeFeatureManager()
        empty_matches = np.empty((0, 5), dtype=np.float32)
        loftr = FakeLoFTR([empty_matches])
        tracker = make_tracker(feature_manager, loftr)

        counts = tracker.find_corres([self.pair])

        self.assertEqual(counts, [0])
        self.assertEqual(loftr.calls, 1)
        self.assertEqual(feature_manager._raw_matches[self.pair].shape, (0, 4))
        self.assertEqual(feature_manager._raw_matches[self.pair].dtype, np.uint16)
        self.assertEqual(feature_manager.ransac_pairs, [])

    def test_existing_filtered_matches_are_reused(self):
        feature_manager = FakeFeatureManager()
        feature_manager._matches[self.pair] = [object() for _ in range(7)]
        loftr = FakeLoFTR()
        tracker = make_tracker(feature_manager, loftr)

        counts = tracker.find_corres([self.pair])

        self.assertEqual(counts, [7])
        self.assertEqual(loftr.calls, 0)
        self.assertEqual(feature_manager.converted_pairs, [])

    def test_cached_and_new_pairs_are_both_converted(self):
        cached_pair = self.pair
        new_pair = (FakeFrame(), FakeFrame())
        feature_manager = FakeFeatureManager()
        feature_manager._raw_matches[cached_pair] = np.zeros((6, 4), dtype=np.uint16)
        new_matches = np.zeros((7, 5), dtype=np.float32)
        loftr = FakeLoFTR([new_matches])
        tracker = make_tracker(feature_manager, loftr)

        counts = tracker.find_corres([cached_pair, new_pair])

        self.assertEqual(counts, [6, 7])
        self.assertEqual(loftr.calls, 1)
        self.assertEqual(feature_manager.converted_pairs, [cached_pair, new_pair])
        self.assertEqual(feature_manager.ransac_pairs, [cached_pair, new_pair])


if __name__ == "__main__":
    unittest.main()
