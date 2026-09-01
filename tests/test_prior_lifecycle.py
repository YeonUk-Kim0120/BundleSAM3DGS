"""CPU tests for the milestone-③ lifecycle prototype (experiments module).

Run: python3 -m unittest tests.test_prior_lifecycle
"""

import unittest

import torch

from prior_lifecycle import (
    STATE_CONTRADICTED,
    STATE_SUSPECT,
    STATE_UNSEEN,
    STATE_VERIFIED,
    LifecycleFields,
    TransitionThresholds,
    accumulate_support_residuals,
    apply_transitions,
    depth_evidence_masks,
    independent_view_mask,
)


def make_fields(states, lineage=None):
    n = len(states)
    fields = LifecycleFields.create(n, lineage_prior=True)
    fields.state[:] = torch.tensor(states, dtype=torch.int8)
    if lineage is not None:
        fields.lineage[:] = torch.tensor(lineage, dtype=torch.bool)
    return fields


class EvidenceMaskTest(unittest.TestCase):
    def test_four_way_classification(self):
        tol = 0.01
        gaussian_depth = torch.tensor([0.50, 0.45, 0.60, 0.60, 0.60])
        observed_depth = torch.tensor([0.505, 0.50, 0.50, 0.50, 0.50])
        valid = torch.tensor([True, True, True, True, False])
        # front depth: splat 2 occluded by something at 0.5; splat 3 frontmost
        # (its own depth is the front → no separate occluder)
        front = torch.tensor([0.50, 0.45, 0.50, 0.60, 0.50])
        alpha = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        ev = depth_evidence_masks(gaussian_depth, observed_depth, valid,
                                  front, alpha, tol)
        self.assertTrue(bool(ev["support"][0]))          # |Δ|=5mm ≤ 1cm
        self.assertTrue(bool(ev["free_space"][1]))       # 5cm in front
        self.assertTrue(bool(ev["behind_occluded"][2]))  # behind, occluded
        self.assertTrue(bool(ev["behind_miss"][3]))      # behind, frontmost
        for key in ev:
            self.assertFalse(bool(ev[key][4]))           # invalid pixel: nothing

    def test_grazing_view_widens_support_tolerance(self):
        # 2cm along-ray difference: rejected front-on, but at a grazing view
        # (|cos| = 1/3 → tolerance ×3 = 3cm) the same offset is support.
        tol = 0.01
        common = dict(
            gaussian_depth=torch.tensor([0.52, 0.52]),
            observed_depth=torch.tensor([0.50, 0.50]),
            valid_observation=torch.tensor([True, True]),
            geometric_front_depth=torch.tensor([0.52, 0.52]),
            geometric_alpha=torch.tensor([1.0, 1.0]),
            depth_tolerance=tol,
        )
        ev = depth_evidence_masks(
            **common, view_abs_cos=torch.tensor([1.0, 1.0 / 3.0])
        )
        self.assertFalse(bool(ev["support"][0]))
        self.assertTrue(bool(ev["support"][1]))
        # cap: |cos| below 1/cap must not widen further than cap×tol
        ev = depth_evidence_masks(
            **common, view_abs_cos=torch.tensor([0.05, 0.05]),
            grazing_tolerance_cap=1.5,
        )
        self.assertFalse(bool(ev["support"][0]))  # 2cm > 1.5cm cap

    def test_transparent_occluder_does_not_shield(self):
        # geometric alpha below threshold → no occluder → behind_miss.
        tol = 0.01
        ev = depth_evidence_masks(
            torch.tensor([0.60]), torch.tensor([0.50]),
            torch.tensor([True]), torch.tensor([0.50]),
            torch.tensor([0.2]), tol,
        )
        self.assertTrue(bool(ev["behind_miss"][0]))


class IndependentViewTest(unittest.TestCase):
    def test_first_evidence_always_accepted(self):
        accepted = independent_view_mask(
            candidate=torch.tensor([True]),
            conflict_count=torch.tensor([0], dtype=torch.int16),
            last_conflict_view=torch.tensor([[0.0, 0.0, 1.0]]),
            current_view_direction=torch.tensor([[0.0, 0.0, 1.0]]),
            min_view_angle_deg=10.0,
        )
        self.assertTrue(bool(accepted[0]))

    def test_same_view_rejected_then_different_view_accepted(self):
        count = torch.tensor([1, 1], dtype=torch.int16)
        last = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        # 5° (same-ish) and 15° (different) directions
        import math
        d5 = torch.tensor([math.sin(math.radians(5)), 0.0,
                           math.cos(math.radians(5))])
        d15 = torch.tensor([math.sin(math.radians(15)), 0.0,
                            math.cos(math.radians(15))])
        accepted = independent_view_mask(
            candidate=torch.tensor([True, True]),
            conflict_count=count,
            last_conflict_view=last,
            current_view_direction=torch.stack((d5, d15)),
            min_view_angle_deg=10.0,
        )
        self.assertFalse(bool(accepted[0]))
        self.assertTrue(bool(accepted[1]))


class TransitionTest(unittest.TestCase):
    def apply(self, states, conflicts, supported):
        fields = make_fields(states)
        fields.conflict_count[:] = torch.tensor(conflicts, dtype=torch.int16)
        masks = apply_transitions(
            fields, torch.tensor(supported), TransitionThresholds()
        )
        return fields, masks

    def test_unseen_supported_becomes_verified(self):
        fields, _ = self.apply([STATE_UNSEEN], [0], [True])
        self.assertEqual(int(fields.state[0]), STATE_VERIFIED)

    def test_unseen_two_conflicts_contradicted(self):
        fields, _ = self.apply([STATE_UNSEEN], [2], [False])
        self.assertEqual(int(fields.state[0]), STATE_CONTRADICTED)

    def test_unseen_one_conflict_stays_unseen(self):
        fields, _ = self.apply([STATE_UNSEEN], [1], [False])
        self.assertEqual(int(fields.state[0]), STATE_UNSEEN)

    def test_verified_two_conflicts_suspect_three_contradicted(self):
        fields, _ = self.apply([STATE_VERIFIED], [2], [False])
        self.assertEqual(int(fields.state[0]), STATE_SUSPECT)
        fields, _ = self.apply([STATE_VERIFIED], [3], [False])
        self.assertEqual(int(fields.state[0]), STATE_CONTRADICTED)

    def test_suspect_recovers_with_support_and_counter_resets(self):
        fields = make_fields([STATE_SUSPECT])
        fields.conflict_count[:] = 2
        fields.last_conflict_view[:] = torch.tensor([[0.0, 0.0, 1.0]])
        apply_transitions(fields, torch.tensor([True]), TransitionThresholds())
        self.assertEqual(int(fields.state[0]), STATE_VERIFIED)
        self.assertEqual(int(fields.conflict_count[0]), 0)
        self.assertEqual(float(fields.last_conflict_view.abs().sum()), 0.0)

    def test_suspect_third_conflict_contradicted(self):
        fields, _ = self.apply([STATE_SUSPECT], [3], [False])
        self.assertEqual(int(fields.state[0]), STATE_CONTRADICTED)

    def test_contradicted_is_terminal(self):
        fields, _ = self.apply([STATE_CONTRADICTED], [0], [True])
        self.assertEqual(int(fields.state[0]), STATE_CONTRADICTED)

    def test_support_beats_conflict_within_batch(self):
        # Even with enough accumulated conflicts, support in the same batch
        # resets and verifies (sibling: "support in any frame wins").
        fields = make_fields([STATE_UNSEEN])
        fields.conflict_count[:] = 2
        apply_transitions(fields, torch.tensor([True]), TransitionThresholds())
        self.assertEqual(int(fields.state[0]), STATE_VERIFIED)


class FieldOpsTest(unittest.TestCase):
    def test_create_concat_keep_summary(self):
        prior = LifecycleFields.create(3, lineage_prior=True)
        observed = LifecycleFields.create(2, lineage_prior=False)
        both = prior.concat(observed).validated()
        self.assertEqual(len(both), 5)
        summary = both.summary()
        self.assertEqual(summary["unseen"], 3)
        self.assertEqual(summary["verified"], 2)
        self.assertEqual(summary["prior_lineage"], 3)
        kept = both.keep(both.state == STATE_VERIFIED)
        self.assertEqual(len(kept), 2)
        self.assertFalse(bool(kept.lineage.any()))

    def test_validation_rejects_bad_state(self):
        fields = LifecycleFields.create(2, lineage_prior=True)
        fields.state[0] = 7
        with self.assertRaisesRegex(ValueError, "invalid value"):
            fields.validated()


class ResidualAccumulatorTest(unittest.TestCase):
    def test_front_facing_bulge_gives_negative_normal_offset(self):
        # Prior bulges 8mm toward the camera (gaussian in front of surface);
        # outward normal faces the camera (opposite the ray) → residual < 0
        # (move inward), magnitude ≈ 8mm.
        fields = LifecycleFields.create(1, lineage_prior=True)
        accumulate_support_residuals(
            fields,
            support=torch.tensor([True]),
            gaussian_depth=torch.tensor([0.492]),
            observed_depth=torch.tensor([0.500]),
            normals=torch.tensor([[0.0, 0.0, -1.0]]),
            view_directions=torch.tensor([[0.0, 0.0, 1.0]]),
        )
        self.assertEqual(int(fields.residual_count[0]), 1)
        self.assertAlmostEqual(float(fields.residual_sum[0]), -0.008, places=5)

    def test_grazing_observation_skipped(self):
        fields = LifecycleFields.create(1, lineage_prior=True)
        accumulate_support_residuals(
            fields,
            support=torch.tensor([True]),
            gaussian_depth=torch.tensor([0.5]),
            observed_depth=torch.tensor([0.505]),
            normals=torch.tensor([[1.0, 0.0, 0.0]]),  # ⟂ to the ray
            view_directions=torch.tensor([[0.0, 0.0, 1.0]]),
        )
        self.assertEqual(int(fields.residual_count[0]), 0)


if __name__ == "__main__":
    unittest.main()
