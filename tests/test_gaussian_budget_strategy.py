"""Regressions for percentile topology operator selection."""

import unittest

import torch

from gaussian_budget_strategy import (
    TOPOLOGY_MODES,
    select_budgeted_topology,
    split_child_opacity,
)


class GaussianBudgetSelectionTest(unittest.TestCase):
    def test_operator_modes_share_three_percent_candidates(self):
        n_gaussians = 100
        grad2d = torch.arange(n_gaussians, dtype=torch.float32)
        count = torch.ones(n_gaussians)
        log_scales = torch.zeros(n_gaussians, 3)
        log_scales[98] = -2.0
        log_scales[99] = 1.0
        opacity = torch.linspace(0.01, 0.99, n_gaussians)
        protected = torch.zeros(n_gaussians, dtype=torch.bool)
        protected[90:] = True

        selections = {
            mode: select_budgeted_topology(
                grad2d,
                count,
                log_scales,
                torch.logit(opacity),
                protected,
                fraction=0.03,
                mode=mode,
            )
            for mode in TOPOLOGY_MODES
        }

        expected_grow = {97, 98, 99}
        expected_prune = [0, 1, 2]
        for selection in selections.values():
            self.assertEqual(selection.max_replacements, 3)
            self.assertEqual(set(selection.grow.tolist()), expected_grow)
            self.assertEqual(selection.prune.tolist(), expected_prune)
            self.assertTrue(expected_grow.isdisjoint(selection.prune.tolist()))
            self.assertTrue(all(index < 90 for index in selection.prune.tolist()))

        mixed = selections["mixed"]
        self.assertEqual(set(mixed.duplicate.tolist()), {97, 98})
        self.assertEqual(mixed.split.tolist(), [99])
        self.assertEqual(
            set(selections["duplicate_only"].duplicate.tolist()), expected_grow
        )
        self.assertEqual(selections["duplicate_only"].split.numel(), 0)
        self.assertEqual(selections["split_only"].duplicate.numel(), 0)
        self.assertEqual(set(selections["split_only"].split.tolist()), expected_grow)
        self.assertEqual(selections["prune_only"].duplicate.numel(), 0)
        self.assertEqual(selections["prune_only"].split.numel(), 0)

    def test_invalid_operator_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            select_budgeted_topology(
                torch.ones(4),
                torch.ones(4),
                torch.zeros(4, 3),
                torch.zeros(4),
                torch.zeros(4, dtype=torch.bool),
                fraction=0.25,
                mode="invalid",
            )

    def test_exact_rank_selection_and_append_protection(self):
        n_gaussians = 100
        grad2d = torch.arange(n_gaussians, dtype=torch.float32)
        count = torch.ones(n_gaussians)
        log_scales = torch.zeros(n_gaussians, 3)
        log_scales[98] = -2.0
        log_scales[99] = 1.0
        opacity = torch.linspace(0.01, 0.99, n_gaussians)
        protected = torch.zeros(n_gaussians, dtype=torch.bool)
        protected[90:] = True

        selected = select_budgeted_topology(
            grad2d,
            count,
            log_scales,
            torch.logit(opacity),
            protected,
            fraction=0.02,
        )

        self.assertEqual(selected.max_replacements, 2)
        self.assertEqual(selected.duplicate.tolist(), [98])
        self.assertEqual(selected.split.tolist(), [99])
        self.assertEqual(selected.prune.tolist(), [0, 1])
        self.assertTrue(all(index < 90 for index in selected.prune.tolist()))

    def test_unobserved_gaussians_are_not_ranked(self):
        n_gaussians = 100
        grad2d = torch.arange(n_gaussians, dtype=torch.float32)
        count = torch.ones(n_gaussians)
        count[99] = 0.0
        grad2d[99] = 1e9
        selected = select_budgeted_topology(
            grad2d,
            count,
            torch.zeros(n_gaussians, 3),
            torch.linspace(-4.0, 4.0, n_gaussians),
            torch.zeros(n_gaussians, dtype=torch.bool),
            fraction=0.02,
        )
        grow = set(selected.duplicate.tolist() + selected.split.tolist())
        self.assertNotIn(99, grow)
        self.assertNotIn(99, selected.prune.tolist())
        self.assertEqual(grow, {97, 98})

    def test_candidate_shortage_reduces_both_sides(self):
        n_gaussians = 100
        count = torch.zeros(n_gaussians)
        count[:4] = 1.0
        protected = torch.zeros(n_gaussians, dtype=torch.bool)
        protected[:2] = True
        selected = select_budgeted_topology(
            torch.arange(n_gaussians, dtype=torch.float32),
            count,
            torch.zeros(n_gaussians, 3),
            torch.linspace(-5.0, 5.0, n_gaussians),
            protected,
            fraction=0.10,
        )
        self.assertEqual(selected.replacement_count, 1)
        self.assertEqual(
            selected.duplicate.numel() + selected.split.numel(),
            selected.prune.numel(),
        )

    def test_two_child_opacity_preserves_composite(self):
        parent = torch.tensor([0.001, 0.1, 0.6, 0.99], dtype=torch.float64)
        child = split_child_opacity(parent)
        composite = 1.0 - (1.0 - child) ** 2
        torch.testing.assert_close(composite, parent, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.isfinite(torch.logit(child)).all())


if __name__ == "__main__":
    unittest.main()
