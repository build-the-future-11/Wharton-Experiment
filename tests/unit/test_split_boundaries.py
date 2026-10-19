"""Deterministic split-contract regressions; no data downloads or experiments."""

import unittest
from unittest.mock import patch

import numpy as np

from wharton_lab.splits.walk_forward import iter_fold_arrays, walk_forward_folds


class SplitBoundaryTests(unittest.TestCase):
    def assert_rejected_before_construction(self, **changes):
        kwargs = dict(n_samples=100, min_train=20, test_size=10,
                      step=None, purge_gap=0, lockbox_size=0)
        kwargs.update(changes)
        # On the old implementation a negative step could loop indefinitely.
        # Guard fold construction so the before-fix regression fails promptly.
        with patch("wharton_lab.splits.walk_forward.WalkForwardSplit",
                   side_effect=AssertionError("invalid parameters reached fold construction")):
            with self.assertRaises(ValueError):
                walk_forward_folds(**kwargs)

    def test_invalid_sizes_and_steps_fail_before_iteration(self):
        invalid = {
            "n_samples": [0, -1, True, 100.5, "100"],
            "min_train": [0, -1, False, 20.5],
            "test_size": [0, -1, True, 10.5],
            "step": [0, -1, True, 1.5],
            "purge_gap": [-1, True, 0.5],
            "lockbox_size": [-1, True, 1.5, 101],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(parameter=name, value=value):
                    self.assert_rejected_before_construction(**{name: value})

    def test_oversized_lockbox_cannot_create_negative_training_slice(self):
        # Previously returned train [0:97] and test [2:100], overlapping
        # rows 2..96, because the train slice stop was -3.
        self.assert_rejected_before_construction(purge_gap=5, lockbox_size=98)

    def test_prelockbox_training_must_meet_minimum_after_purge(self):
        self.assert_rejected_before_construction(purge_gap=5, lockbox_size=76)

    def test_regular_folds_and_lockbox_have_disjoint_numeric_boundaries(self):
        folds = walk_forward_folds(200, min_train=50, test_size=10,
                                   purge_gap=2, lockbox_size=20)
        self.assertEqual(folds[0].train_indices, slice(0, 50))
        self.assertEqual(folds[0].test_indices, slice(52, 62))
        self.assertEqual(folds[-1].train_indices, slice(0, 178))
        self.assertEqual(folds[-1].test_indices, slice(180, 200))
        self.assertTrue(folds[-1].is_lockbox)
        values = np.arange(200)
        for expected_id, (fold, Xtr, ytr, Xte, yte) in enumerate(
                iter_fold_arrays(values, values, folds)):
            self.assertEqual(fold.fold_id, expected_id)
            self.assertGreaterEqual(len(Xtr), 50)
            self.assertEqual(fold.train_indices.stop + 2, fold.test_indices.start)
            self.assertFalse(np.intersect1d(Xtr, Xte).size)
            np.testing.assert_array_equal(Xtr, ytr)
            np.testing.assert_array_equal(Xte, yte)
            if not fold.is_lockbox:
                self.assertLessEqual(fold.test_indices.stop, 180)

    def test_lockbox_only_boundary_is_explicitly_supported(self):
        folds = walk_forward_folds(25, min_train=20, test_size=10,
                                   purge_gap=2, lockbox_size=3)
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].train_indices, slice(0, 20))
        self.assertEqual(folds[0].test_indices, slice(22, 25))
        self.assertEqual(folds[0].fold_id, 0)
        self.assertTrue(folds[0].is_lockbox)

    def test_without_lockbox_an_ordinary_test_window_must_fit(self):
        with self.assertRaises(ValueError):
            walk_forward_folds(25, min_train=20, test_size=10, purge_gap=2)

    def test_numpy_integer_sizes_and_smaller_positive_step_remain_valid(self):
        folds = walk_forward_folds(np.int64(40), min_train=np.int64(20),
                                   test_size=np.int64(10), step=np.int64(5))
        self.assertEqual([f.test_indices for f in folds],
                         [slice(20, 30), slice(25, 35), slice(30, 40)])
        for f in folds:
            self.assertLessEqual(f.train_indices.stop, f.test_indices.start)


if __name__ == "__main__":
    unittest.main()
