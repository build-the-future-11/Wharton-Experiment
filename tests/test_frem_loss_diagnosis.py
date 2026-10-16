"""Small arithmetic and pairing checks; no scientific stack imports or fits."""

import math
import unittest

from scripts.diagnose_frem_loss import pair_rows, shapley_reductions, weighted


class LossDiagnosisTests(unittest.TestCase):
    def test_additive_decomposition_recovers_known_effects(self):
        effects = [0.7, -0.2, 0.1]
        scores = {
            mask: 5 - sum(value for i, value in enumerate(effects) if mask & (1 << i)) for mask in range(8)
        }
        for actual, expected in zip(shapley_reductions(scores, 3), effects, strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_interaction_is_shared_across_replacement_orders(self):
        self.assertEqual(shapley_reductions({0: 1, 1: 1, 2: 1, 3: 0}, 2), [0.5, 0.5])

    def test_incomplete_decomposition_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            shapley_reductions({0: 1, 1: 0}, 2)

    def test_weighting_does_not_overweight_dense_dates(self):
        rows = [
            {"time": "0", "asset": "A"},
            {"time": "0", "asset": "B"},
            {"time": "86400000000000", "asset": "A"},
        ]
        self.assertTrue(math.isclose(weighted(rows, [0, 2, 10]), 5.5))

    def test_pairing_aligns_by_key_and_rejects_outcome_changes(self):
        def row(t, value):
            return {
                "time": str(t),
                "asset": "A",
                "actual": str(value),
                "entry_time": "2",
                "exit_time": "3",
                "label_time": "4",
                "asset_id": "0",
            }

        rows = [row(0, 1), row(1, 2)]
        self.assertEqual(pair_rows(rows, rows[::-1]), rows)
        with self.assertRaisesRegex(ValueError, "outcomes"):
            pair_rows(rows, [row(0, 3), rows[1]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pair_rows(rows, [rows[0], rows[0]])


if __name__ == "__main__":
    unittest.main()
