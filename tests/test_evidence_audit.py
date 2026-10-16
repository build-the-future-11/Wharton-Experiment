"""Small stdlib-only adversarial checks; no model or scientific-stack imports."""

import math
import unittest

from scripts.audit_engineering_evidence import balanced_mean, nll, score_rows


class EvidenceAuditTests(unittest.TestCase):
    def row(self):
        return {"model": "frem", "seed": "11", "time": "100", "entry_time": "110", "exit_time": "120",
                "label_time": "130", "asset": "A", "asset_id": "0", "actual": "0", "prediction": "0",
                "scale": "1", "student_df": "1"}

    def test_cauchy_density_at_origin(self):
        self.assertAlmostEqual(nll(0, 0, 1, 1), math.log(math.pi))

    def test_scale_jacobian(self):
        self.assertAlmostEqual(nll(0, 0, 2, 1) - nll(0, 0, 1, 1), math.log(2))

    def test_unequal_asset_observation_counts(self):
        rows = [("day1", "A", 0), ("day1", "A", 0), ("day1", "B", 8), ("day2", "A", 2)]
        self.assertEqual(balanced_mean(rows), 3)

    def test_invalid_distribution_rejected(self):
        for values in [(0, 0, 0, 1), (0, 0, 1, 0), (float("nan"), 0, 1, 1)]:
            with self.assertRaises(ValueError):
                nll(*values)

    def test_protected_or_unmatured_observations_rejected(self):
        for column, value in [("time", "90"), ("entry_time", "100"),
                              ("label_time", "119"), ("label_time", "140")]:
            row = self.row()
            row[column] = value
            with self.assertRaisesRegex(ValueError, "Timing"):
                score_rows([row], "frem", 11, 90, 140)

    def test_duplicate_prediction_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            score_rows([self.row(), self.row()], "frem", 11, 90, 140)

    def test_wrong_model_or_seed_rejected(self):
        for model, seed in [("ridge", 11), ("frem", 29)]:
            with self.assertRaisesRegex(ValueError, "model/seed"):
                score_rows([self.row()], model, seed, 90, 140)

    def test_empty_evidence_rejected(self):
        with self.assertRaises(ValueError):
            balanced_mean([])


if __name__ == "__main__":
    unittest.main()
