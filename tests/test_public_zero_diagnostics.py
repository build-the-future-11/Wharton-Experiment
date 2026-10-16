"""Standard-library tests: no scientific stack or model fitting."""

import unittest

from frem.calibration_diagnostics import calibration_diagnostics
from scripts.diagnose_public_zero import align, decompose


class DiagnosticTests(unittest.TestCase):
    def test_rejects_misaligned_outcomes(self):
        row = {"time": "1", "asset": "a", "actual": "1", "label_time": "2", "asset_id": "0", "target_month": "m"}
        with self.assertRaises(ValueError):
            align([row], [{**row, "actual": "2"}])

    def test_decomposition_identical_distributions(self):
        row = {"time": "1", "asset": "a", "actual": "1", "prediction": "0", "scale": "1", "student_df": "5"}
        self.assertEqual(decompose([row], [row]), {"prediction": 0, "scale": 0, "student_df": 0})

    def test_penalty_allows_memory_win_without_forcing_it(self):
        grid = [{"weight": 0, "validation_nll": 2}, {"weight": 0.25, "validation_nll": 1}]
        result = calibration_diagnostics(grid, [0, 4], [], 100)
        self.assertEqual(result["validation_nll_penalty_by_weight"]["0.25"], -1)
        self.assertIsNone(result["record_age_days_at_validation_start"])
        self.assertEqual(result["retrieval_count_histogram"], {0: 1, 4: 1})

    def test_rejects_unmatured_records(self):
        with self.assertRaises(ValueError):
            calibration_diagnostics(
                [{"weight": 0, "validation_nll": 1}], [1], [{"asset_id": 0, "label_time": 100}], 100
            )


if __name__ == "__main__":
    unittest.main()
