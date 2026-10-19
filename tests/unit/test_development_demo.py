"""Artificial chronology, replay and accounting fixtures; no market evaluation."""
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wharton_lab.research.development_demo import (
    ASSETS, _fit, build_design, evaluate_development, observations_from_returns,
    rank_weights, run_development_demo, simulate_portfolio_path, synthetic_returns,
)


class DevelopmentForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations = observations_from_returns(synthetic_returns())
        cls.design = build_design(cls.observations)
        cls.evaluation = evaluate_development(cls.design)

    def test_future_labels_and_features_cannot_change_first_decision_forecasts(self):
        first_sample = self.evaluation["folds"][0]["test_start"]
        cutoff = self.design.decisions[first_sample]
        modified = tuple(replace(r, value=r.value + 0.2) if r.event_time >= cutoff else r
                         for r in self.observations)
        changed = evaluate_development(build_design(modified))
        before = {(r["asset"], r["model"]): r["forecast"] for r in self.evaluation["forecasts"] if r["sample"] == first_sample}
        after = {(r["asset"], r["model"]): r["forecast"] for r in changed["forecasts"] if r["sample"] == first_sample}
        self.assertEqual(before, after)

    def test_nested_maturity_and_train_only_scaler_receipts(self):
        for attempt in self.evaluation["selection_attempts"]:
            self.assertTrue(attempt["train"]["all_labels_mature"])
            self.assertTrue(attempt["validation"]["all_labels_mature"])
            self.assertLess(attempt["train"]["stop_exclusive"], attempt["validation"]["start"])
            self.assertLessEqual(attempt["validation"]["stop_exclusive"],
                                 self.evaluation["folds"][attempt["outer_fold"]]["train"]["stop_exclusive"])
        states = [p for p in self.evaluation["model_parameters"] if p["stage"] == "outer_final"]
        for state, fold in zip(states, self.evaluation["folds"]):
            expected = self.design.X[:fold["train"]["stop_exclusive"]].reshape(-1, 8).mean(axis=0)
            np.testing.assert_allclose(state["scaler_mean"], expected, atol=1e-14)

    def test_immature_label_rejected_before_outer_fit(self):
        times = list(self.design.maturity)
        first = self.evaluation["folds"][0]
        times[first["train"]["stop_exclusive"] - 1] = self.design.decisions[first["test_start"]] + timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "not mature"):
            evaluate_development(replace(self.design, maturity=tuple(times)))

    def test_saved_scaler_and_coefficients_replay_prediction(self):
        state = next(p for p in self.evaluation["model_parameters"] if p["stage"] == "outer_final")
        sample = self.evaluation["folds"][0]["test_start"]
        replay = ((self.design.X[sample] - state["scaler_mean"]) / state["scaler_scale"]) @ np.asarray(state["coefficients"]) + state["intercept"]
        expected = [next(r["forecast"] for r in self.evaluation["forecasts"]
                         if r["sample"] == sample and r["model"] == "ridge" and r["asset"] == a) for a in ASSETS]
        np.testing.assert_allclose(replay, expected, rtol=1e-12, atol=1e-12)


class DevelopmentLedgerTests(unittest.TestCase):
    def test_zero_returns_entry_cost_once_and_second_trade_zero(self):
        weights = np.tile([0.4, 0.4, 0.0], (2, 1))
        rows = simulate_portfolio_path(np.zeros((2, 3)), weights)
        expected = 1 / (1 + 0.001 * 0.8)
        self.assertAlmostEqual(rows[0]["wealth_after"], expected, places=12)
        self.assertAlmostEqual(rows[0]["total_cost"], 1 - expected, places=12)
        self.assertAlmostEqual(rows[1]["wealth_after"], expected, places=12)
        self.assertAlmostEqual(rows[1]["gross_traded_notional"], 0, places=12)
        self.assertAlmostEqual(rows[1]["total_cost"], 0, places=12)

    def test_next_session_return_applied_at_same_row_and_drift_carried(self):
        returns = np.array([[0.1, 0, 0], [0, 0, 0]])
        rows = simulate_portfolio_path(returns, np.tile([0.4, 0.4, 0], (2, 1)), fee=0)
        self.assertAlmostEqual(rows[0]["wealth_after"], 1.04)
        self.assertAlmostEqual(rows[1]["wealth_before"], 1.04)
        self.assertAlmostEqual(rows[1]["position_before_ASSET_A"], 0.44)
        self.assertAlmostEqual(rows[1]["gross_traded_notional"], 0.04)
        self.assertAlmostEqual(rows[1]["wealth_after"], 1.04)

    def test_negative_forecasts_and_ties_use_fixed_rank_rule(self):
        weights = rank_weights([[-0.1, -0.2, -0.3], [0, 0, 0], [-3, -1, -2]])
        np.testing.assert_array_equal(weights, [[0.4, 0.4, 0], [0.4, 0.4, 0], [0, 0.4, 0.4]])

    def test_existing_nonempty_output_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retained.txt"
            path.write_text("retain these exact bytes")
            with self.assertRaises(FileExistsError):
                run_development_demo(tmp)
            self.assertEqual(path.read_text(), "retain these exact bytes")


class DevelopmentReceiptTests(unittest.TestCase):
    def test_late_fit_failure_retains_completed_attempts_and_model_states(self):
        calls = 0

        def fail_third_fit(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected late fit failure")
            return _fit(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed_development"
            with patch("wharton_lab.research.development_demo._fit", side_effect=fail_third_fit):
                with self.assertRaisesRegex(RuntimeError, "injected late fit failure"):
                    run_development_demo(output)
            attempts = [json.loads(line) for line in
                        (output / "selection_attempts.journal.jsonl").read_text().splitlines()]
            states = [json.loads(line) for line in
                      (output / "model_parameters.journal.jsonl").read_text().splitlines()]
            completed = [row for row in attempts if row["status"] == "completed"]
            failed = [row for row in attempts if row["status"] == "failed"]
            self.assertEqual(len(completed), 2)
            self.assertEqual(len(states), 2)
            self.assertEqual(len(failed), 1)
            self.assertEqual({row["model_id"] for row in completed},
                             {row["model_id"] for row in states})
            self.assertTrue(all("coefficients" in row and "scaler_mean" in row for row in states))
            self.assertTrue(all("mse" in row for row in completed))
            terminal = json.loads((output / "terminal_receipt.json").read_text())
            self.assertEqual(terminal["status"], "failed")
            self.assertEqual(terminal["exit_code"], 1)
            self.assertEqual(terminal["error_type"], "RuntimeError")
            self.assertEqual(terminal["stage"], "evaluation")
            self.assertEqual(terminal["attempt_counts"],
                             {"started": 3, "completed": 2, "failed": 1, "model_states": 2})
            self.assertTrue(terminal["source_sha256"])
            self.assertEqual(len(terminal["config_sha256"]), 64)
            self.assertEqual(len(terminal["data_sha256"]), 64)
            self.assertIn("process_argv", terminal)
            self.assertTrue((output / "execution_context.json").is_file())
            self.assertTrue((output / "synthetic_returns.csv").is_file())
            self.assertFalse((output / "numerical_results.json").exists())


if __name__ == "__main__":
    unittest.main()
