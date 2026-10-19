"""Optimizer-boundary regressions with injected results, not optimization runs."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from wharton_lab.models.m11.accounting import empirical_cvar, simulate_ledger
from wharton_lab.models.m11.mpc import PortfolioState
from wharton_lab.models.m11.mpc_verified import solve_mpc


class MPCPolicyContractTests(unittest.TestCase):
    def parameters(self, *, third_asset=False, cash_buffer=False):
        if cash_buffer:
            returns = np.array([[[0.5, 0.0]], [[0.0, 0.75]]])
            obligations = np.array([1.24])
        elif third_asset:
            returns = np.array([[[1/3, -0.5, -1.0]], [[-1/3, 0.5, -1.0]]])
            obligations = np.array([1.0])
        else:
            returns = np.array([[[1/3, -0.5]], [[-1/3, 0.5]]])
            obligations = np.array([1.0])
        return dict(
            state=PortfolioState(cash=1.0, holdings=np.zeros(returns.shape[2])),
            scenario_returns=returns, obligations=obligations, horizon=1,
            cvar_alpha=0.95, max_weight=0.6 if third_asset else 0.8,
            tcost_bps=0.0, allow_cash=cash_buffer,
            min_cash_buffer=0.2 if cash_buffer else 0.0,
        )

    def injected_solve(self, weights, *, params=None, success=True):
        params = self.parameters() if params is None else params
        optimizer = SimpleNamespace(x=np.asarray(weights, dtype=float),
                                    success=success, message="injected test result")
        with patch("wharton_lab.models.m11.mpc_verified.minimize", return_value=optimizer):
            result = solve_mpc(**params, solver="slsqp")
        return result, params

    def assert_policy_and_score(self, result, params):
        weights = result.full_horizon_weights
        self.assertTrue(result.success)
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreaterEqual(float(weights.min()), 0.0)
        self.assertLessEqual(float(weights.max()), params["max_weight"])
        self.assertLessEqual(float(weights.sum(axis=1).max()),
                             1.0 - params["min_cash_buffer"])
        if not params["allow_cash"]:
            np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(result.first_action_weights, weights[0])
        ledger = simulate_ledger(
            weights, params["scenario_returns"], params["state"].cash,
            params["state"].holdings, params["obligations"],
            params["tcost_bps"] * 1e-4,
        )
        initial = params["state"].cash + params["state"].holdings.sum()
        exact_score = (empirical_cvar(ledger["unpaid"].sum(axis=1) / initial,
                                     params["cvar_alpha"])
                       + 1e-4 * float(np.square(weights).mean()))
        self.assertAlmostEqual(result.objective, exact_score, places=14)

    def test_selected_overbudget_candidate_is_repaired_and_scored_as_returned(self):
        # Its normalized allocation beats every finite-bank policy, so this
        # exercises the SLSQP result rather than accidentally checking a bank row.
        result, params = self.injected_solve([0.60000003, 0.40000002])
        self.assertEqual(result.method, "slsqp")
        self.assertTrue(result.optimizer_success)
        self.assert_policy_and_score(result, params)

    def test_fully_invested_small_deficit_is_filled_without_exceeding_asset_cap(self):
        # Multiplying this row by 1 / row.sum() would breach the 0.6 cap.
        result, params = self.injected_solve(
            [0.6, 0.39999995, 0.0], params=self.parameters(third_asset=True))
        self.assertEqual(result.method, "slsqp")
        self.assertTrue(result.optimizer_success)
        self.assert_policy_and_score(result, params)

    def test_negative_and_above_cap_tolerance_noise_is_removed(self):
        result, params = self.injected_solve(
            [0.600000005, 0.4, -0.000000005], params=self.parameters(third_asset=True))
        self.assertEqual(result.method, "slsqp")
        self.assertTrue(result.optimizer_success)
        self.assert_policy_and_score(result, params)
        self.assertEqual(result.full_horizon_weights[0, 2], 0.0)

    def test_minimum_cash_buffer_applies_to_exact_returned_weights(self):
        result, params = self.injected_solve(
            [0.48000003, 0.32000002], params=self.parameters(cash_buffer=True))
        self.assertEqual(result.method, "slsqp")
        self.assertTrue(result.optimizer_success)
        self.assert_policy_and_score(result, params)

    def test_materially_infeasible_optimizer_output_is_not_projected_into_success(self):
        params = self.parameters()
        bank = solve_mpc(**params, solver="finite_candidates")
        attempted = [
            [0.6002, 0.4],             # material budget excess
            [0.5999994, 0.3999996],     # deficit beyond absolute equality tolerance
            [0.80000002, 0.19999998],   # asset cap beyond tolerance
            [-0.00002, 1.00002],       # material negative holding
            [np.nan, 0.4],
        ]
        for weights in attempted:
            with self.subTest(weights=weights):
                result, _ = self.injected_solve(weights, params=params)
                self.assertFalse(result.optimizer_success)
                self.assertEqual(result.method, "finite_policy_bank")
                np.testing.assert_array_equal(result.full_horizon_weights,
                                              bank.full_horizon_weights)
                self.assertEqual(result.objective, bank.objective)
                self.assert_policy_and_score(result, params)

    def test_valid_fully_invested_optimizer_policy_is_unchanged(self):
        result, params = self.injected_solve([0.6, 0.4])
        self.assertEqual(result.method, "slsqp")
        self.assertTrue(result.optimizer_success)
        np.testing.assert_array_equal(result.full_horizon_weights, [[0.6, 0.4]])
        self.assert_policy_and_score(result, params)

    def test_failed_optimizer_does_not_promote_its_feasible_attempt(self):
        result, params = self.injected_solve([0.6, 0.4], success=False)
        self.assertFalse(result.optimizer_success)
        self.assertEqual(result.method, "finite_policy_bank")
        self.assert_policy_and_score(result, params)

    def test_known_finite_bank_policies_remain_unchanged(self):
        for cash_buffer, expected in [(False, [[0.5, 0.5]]), (True, [[0.4, 0.4]])]:
            with self.subTest(cash_buffer=cash_buffer):
                params = self.parameters(cash_buffer=cash_buffer)
                with patch("wharton_lab.models.m11.mpc_verified.minimize",
                           side_effect=AssertionError("finite bank called optimizer")):
                    result = solve_mpc(**params, solver="finite_candidates")
                self.assertEqual(result.method, "finite_policy_bank")
                self.assertFalse(result.optimizer_success)
                np.testing.assert_array_equal(result.full_horizon_weights, expected)
                self.assert_policy_and_score(result, params)


if __name__ == "__main__":
    unittest.main()
