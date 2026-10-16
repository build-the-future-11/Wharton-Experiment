"""Hand-verifiable accounting, causality, completeness and mandate contracts."""

import copy
import math
import tempfile
import unittest
from pathlib import Path

from successors.frem_portfolio_v1.engine import (
    Forecast,
    backtest,
    ensemble,
    external_forecasts,
    laura_path,
    paired_interval,
    rebalance,
    relative_wealth,
    target_weights,
)
from successors.frem_portfolio_v1.evidence import assert_paired, periods, read_predictions, sha256, write


def fixture(returns=(0.1, -0.1)):
    return [
        {
            "time": i,
            "end": i + 1,
            "currency": "USD",
            "forecasts": [Forecast("A", 0.0, 0.04), Forecast("B", 0.0, 0.04)],
            "returns": {"A": value, "B": value},
        }
        for i, value in enumerate(returns)
    ]


class AccountingTests(unittest.TestCase):
    def test_one_asset_round_trip_exact_cost_basis(self):
        ps = [
            {
                "time": 0,
                "end": 1,
                "currency": "USD",
                "forecasts": [Forecast("A", 0, 0.04)],
                "returns": {"A": 0.1},
            }
        ]
        result = backtest(ps, policy="equal_weight", cost_bps=100, initial_wealth=100)
        self.assertAlmostEqual(result["terminal_wealth"], 100 / 1.01 * 1.1 * 0.99)

    def test_sell_and_buy_self_finance(self):
        holdings, cash, trade = rebalance({"A": 80.0, "B": 0}, 20, {"A": 0, "B": 1}, 0.01)
        expected = (100 - 0.8) / 1.01
        self.assertAlmostEqual(holdings["B"], expected)
        self.assertAlmostEqual(cash + sum(holdings.values()) + trade["fee"], 100)

    def test_drifted_holdings_not_reset(self):
        ps = fixture((0, 0))
        ps[0]["returns"] = {"A": 1.0, "B": 0.0}
        result = backtest(ps, policy="equal_weight", cost_bps=0, initial_wealth=100)
        self.assertEqual(result["periods"][0]["holdings"], {"A": 100, "B": 50})
        self.assertAlmostEqual(result["periods"][1]["notional"], 50)

    def test_flat_zero_cost_is_counterexample_to_six_percent(self):
        ps = fixture((0,) * 12)
        a, b = backtest(ps, cost_bps=0), backtest(ps, policy="equal_weight", cost_bps=0)
        self.assertEqual(relative_wealth(a["terminal_wealth"], b["terminal_wealth"]), 0)

    def test_default_retained_and_not_rebought(self):
        ps = fixture((0, 0))
        ps[0]["returns"]["A"] = -1
        result = backtest(ps, policy="equal_weight", cost_bps=0, initial_wealth=100)
        self.assertEqual(result["terminal_wealth"], 50)
        self.assertEqual(result["defaults"], ["A"])
        self.assertEqual(result["periods"][1]["holdings"]["A"], 0)

    def test_ruin_and_undefined_relative_denominator(self):
        result = backtest(fixture((-1, 0)), policy="equal_weight", cost_bps=0)
        self.assertEqual(result["terminal_wealth"], 0)
        self.assertEqual(result["max_drawdown"], 1)
        with self.assertRaises(ValueError):
            relative_wealth(0, 0)

    def test_cash_no_trading_fees(self):
        result = backtest(fixture(), policy="cash", cost_bps=50)
        self.assertEqual(result["terminal_wealth"], 100000)
        self.assertEqual(result["fees"], 0)

    def test_future_outcomes_do_not_change_first_weights(self):
        ps = fixture()
        altered = copy.deepcopy(ps)
        altered[0]["returns"] = {"A": -0.99, "B": 10}
        a = backtest(ps)["periods"][0]["weights"]
        b = backtest(altered)["periods"][0]["weights"]
        self.assertEqual(a, b)

    def test_missing_return_is_not_zero_imputed(self):
        ps = fixture()
        del ps[1]["returns"]["B"]
        with self.assertRaises(ValueError):
            backtest(ps)

    def test_mixed_currency_rejected(self):
        ps = fixture()
        ps[1]["currency"] = "JPY"
        with self.assertRaises(ValueError):
            backtest(ps)

    def test_overlap_and_unavailable_forecast(self):
        for field, value in (("time", 0), ("available_at", 2)):
            ps = fixture()
            ps[1][field] = value
            with self.assertRaises(ValueError):
                backtest(ps)

    def test_cost_and_leverage_invalid(self):
        for fee in (-0.1, 1, math.nan):
            with self.assertRaises(ValueError):
                rebalance({"A": 0}, 100, {"A": 1}, fee)
        with self.assertRaises(ValueError):
            rebalance({"A": 0}, 100, {"A": 1.01}, 0)

    def test_return_below_total_loss_rejected(self):
        with self.assertRaises(ValueError):
            backtest(fixture((-1.01,)))

    def test_no_trade_band_accounts_for_drift(self):
        holdings, cash, receipt = rebalance({"A": 50}, 50, {"A": 0.501}, 0.01, no_trade_band=0.02)
        self.assertEqual((holdings, cash, receipt["fee"]), ({"A": 50}, 50, 0))


class SignalTests(unittest.TestCase):
    def test_student_scale_is_not_standard_deviation(self):
        f = Forecast.student("A", 0, 2, 5)
        self.assertAlmostEqual(f.variance, 20 / 3)

    def test_invalid_distribution_rejected(self):
        for scale, df in ((0, 5), (1, 2), (math.inf, 5), (1, math.nan)):
            with self.assertRaises(ValueError):
                Forecast.student("A", 0, scale, df)

    def test_negative_signal_falls_to_anchor_cash(self):
        result = target_weights([Forecast("A", -0.1, 0.04), Forecast("B", -0.1, 0.04)])
        self.assertEqual(result, {"A": 0.4, "B": 0.4})

    def test_signal_cap_and_no_leverage(self):
        weights = target_weights([Forecast(str(i), 100, 0.0001) for i in range(12)])
        self.assertLessEqual(max(weights.values()), 0.15)
        self.assertLessEqual(sum(weights.values()), 1 + 1e-12)

    def test_ensemble_includes_model_disagreement(self):
        value = ensemble([[Forecast("A", -1, 1)], [Forecast("A", 1, 1)]])[0]
        self.assertEqual((value.location, value.variance), (0, 2))

    def test_ensemble_missing_assets_rejected(self):
        with self.assertRaises(ValueError):
            ensemble([[Forecast("A", 0, 1)], [Forecast("B", 0, 1)]])

    def test_bootstrap_identity_and_short_block(self):
        self.assertEqual(paired_interval([1] * 12, [1] * 12, repetitions=100)["interval"], [0, 0])
        self.assertEqual(paired_interval([1] * 10, [1] * 10, block=6)["status"], "INSUFFICIENT_DATES")


class FundingTests(unittest.TestCase):
    def test_zero_growth_obligation_shortfall(self):
        value = laura_path([0] * 6)
        self.assertEqual(value["wealth_before_2033_payment"], 450000)
        self.assertEqual(value["reserve_annuity_due"], 500000)
        self.assertEqual(value["reserve_shortfall"], 50000)
        self.assertIsNone(value["facility_capacity"])
        self.assertIsNone(value["funding_probability"])

    def test_contribution_timing_and_annuity_due(self):
        value = laura_path([0.04] * 6, reserve_discount=0.02, flexibility_buffer=25000)
        self.assertAlmostEqual(value["wealth_before_2033_payment"], 300000 * 1.04**6 + 150000 * 1.04**5)
        self.assertAlmostEqual(value["reserve_annuity_due"], 50000 * (1 - 1.02**-10) / 0.02 * 1.02)
        self.assertAlmostEqual(value["facility_capacity"], value["surplus_before_buffer"] - 25000)

    def test_sequence_risk_not_geometric_average_shortcut(self):
        first = laura_path([-0.3, 0.3, 0, 0, 0, 0])["wealth_before_2033_payment"]
        second = laura_path([0.3, -0.3, 0, 0, 0, 0])["wealth_before_2033_payment"]
        self.assertNotEqual(first, second)


class ExternalContractTests(unittest.TestCase):
    def payload(self):
        return {
            "currency": "USD",
            "horizon": "1M",
            "source_sha256": "a" * 64,
            "trained_through": "2024-01-01T00:00:00Z",
            "max_input_available_at": "2024-02-01T00:00:00Z",
            "generated_at": "2024-02-01T00:00:00Z",
            "forecasts": [{"asset": "A", "location": 0, "scale": 0.1, "df": 5}],
        }

    def parse(self, data):
        return external_forecasts(
            data, decision_time="2024-02-02T00:00:00Z", assets=["A"], currency="USD", horizon="1M"
        )

    def test_valid_framework_forecast(self):
        self.assertEqual(self.parse(self.payload())[0].asset, "A")

    def test_future_information_rejected(self):
        for key in ("trained_through", "max_input_available_at", "generated_at"):
            data = self.payload()
            data[key] = "2025-01-01T00:00:00Z"
            with self.assertRaises(ValueError):
                self.parse(data)

    def test_unknown_horizon_and_missing_hash(self):
        for key, val in (("horizon", "1D"), ("source_sha256", ""), ("currency", "JPY")):
            data = self.payload()
            data[key] = val
            with self.assertRaises(ValueError):
                self.parse(data)


class EvidenceTests(unittest.TestCase):
    def test_nonoverwrite_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            write(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                write(path, {"ok": False})

    def test_pairing_checks_outcomes(self):
        a, b = fixture(), fixture()
        b[0]["returns"]["A"] = 0
        with self.assertRaises(ValueError):
            assert_paired(a, b)

    def test_missing_grid_not_silently_removed(self):
        rows = [
            {
                "time": 0,
                "end": 1,
                "available_at": 0,
                "asset": "A",
                "forecast": Forecast("A", 0, 1),
                "return": 0,
            }
        ]
        with self.assertRaises(ValueError):
            periods(rows, "USD", ["A", "B"])

    def test_prediction_hash_and_protected_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            path.write_text(
                "time,label_time,asset,prediction,scale,student_df,actual\n1735689600000000000,1738540800000000000,A,0,.1,5,0\n"
            )
            with self.assertRaises(ValueError):
                read_predictions(path, "x" * 64)
            with self.assertRaises(ValueError):
                read_predictions(path, sha256(path))


if __name__ == "__main__":
    unittest.main()
