"""Independent numerical checks for closed-evidence diagnostic arithmetic."""

import math
import unittest

from scripts.diagnose_s8_completion import (
    audit_ledger,
    moments_and_pit,
    pearson,
    ranks,
    retrieval_support,
    student_cdf,
)


class SavedS8DiagnosticsTests(unittest.TestCase):
    def test_student_cdf_matches_cauchy_closed_form(self):
        for value in (-100, -3, -1, -0.1, 0, 0.1, 1, 3, 100):
            self.assertAlmostEqual(student_cdf(value, 1), 0.5 + math.atan(value) / math.pi, places=12)

    def test_student_five_cdf_matches_numerical_integral(self):
        # Independent Simpson quadrature of the t5 density over [0, 1.7].
        df, high, count = 5, 1.7, 10000
        step = high / count
        normalizer = math.gamma(3) / (math.sqrt(5 * math.pi) * math.gamma(2.5))
        pdf = lambda x: normalizer * (1 + x * x / df) ** -3
        integral = (
            step
            / 3
            * (pdf(0) + pdf(high) + sum((4 if i % 2 else 2) * pdf(i * step) for i in range(1, count)))
        )
        self.assertAlmostEqual(student_cdf(high, df), 0.5 + integral, places=11)
        self.assertAlmostEqual(student_cdf(-high, df), 0.5 - integral, places=11)

    def test_mixture_moments_include_between_component_variance(self):
        mean, var, pit = moments_and_pit([-1, 1], [1, 1], [5, 5], [0.5, 0.5], 0)
        self.assertEqual(mean, 0)
        self.assertAlmostEqual(var, 1 + 5 / 3)
        self.assertAlmostEqual(pit, 0.5)

    def test_invalid_mixture_and_undefined_rank_rejected_or_explicit(self):
        with self.assertRaises(ValueError):
            moments_and_pit([0], [0], [5], [1], 0)
        with self.assertRaises(ValueError):
            moments_and_pit([0], [1], [5], [0.5], 0)
        self.assertIsNone(pearson(ranks([1, 1, 1]), ranks([1, 2, 3])))
        self.assertEqual(ranks([4, 4, 2]), [2.5, 2.5, 1.0])

    @staticmethod
    def ledger_fixture():
        # $50k SPY, $20k TLT; $70 entry fee; +10%/-5% returns.
        # End holdings $55k/$19k, $74 terminal fee. Wealth = $103856.
        returns = {1: {"end": 2, "SPY": 0.1, "TLT": -0.05, "GLD": 0}}
        holdings = {"SPY": 55000.0, "TLT": 19000.0, "GLD": 0.0}
        ledger = {
            "terminal_wealth": 103856.0,
            "fees": 144.0,
            "periods": [
                {
                    "time": 1,
                    "end": 2,
                    "nav_start": 100000.0,
                    "nav_end": 103856.0,
                    "notional": 70000.0,
                    "fee": 70.0,
                    "liquidation_fee": 74.0,
                    "cash": 29930.0,
                    "holdings": holdings,
                    "weights": {"SPY": 50000 / 99930, "TLT": 20000 / 99930, "GLD": 0},
                }
            ],
        }
        return ledger, returns

    def test_fixed_position_fee_rebate_exact_dollar_identity(self):
        ledger, returns = self.ledger_fixture()
        result = audit_ledger(ledger, 10, returns)
        self.assertAlmostEqual(result["gross_pnl"], 4000)
        self.assertAlmostEqual(result["gross_pnl_by_asset"]["SPY"], 5000)
        self.assertAlmostEqual(result["gross_pnl_by_asset"]["TLT"], -1000)
        self.assertAlmostEqual(result["fixed_dollar_positions_fees_rebated_to_cash_wealth"], 104000)
        self.assertAlmostEqual(result["net_pnl"], 3856)

    def test_wrong_fee_or_unsynchronized_date_fails(self):
        ledger, returns = self.ledger_fixture()
        ledger["periods"][0]["fee"] += 1
        with self.assertRaises(ValueError):
            audit_ledger(ledger, 10, returns)
        ledger, returns = self.ledger_fixture()
        returns[1]["end"] = 3
        with self.assertRaises(ValueError):
            audit_ledger(ledger, 10, returns)

    def test_exposure_is_reconstructed_when_optional_saved_weights_absent(self):
        ledger, returns = self.ledger_fixture()
        del ledger["periods"][0]["weights"]
        result = audit_ledger(ledger, 10, returns)
        self.assertAlmostEqual(result["average_risky_exposure"], 70000 / 99930)

    def test_legacy_terminal_liquidation_outside_last_period_is_reconciled(self):
        ledger, returns = self.ledger_fixture()
        del ledger["periods"][0]["liquidation_fee"]
        ledger["periods"][0]["nav_end"] += 74
        result = audit_ledger(ledger, 10, returns)
        self.assertAlmostEqual(result["net_pnl"], 3856)
        self.assertAlmostEqual(result["years"]["1970"]["net_return"], 0.03856)

    def test_retrieval_excludes_unmatured_and_self_and_caps_at_sixty(self):
        records = [{"available": float(i), "key": [0.0]} for i in range(70)]
        trace = [
            {
                "stage": "evaluation",
                "date": 65.0,
                "available": 66.0,
                "local_expert_weights": [0.5, 0.5],
                "global_expert_weights": [0.5, 0.5],
            }
        ]
        result = retrieval_support(records, trace)["rows"][0]
        self.assertEqual(result["mature_rows"], 60)
        self.assertAlmostEqual(result["effective_weight_support"], 60)
        self.assertAlmostEqual(result["largest_date_weight"], 1 / 60)
        self.assertAlmostEqual(result["oldest_age_days"], 60 / 86400)

    def test_retrieval_concentration_reduces_effective_support(self):
        records = [
            {"available": 1.0, "key": [0.0]},
            {"available": 2.0, "key": [10.0]},
            {"available": 4.0, "key": [0.0]},
        ]
        trace = [
            {
                "stage": "evaluation",
                "date": 3.0,
                "available": 4.0,
                "local_expert_weights": [0.5, 0.5],
                "global_expert_weights": [0.5, 0.5],
            }
        ]
        result = retrieval_support(records, trace)["rows"][0]
        self.assertAlmostEqual(result["effective_weight_support"], 1)


if __name__ == "__main__":
    unittest.main()
