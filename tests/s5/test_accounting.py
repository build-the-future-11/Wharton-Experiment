import unittest

from successors.frem_portfolio_v1.engine import Forecast, backtest
from successors.frem_s5.experiment import volatility_ledger


class AccountingTests(unittest.TestCase):
    def test_full_exposure_volatility_control_matches_equal_weight(self):
        periods = [
            {
                "time": i,
                "end": i + 1,
                "currency": "USD",
                "forecasts": [Forecast("a", 0, 0.01 / 12), Forecast("b", 0, 0.01 / 12)],
                "returns": {"a": a, "b": b},
            }
            for i, (a, b) in enumerate(((0.1, -0.05), (-0.2, 0.1), (0.04, 0.03)))
        ]
        for cost in (0, 10, 40):
            actual = volatility_ledger(periods, cost)
            expected = backtest(periods, policy="equal_weight", cost_bps=cost)
            self.assertAlmostEqual(actual["terminal_wealth"], expected["terminal_wealth"], places=8)
            self.assertAlmostEqual(actual["fees"], expected["fees"], places=8)

    def test_high_volatility_leaves_cash(self):
        periods = [
            {
                "time": 0,
                "end": 1,
                "currency": "USD",
                "forecasts": [Forecast("a", 0, 1 / 12)],
                "returns": {"a": 0.1},
            }
        ]
        actual = volatility_ledger(periods, 0)
        self.assertAlmostEqual(actual["terminal_wealth"], 101000)
        self.assertAlmostEqual(actual["periods"][0]["cash"], 90000)
