"""Standard-library engineering tests; constructed cases are not market evidence."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from successors.frem_audit_v1.contracts import (
    ReturnInterval,
    compound,
    finite_scenario_moments,
    horizon_log_variance,
    self_financing_target,
)
from successors.frem_audit_v1.ledger import audit_ledger
from successors.frem_audit_v1.stream import iter_array, iter_object


class StreamingTests(unittest.TestCase):
    def decode(self, text, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.json"
            path.write_text(text)
            return dict(iter_object(path, **kwargs))

    def test_every_small_chunk_preserves_nested_strings_and_numbers(self):
        value = {"a": ["é } \\\" ,", {"b": 1.234e-15}], "n": 123456789, "z": None, "b": False}
        for chunk in range(1, 40):
            with self.subTest(chunk=chunk):
                self.assertEqual(self.decode(json.dumps(value), chunk_chars=chunk), value)

    def test_empty_and_surrounding_whitespace(self):
        self.assertEqual(self.decode(" \n {} \n ", chunk_chars=1), {})

    def test_duplicate_top_key_rejected(self):
        with self.assertRaises(ValueError):
            self.decode('{"a":1,"a":2}')

    def test_nested_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            self.decode('{"a":{"x":1,"x":2}}')

    def test_nonfinite_numbers_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                self.decode('{"a":' + token + '}')

    def test_malformed_and_trailing_data_rejected(self):
        for text in ('', '[1]', '{', '{"x":1,}', '{"x":1} false', '{"x":1e}', '{"x":01}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.decode(text, chunk_chars=1)

    def test_large_value_is_bounded(self):
        with self.assertRaises(ValueError):
            self.decode(json.dumps({"a": "x" * 100}), chunk_chars=5, max_value_chars=30)

    def test_array_records_and_numeric_chunk_boundaries(self):
        value = [{"assets": ["SPY", "TLT", "GLD"]}, 123.456e-10, [], "é \\\" ]"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ensemble.json'
            path.write_text(json.dumps(value))
            for size in range(1, 20):
                self.assertEqual(list(iter_array(path, chunk_chars=size)), value)

    def test_array_trailing_comma_and_wrong_container_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ensemble.json'
            for text in ('[1,]', '{}', '[1] null'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    list(iter_array(path, chunk_chars=1))


class HorizonTests(unittest.TestCase):
    def test_compounding_is_not_sum(self):
        rows = [ReturnInterval(0, 1, 1, .1), ReturnInterval(1, 3, 3, -.1)]
        self.assertAlmostEqual(compound(rows, start=0, end=3, decision_time=4), -.01)

    def test_variable_length_intervals(self):
        rows = [ReturnInterval(0, 3, 3, .1), ReturnInterval(3, 13, 13, .2)]
        self.assertAlmostEqual(compound(rows, start=0, end=13, decision_time=14), .32)

    def test_total_loss_preserved(self):
        rows = [ReturnInterval(0, 1, 1, -1), ReturnInterval(1, 2, 2, .2)]
        self.assertEqual(compound(rows, start=0, end=2, decision_time=3), -1)

    def test_future_and_same_time_outcome_rejected(self):
        for decision in (1, 2):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                compound([ReturnInterval(0, 1, 2, .1)], start=0, end=1, decision_time=decision)

    def test_gap_and_overlap_rejected(self):
        for second_start in (.5, 2):
            with self.subTest(start=second_start), self.assertRaises(ValueError):
                compound([ReturnInterval(0, 1, 1, .1), ReturnInterval(second_start, 3, 3, .1)],
                         start=0, end=3, decision_time=4)

    def test_wrong_end_rejected(self):
        with self.assertRaises(ValueError):
            compound([ReturnInterval(0, 1, 1, .1)], start=0, end=2, decision_time=3)

    def test_invalid_intervals_rejected(self):
        for args in ((0, 0, 1, 0), (0, 1, 0, 0), (0, 1, 1, -1.1), (0, 1, 1, float('nan'))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ReturnInterval(*args)

    def test_serial_covariance_changes_horizon_risk(self):
        self.assertAlmostEqual(horizon_log_variance(.01, [0, 0], 3), .03)
        self.assertAlmostEqual(horizon_log_variance(.01, [.005, .0025], 3), .055)

    def test_missing_lags_rejected(self):
        with self.assertRaises(ValueError):
            horizon_log_variance(.01, [], 3)

    def test_impossible_sum_variance_rejected(self):
        with self.assertRaises(ValueError):
            horizon_log_variance(.01, [-.01, -.01], 3)


class ScenarioTests(unittest.TestCase):
    def test_joint_covariance_detects_offsetting_assets(self):
        mean, covariance = finite_scenario_moments([[-.1, .1], [.1, -.1]], [.5, .5])
        self.assertEqual(mean, [0, 0])
        self.assertAlmostEqual(covariance[0][0], .01)
        self.assertAlmostEqual(covariance[0][1], -.01)
        self.assertAlmostEqual(sum(sum(row) for row in covariance) / 4, 0)

    def test_covariance_quadratic_matches_scenario_variance(self):
        rows, p, w = [[-.1, .2], [.05, -.1], [.2, .1]], [.2, .3, .5], [.3, .6]
        mean, covariance = finite_scenario_moments(rows, p)
        direct = sum(q * (sum(a * b for a, b in zip(w, x)) - sum(a * b for a, b in zip(w, mean)))**2
                     for q, x in zip(p, rows))
        matrix = sum(w[i] * covariance[i][j] * w[j] for i in range(2) for j in range(2))
        self.assertAlmostEqual(matrix, direct)
        self.assertGreaterEqual(matrix, 0)

    def test_invalid_probability_support_and_shape_rejected(self):
        for rows, p in (([[0]], [.5]), ([[0]], [-1]), ([[-1.01]], [1]), ([[0], [0, 1]], [.5, .5])):
            with self.subTest(rows=rows, p=p), self.assertRaises(ValueError):
                finite_scenario_moments(rows, p)

    def test_exp_mean_log_is_not_mean_simple(self):
        logs = [math.log(.5), math.log(1.5)]
        self.assertAlmostEqual(sum([-.5, .5]) / 2, 0)
        self.assertLess(math.expm1(sum(logs) / 2), -.13)


class AccountingReferenceTests(unittest.TestCase):
    def test_entry_exact_post_fee_basis(self):
        result = self_financing_target([0], 100, [1], .01)
        self.assertAlmostEqual(result['assets'][0], 100 / 1.01)
        self.assertAlmostEqual(result['fee'], 100 / 101)
        self.assertAlmostEqual(result['cash'], 0)

    def test_exit_exact(self):
        result = self_financing_target([100], 0, [0], .01)
        self.assertEqual(result['assets'], [0])
        self.assertAlmostEqual(result['cash'], 99)
        self.assertAlmostEqual(result['fee'], 1)

    def test_hold_has_no_fee(self):
        result = self_financing_target([40, 40], 20, [.4, .4], .01)
        self.assertAlmostEqual(result['fee'], 0)
        self.assertAlmostEqual(sum(result['assets']) + result['cash'], 100)

    def test_long_only_cash_budget_rejections(self):
        for holdings, cash, weights, fee in (([-1], 10, [1], 0), ([0], 10, [1.1], 0),
                                           ([0], 10, [1], 1), ([0], -1, [0], .01)):
            with self.subTest(weights=weights, fee=fee), self.assertRaises(ValueError):
                self_financing_target(holdings, cash, weights, fee)


def example_ledger():
    start = 1577836800
    return {
        'terminal_wealth': 99.5, 'net_return': -.005, 'fees': 0,
        'max_drawdown': 1 - 99.5 / 105,
        'periods': [
            {'time': start, 'end': start+7*86400, 'nav_start': 100, 'nav_end': 105,
             'fee': 0, 'notional': 50, 'cash': 50, 'weights': {'A': .5},
             'holdings': {'A': 55}, 'contributions': {'A': 5}, 'net_factor': 1.05},
            {'time': start+7*86400, 'end': start+14*86400, 'nav_start': 105, 'nav_end': 99.5,
             'fee': 0, 'notional': 0, 'cash': 50, 'weights': {'A': 55/105},
             'holdings': {'A': 49.5}, 'contributions': {'A': -5.5}, 'net_factor': 99.5/105,
             'liquidation_fee': 0},
        ],
    }


class LedgerTests(unittest.TestCase):
    def test_dollar_attribution_telescope(self):
        result = audit_ledger(example_ledger(), cost_bps=0)
        self.assertAlmostEqual(result['asset_contributions']['A'], -.5)
        self.assertAlmostEqual(result['terminal_wealth'], 99.5)
        self.assertEqual(result['annual']['2020']['periods'], 2)

    def test_missing_weights_recovered_from_saved_contributions(self):
        ledger = example_ledger()
        for row in ledger['periods']:
            row.pop('weights'); row.pop('net_factor')
        self.assertAlmostEqual(audit_ledger(ledger, cost_bps=0)['gross_dollar_contribution'], -.5)

    def test_drawdown_reconstructed_when_not_stored(self):
        ledger = example_ledger(); ledger.pop('max_drawdown')
        self.assertAlmostEqual(audit_ledger(ledger, cost_bps=0)['max_drawdown'], 1-99.5/105)

    def test_dimensionless_ratio_requires_tight_tolerance(self):
        ledger = example_ledger(); ledger['periods'][0]['net_factor'] += 1e-8
        with self.assertRaises(ValueError):
            audit_ledger(ledger, cost_bps=0)

    def test_positional_decisions_require_explicit_asset_order(self):
        ledger = example_ledger()
        ledger['periods'][0]['decision'] = {'mean':[.01], 'target':[.5]}
        with self.assertRaises(ValueError):
            audit_ledger(ledger, cost_bps=0)
        result = audit_ledger(ledger, cost_bps=0, decision_assets=['A'])
        self.assertEqual(result['positive_log_mean_decisions']['A'], 1)

    def test_temporal_gap_rejected(self):
        ledger = example_ledger(); ledger['periods'][1]['time'] += 1
        with self.assertRaises(ValueError):
            audit_ledger(ledger, cost_bps=0)

    def test_duplicate_positional_asset_identity_rejected(self):
        ledger = example_ledger()
        ledger['periods'][0]['decision'] = {'mean': [.01, .02], 'target': [.5, .5]}
        with self.assertRaises(ValueError):
            audit_ledger(ledger, cost_bps=0, decision_assets=['A', 'A'])

    def test_corrupted_nav_rejected(self):
        ledger = example_ledger(); ledger['periods'][1]['nav_start'] += 1
        with self.assertRaises(ValueError):
            audit_ledger(ledger, cost_bps=0)

    def test_corrupted_contribution_rejected(self):
        ledger = example_ledger(); ledger['periods'][0]['contributions']['A'] = 99
        with self.assertRaises(ValueError):
            audit_ledger(ledger, cost_bps=0)

    def test_nonfinite_values_rejected(self):
        ledger = example_ledger(); ledger['periods'][0]['fee'] = float('nan')
        with self.assertRaises(ValueError):
            audit_ledger(ledger, cost_bps=0)

    def test_terminal_fee_not_omitted(self):
        ledger = example_ledger()
        with self.assertRaises(ValueError):
            audit_ledger(ledger, cost_bps=10)

    def test_new_peak_before_final_liquidation(self):
        ledger = example_ledger(); ledger['periods'] = [copy.deepcopy(ledger['periods'][0])]
        row = ledger['periods'][0]
        post = 100 / 1.0005
        row.update(fee=post*.0005, notional=post*.5, cash=post*.5,
                   holdings={'A':post*.55}, contributions={'A':post*.05})
        row['liquidation_fee'] = row['holdings']['A']*.001
        row['nav_end'] = post*1.05-row['liquidation_fee']
        row['net_factor'] = row['nav_end']/100
        ledger.update(terminal_wealth=row['nav_end'], net_return=row['nav_end']/100-1,
                      fees=row['fee']+row['liquidation_fee'],
                      max_drawdown=row['liquidation_fee']/(post*1.05))
        self.assertLess(audit_ledger(ledger, cost_bps=10)['max_absolute_accounting_error'], 1e-9)

    def test_root_only_liquidation_schema(self):
        ledger = example_ledger()
        # A zero-trade, pre-existing risky holding supports a nonzero exit fee.
        row = ledger['periods'][0]
        row.update(nav_start=100, nav_end=110, weights={'A':1.0}, cash=0,
                   holdings={'A':110}, contributions={'A':10}, notional=0,
                   net_factor=1.1)
        ledger['periods'] = [row]
        ledger.update(terminal_wealth=109.89, net_return=.0989, fees=.11,
                      liquidation_fee=.11, max_drawdown=.001)
        result = audit_ledger(ledger, cost_bps=10)
        self.assertAlmostEqual(result['terminal_wealth'], 109.89)
        self.assertAlmostEqual(result['gross_dollar_contribution'], 10)


if __name__ == '__main__':
    unittest.main()
