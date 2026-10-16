"""Constructed numerical/chronology cases, not empirical market results."""

import copy
import json
import math
import platform
import sys
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from statistics import mean, pvariance
from unittest.mock import patch

from successors.frem_audit_v1.audit import digest
from successors.frem_s9.data import ASSETS, DAY, date_of, outcome, prepare, split, stamp
from successors.frem_s9.experiment import interval, percentile, report, study
from successors.frem_s9.model import fit, predict, validate_checkpoint
from successors.frem_s9.policy import (
    ACTIONS,
    buy_hold,
    choose,
    moments,
    objective,
    round_trip,
    simulate,
    terminal_factor,
)
from successors.frem_s9.run import assess, main, verify


def sample(index):
    origin = stamp('2005-01-03')+index*7*DAY+12*3600
    value = .01*math.sin(index/3)
    return {'index': index, 'origin': origin, 'start': origin+2*3600,
            'end': origin+28*DAY+2*3600, 'available': origin+28*DAY+12*3600,
            'feature_available': origin-DAY, 'x': [math.sin(index/(j+2)) for j in range(12)],
            'h1': {'g': [1+value, 1-value/2, 1+value/3], 'd': [.001, .002, 0.]},
            'h4': {'g': [1+value*2, 1-value, 1+value/2], 'd': [.004, .008, 0.]}}


def raw_fixture(root):
    sources = []
    for asset in ASSETS:
        rows = [{'timestamp': stamp('2005-01-03')+i*7*DAY+14*3600,
                 'open': 100+i, 'close': 101+i} for i in range(160)]
        value = {'symbol': asset, 'rows': rows, 'split_events': {}, 'dividends': {}}
        p = root/f'{asset}.json'; p.write_text(json.dumps(value))
        sources.append({'symbol': asset, 'retained_sha256': digest(p), 'retrieved_at': '2026-10-13'})
    (root/'acquisition.json').write_text(json.dumps({'sources': sources}))


class DataTests(unittest.TestCase):
    def test_dividend_boundary_and_no_reinvestment(self):
        prices = {a: {'a': 100, 'b': 110} for a in ASSETS}
        dividends = {a: [('a', 7), ('b', 2)] for a in ASSETS}
        value = outcome(prices, dividends, 'a', 'b')
        self.assertEqual(value, {'g': [1.1]*3, 'd': [.02]*3})

    def test_source_adapter_calendar_and_maturity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); raw_fixture(root)
            data, receipt = prepare(root)
            self.assertEqual(receipt['daily_sessions'], 160)
            self.assertTrue(all(r['feature_available'] < r['origin'] < r['start'] < r['end'] <= r['available'] for r in data))
            self.assertTrue(all(len(r['x']) == 12 for r in data))
            self.assertEqual(date_of(data[0]['origin']), '2005-04-11')

    def test_hash_corruption_rejected_before_parse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); raw_fixture(root)
            (root/'SPY.json').write_text('broken')
            with self.assertRaisesRegex(ValueError, 'hash'):
                prepare(root)

    def test_calendar_omission_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); raw_fixture(root)
            p = root/'TLT.json'; value = json.loads(p.read_text()); value['rows'].pop(20)
            p.write_text(json.dumps(value))
            acquisition = json.loads((root/'acquisition.json').read_text())
            acquisition['sources'][1]['retained_sha256'] = digest(p)
            (root/'acquisition.json').write_text(json.dumps(acquisition))
            with self.assertRaisesRegex(ValueError, 'calendar'):
                prepare(root)

    def test_split_purges_four_interval_outcomes(self):
        rows = [sample(i) for i in range(150)]
        cutoff = rows[80]['origin']-12*3600
        train, evaluation = split(rows, cutoff, rows[-1]['origin'], minimum_evaluation=4)
        self.assertTrue(all(r['available'] < cutoff for r in train))
        self.assertTrue(all(a['end'] == b['start'] for a, b in pairwise(evaluation)))
        self.assertEqual(evaluation[0]['index'], 80)

    def test_future_fold_and_insufficient_samples_rejected(self):
        with self.assertRaises(ValueError):
            split([], stamp('2016-01-01'), stamp('2018-01-01'))
        with self.assertRaises(ValueError):
            split([sample(0)], stamp('2010-01-01'), stamp('2013-01-01'))

    def test_later_source_changes_cannot_change_earlier_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); raw_fixture(root)
            before, _ = prepare(root)
            p = root/'SPY.json'; value = json.loads(p.read_text())
            value['rows'][-1]['open'] *= 2
            p.write_text(json.dumps(value))
            acquisition = json.loads((root/'acquisition.json').read_text())
            acquisition['sources'][0]['retained_sha256'] = digest(p)
            (root/'acquisition.json').write_text(json.dumps(acquisition))
            after, _ = prepare(root)
            self.assertEqual([r['x'] for r in before], [r['x'] for r in after])
            self.assertNotEqual(before[-1]['h4'], after[-1]['h4'])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.rows = [sample(i) for i in range(64)]
        self.cutoff = sample(70)['origin']
        self.model = fit(self.rows, self.cutoff)
        self.query = sample(80)

    def test_horizons_use_identical_neighbors(self):
        a = predict(self.model, self.query, horizon=1, local=True)
        b = predict(self.model, self.query, horizon=4, local=True)
        self.assertEqual(a['origins'], b['origins'])
        self.assertEqual(len(a['origins']), 32)

    def test_global_retains_every_training_origin(self):
        result = predict(self.model, self.query, horizon=4, local=False)
        self.assertEqual(len(result['origins']), 64)

    def test_constant_repetition_price_and_dividend_formula(self):
        result = predict(self.model, self.query, horizon=1, local=False)
        r = self.rows[3]['h1']; out = result['scenarios'][3]
        self.assertAlmostEqual(out['g'][0], r['g'][0]**4)
        self.assertAlmostEqual(out['d'][0], r['d'][0]*sum(r['g'][0]**k for k in range(4)))

    def test_checkpoint_roundtrip_exact(self):
        reloaded = validate_checkpoint(json.loads(json.dumps(self.model)))
        self.assertEqual(predict(reloaded, self.query, horizon=4, local=True),
                         predict(self.model, self.query, horizon=4, local=True))

    def test_scaling_uses_train_only(self):
        before = copy.deepcopy(self.model)
        self.query['x'] = [1e3]*12
        predict(self.model, self.query, horizon=4, local=True)
        self.assertEqual(before, self.model)
        self.assertAlmostEqual(self.model['means'][0], mean(r['x'][0] for r in self.rows))

    def test_constant_features_and_tie_order(self):
        for r in self.rows:r['x'] = [0.]*12
        model = fit(self.rows, self.cutoff)
        self.query['x'] = [0.]*12
        self.assertEqual(model['scales'], [1.]*12)
        self.assertEqual(predict(model, self.query, horizon=4, local=True)['origins'],
                         [r['origin'] for r in self.rows[:32]])

    def test_unmatured_training_rejected(self):
        self.rows[-1]['available'] = self.cutoff
        with self.assertRaises(ValueError):fit(self.rows, self.cutoff)

    def test_checkpoint_leakage_rejected(self):
        self.model['training'][-1]['available'] = self.cutoff
        with self.assertRaises(ValueError):validate_checkpoint(self.model)

    def test_query_leakage_rejected(self):
        self.query['feature_available'] = self.query['origin']
        with self.assertRaises(ValueError):predict(self.model, self.query, horizon=4, local=True)

    def test_nonfinite_features_and_invalid_support_rejected(self):
        self.rows[0]['x'][0] = float('nan')
        with self.assertRaises(ValueError):fit(self.rows, self.cutoff)
        self.rows[0] = sample(0); self.rows[0]['h4']['g'][0] = 0
        with self.assertRaises(ValueError):fit(self.rows, self.cutoff)


class PolicyTests(unittest.TestCase):
    def test_entry_exit_and_dividend_cash_not_taxed_as_sale(self):
        result = round_trip(100, [1, 0, 0], {'g': [1, 1, 1], 'd': [.1, 0, 0]}, .01)
        self.assertAlmostEqual(result['nav_end'], 100/1.01*1.09)
        self.assertAlmostEqual(result['exit_fee'], 100/1.01*.01)

    def test_cash_is_exact_and_has_zero_fees(self):
        r = round_trip(100, [0, 0, 0], sample(0)['h4'], .004)
        self.assertEqual(r['nav_end'], 100); self.assertEqual(r['fees'], 0)

    def test_joint_objective_equals_direct_scenario_variance(self):
        scenarios = [sample(i)['h4'] for i in range(12)]
        w = (.25, .25, .25); c = .001
        mu, cov = moments(scenarios, c)
        values = [terminal_factor(w, s, c) for s in scenarios]
        self.assertAlmostEqual(objective(w, mu, cov, c, False), mean(values)-1-4*pvariance(values), places=14)

    def test_negative_covariance_changes_risk(self):
        scenarios = [{'g': [1.1, .9, 1], 'd': [0, 0, 0]}, {'g': [.9, 1.1, 1], 'd': [0, 0, 0]}]
        mu, cov = moments(scenarios, 0)
        self.assertGreater(objective((.3, .3, 0), mu, cov, 0, False),
                           objective((.3, .3, 0), mu, cov, 0, True))

    def test_finite_grid_optimum_and_limits(self):
        scenarios = [sample(i)['h4'] for i in range(40)]
        selected = choose(scenarios, .001); mu, cov = moments(scenarios, .001)
        self.assertEqual(selected['objective'], max(objective(w, mu, cov, .001, False) for w in ACTIONS))
        self.assertTrue(all(sum(w) <= .8+1e-12 and max(w) <= 1/3 for w in ACTIONS))
        self.assertGreaterEqual(selected['objective'], 0)

    def test_tie_chooses_cash(self):
        self.assertEqual(choose([{'g': [1, 1, 1], 'd': [0, 0, 0]}], 0)['weights'], [0, 0, 0])

    def test_invalid_fee_weight_and_return_rejected(self):
        for weights, fee in (([1.1, 0, 0], .001), ([1, 0, 0], -1), ([-1, 0, 0], 0)):
            with self.assertRaises(ValueError):terminal_factor(weights, sample(0)['h4'], fee)

    def test_continuous_simulation_and_final_cash(self):
        periods = [sample(i) for i in (0, 4, 8)]
        ledger = simulate(periods, [[.25]*3]*3, .001)
        self.assertAlmostEqual(ledger['terminal_wealth'], 100000*math.prod(r['net_factor'] for r in ledger['periods']))
        self.assertAlmostEqual(ledger['fees'], sum(r['fees'] for r in ledger['periods']))

    def test_ledger_gap_rejected(self):
        with self.assertRaises(ValueError):simulate([sample(0), sample(5)], [[0]*3]*2, 0)

    def test_buy_hold_dividends_are_not_reinvested(self):
        periods = [sample(0), sample(4)]
        for p in periods:p['h4'] = {'g': [1.1]*3, 'd': [.1]*3}
        ledger = buy_hold(periods, 0)
        self.assertAlmostEqual(ledger['terminal_wealth'], 142000)

    def test_buy_hold_costs_only_entry_and_exit(self):
        periods = [sample(0), sample(4)]
        for p in periods:p['h4'] = {'g': [1]*3, 'd': [0]*3}
        ledger = buy_hold(periods, .001)
        self.assertAlmostEqual(ledger['terminal_wealth'], 100000*(1-.001)/(1+.001))


class StatisticsAndExecutionTests(unittest.TestCase):
    def test_percentile_interpolation(self):
        self.assertEqual(percentile([0, 10], .25), 2.5)

    def test_constant_date_blocks_and_determinism(self):
        result = interval([[2.]*8, [2.]*12], 4, repetitions=20)
        self.assertEqual(result['interval95'], [2, 2])
        self.assertEqual(result, interval([[2.]*8, [2.]*12], 4, repetitions=20))

    def test_block_validation(self):
        for folds, block in (([[1]], 2), ([[float('nan')]*5], 2), ([], 2)):
            with self.assertRaises(ValueError):interval(folds, block)

    def test_folds_weighted_by_original_date_counts(self):
        result = interval([[1.]*4, [3.]*8], 2, repetitions=20)
        self.assertAlmostEqual(result['mean'], 7/3)
        self.assertEqual(result['interval95'], [7/3, 7/3])

    def test_host_admission_floor_and_pageouts(self):
        good = [{'free_plus_speculative_mib': 600, 'pageouts': 10} for _ in range(3)]
        self.assertTrue(assess(good))
        bad = copy.deepcopy(good); bad[1]['free_plus_speculative_mib'] = 511
        self.assertFalse(assess(bad))
        bad = copy.deepcopy(good); bad[-1]['pageouts'] += 1
        self.assertFalse(assess(bad))

    def test_complete_constructed_system_and_regenerated_report(self):
        rows = [sample(i) for i in range(175)]
        folds = [('A', rows[75]['origin']-12*3600, rows[115]['origin']-12*3600),
                 ('B', rows[115]['origin']-12*3600, rows[170]['origin']-12*3600)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = study(rows, root, folds=folds, minimum_evaluation=4, repetitions=20)
            self.assertEqual(result['fits'], 2); self.assertEqual(result['ledgers'], 96)
            self.assertFalse(result['confirmatory'])
            report(json.loads((root/'SUMMARY.json').read_text()), root)
            self.assertIn('Retrospective', (root/'REPORT.md').read_text())
            self.assertEqual(len(list((root/'A').glob('*.json'))), 50)

    def test_freeze_identity_and_environment_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'source.txt').write_text('source')
            frozen = root/'freeze.json'
            value = {'kind': 'S9_RETROSPECTIVE_DEVELOPMENT_FREEZE', 'confirmation': False,
                     'python': sys.version, 'platform': platform.platform(),
                     'identities': {'source.txt': digest(root/'source.txt')}}
            frozen.write_text(json.dumps(value)); verify(root, frozen)
            (root/'source.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'identity'):verify(root, frozen)
            (root/'source.txt').write_text('source'); value['python'] = 'different'
            frozen.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'environment'):verify(root, frozen)

    def test_blocked_launcher_never_prepares_or_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); frozen = root/'freeze.json'; frozen.write_text('{}')
            output = root/'blocked'
            argv = ['s9', '--root', str(root), '--freeze', str(frozen), '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch('successors.frem_s9.run.verify'), \
                    patch('successors.frem_s9.run.admission', return_value={'admitted': False}), \
                    patch('successors.frem_s9.run.prepare') as loader, \
                    patch('successors.frem_s9.run.study') as fitter:
                self.assertEqual(main(), 3)
                loader.assert_not_called(); fitter.assert_not_called()
            receipt = json.loads((output/'RECEIPT.json').read_text())
            self.assertEqual(receipt['market_fits'], 0)
            self.assertEqual(list(root.glob('consumed_*.json')), [])

    def test_consumed_freeze_cannot_fit_again(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); frozen = root/'freeze.json'; frozen.write_text('{}')
            summary = {'fits': 2, 'ledgers': 96, 'status': 'CONSTRUCTED_TEST_ONLY',
                       'horizon_disposition': 'CONSTRUCTED_TEST_ONLY', 'investment_gate_6pct_both_folds': False}
            with patch('successors.frem_s9.run.verify'), \
                    patch('successors.frem_s9.run.admission', return_value={'admitted': True}), \
                    patch('successors.frem_s9.run.prepare', return_value=([], {})), \
                    patch('successors.frem_s9.run.study', return_value=summary) as fitter, \
                    patch('successors.frem_s9.run.report'):
                for i, expected in ((1, 0), (2, 1)):
                    argv = ['s9', '--root', str(root), '--freeze', str(frozen), '--output', str(root/f'run{i}')]
                    with patch.object(sys, 'argv', argv):self.assertEqual(main(), expected)
                self.assertEqual(fitter.call_count, 1)
            self.assertEqual(json.loads((root/'run2/RECEIPT.json').read_text())['market_fits'], 0)


if __name__ == '__main__':
    unittest.main()
