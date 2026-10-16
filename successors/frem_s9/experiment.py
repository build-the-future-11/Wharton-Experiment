"""Fixed crossed diagnostic, preserved checkpoints and descriptive date intervals."""

import math
import random
from statistics import mean

from successors.frem_audit_v1.audit import read, write

from .data import split, stamp
from .model import fit, predict, validate_checkpoint
from .policy import buy_hold, choose, simulate

FOLDS = (('A', stamp('2010-01-01'), stamp('2013-01-01')),
         ('B', stamp('2013-01-01'), stamp('2017-01-01')))
VARIANTS = ('h1_local', 'h4_local', 'h1_global', 'h4_global')
COSTS = (0, 10, 20, 40)


def percentile(values, probability):
    if not values or not 0 <= probability <= 1 or not all(math.isfinite(x) for x in values):
        raise ValueError('invalid percentile')
    ordered = sorted(values)
    position = probability*(len(ordered)-1)
    low = int(position); high = min(low+1, len(ordered)-1)
    return ordered[low]+(position-low)*(ordered[high]-ordered[low])


def interval(folds, block, *, repetitions=2000, seed=1901):
    if (type(block) is not int or block < 1 or type(repetitions) is not int or repetitions < 2
            or not folds or any(len(f) < block for f in folds)
            or not all(math.isfinite(v) for f in folds for v in f)):
        raise ValueError('invalid paired date-block input')
    generator, draws = random.Random(seed), []
    count = sum(map(len, folds))
    for _ in range(repetitions):
        total = 0.
        for fold in folds:
            sampled = []
            while len(sampled) < len(fold):
                start = generator.randrange(len(fold))
                sampled.extend(fold[(start+k) % len(fold)] for k in range(block))
            total += math.fsum(sampled[:len(fold)])
        draws.append(total/count)
    return {'mean': math.fsum(math.fsum(f) for f in folds)/count,
            'interval95': [percentile(draws, .025), percentile(draws, .975)],
            'block_decisions': block, 'repetitions': repetitions, 'seed': seed,
            'scope': 'Descriptive retrospective paired date blocks; not confirmatory significance.'}


def study(samples, output, *, folds=FOLDS, minimum_train=64, minimum_evaluation=24,
          neighbors=32, repetitions=2000):
    results, differences, loss_baseline, loss_direct = {}, [], [], []
    ledger_count, fits = 0, 0
    for name, cutoff, end in folds:
        folder = output/name; folder.mkdir(exist_ok=False)
        train, evaluation = split(samples, cutoff, end, minimum_train=minimum_train,
                                  minimum_evaluation=minimum_evaluation)
        model = fit(train, cutoff, neighbors=neighbors); fits += 1
        write(folder/'model.json', model)
        reloaded = validate_checkpoint(read(folder/'model.json'))
        predictions, losses = [], {v: [] for v in (*VARIANTS, 'zero')}
        targets = {v: {str(c): {mode: [] for mode in ('joint', 'diagonal')} for c in COSTS}
                   for v in VARIANTS}
        for row in evaluation:
            actual = [g+d-1 for g, d in zip(row['h4']['g'], row['h4']['d'], strict=True)]
            saved = {'origin': row['origin'], 'start': row['start'], 'end': row['end'],
                     'actual_total_return': actual, 'forecasts': {}}
            selected_local = None
            for variant in VARIANTS:
                horizon = int(variant[1]); local = variant.endswith('local')
                predicted = predict(model, row, horizon=horizon, local=local)
                if predicted != predict(reloaded, row, horizon=horizon, local=local):
                    raise ValueError('checkpoint round-trip changed prediction')
                if local:
                    if selected_local is not None and predicted['origins'] != selected_local:
                        raise ValueError('unpaired horizon neighbor selection')
                    selected_local = predicted['origins']
                estimate = predicted['mean_total_return']
                losses[variant].append(mean((a-p)**2 for a, p in zip(actual, estimate, strict=True)))
                view = {'training_origins': predicted['origins'], 'mean_total_return': estimate, 'decisions': {}}
                for cost in COSTS:
                    view['decisions'][str(cost)] = {}
                    for mode in ('joint', 'diagonal'):
                        decision = choose(predicted['scenarios'], cost/10000, diagonal=mode == 'diagonal')
                        targets[variant][str(cost)][mode].append(decision['weights'])
                        view['decisions'][str(cost)][mode] = decision
                saved['forecasts'][variant] = view
            losses['zero'].append(mean(a*a for a in actual))
            predictions.append(saved)
        write(folder/'forecasts.json', predictions)
        economics = {}
        for variant in VARIANTS:
            economics[variant] = {}
            for cost in COSTS:
                economics[variant][str(cost)] = {}
                for mode in ('joint', 'diagonal'):
                    ledger = simulate(evaluation, targets[variant][str(cost)][mode], cost/10000)
                    write(folder/f'{variant}_{mode}_{cost}.json', ledger)
                    economics[variant][str(cost)][mode] = {k: v for k, v in ledger.items() if k != 'periods'}
                    ledger_count += 1
        benchmarks = {}
        for label, weights in [('cash', [0., 0., 0.]), ('equal_assets', [1/3]*3),
                               ('matched_equal_assets', [.8/3]*3), ('equal_buy_hold', None)]:
            benchmarks[label] = {}
            for cost in COSTS:
                ledger = (buy_hold(evaluation, cost/10000) if weights is None
                          else simulate(evaluation, [weights]*len(evaluation), cost/10000))
                write(folder/f'benchmark_{label}_{cost}.json', ledger)
                benchmarks[label][str(cost)] = {k: v for k, v in ledger.items() if k != 'periods'}
                ledger_count += 1
        delta = [a-b for a, b in zip(losses['h1_local'], losses['h4_local'], strict=True)]
        differences.append(delta); loss_baseline.extend(losses['h1_local']); loss_direct.extend(losses['h4_local'])
        primary = economics['h4_local']['10']['joint']['terminal_wealth']
        results[name] = {'train_rows': len(train), 'evaluation_blocks': len(evaluation),
                         'first_origin': evaluation[0]['origin'], 'last_outcome': evaluation[-1]['end'],
                         'unused_terminal_days': (end-evaluation[-1]['end'])/86400,
                         'mse': {v: mean(values) for v, values in losses.items()},
                         'paired_mse_difference': delta, 'economics': economics, 'benchmarks': benchmarks,
                         'investment_gate_6pct': primary > 100000 and all(
                             primary >= 1.06*benchmarks[b]['10']['terminal_wealth']
                             for b in ('equal_assets', 'equal_buy_hold'))}
    intervals = {str(b): interval(differences, b, repetitions=repetitions) for b in (2, 4)}
    baseline = mean(loss_baseline)
    relative = (baseline-mean(loss_direct))/baseline if baseline > 0 else None
    promising = (relative is not None and relative >= .05 and all(mean(f) > 0 for f in differences)
                 and all(v['interval95'][0] > 0 for v in intervals.values()))
    result = {'status': 'EXPLORATORY_DIAGNOSTIC_COMPLETE', 'fits': fits, 'ledgers': ledger_count,
              'folds': results, 'relative_mse_improvement': relative, 'paired_intervals': intervals,
              'horizon_disposition': 'PROMISING_EXPLORATORY_ONLY' if promising else 'INCONCLUSIVE_OR_UNSUPPORTED',
              'investment_gate_6pct_both_folds': all(r['investment_gate_6pct'] for r in results.values()),
              'protected_access': False, 'confirmatory': False, 'independent_reproduction': False}
    write(output/'SUMMARY.json', result)
    return result


def report(summary, output):
    lines = ['# FREM-S9 exploratory result', '',
             'Retrospective training-era diagnostic. No confirmation, novelty or future-profit claim.', '',
             f"Horizon disposition: **{summary['horizon_disposition']}**.",
             f"Six-percent exploratory economic gate in both folds: **{summary['investment_gate_6pct_both_folds']}**.", '',
             '| Fold | Scaled local MSE | Direct local MSE | Primary wealth | Equal assets | Buy and hold |',
             '|---|---:|---:|---:|---:|---:|']
    for name, r in summary['folds'].items():
        lines.append(f"| {name} | {r['mse']['h1_local']:.8g} | {r['mse']['h4_local']:.8g} | "
                     f"{r['economics']['h4_local']['10']['joint']['terminal_wealth']:.2f} | "
                     f"{r['benchmarks']['equal_assets']['10']['terminal_wealth']:.2f} | "
                     f"{r['benchmarks']['equal_buy_hold']['10']['terminal_wealth']:.2f} |")
    lines += ['', 'All variants, costs and controls are retained in SUMMARY.json and individual ledgers.',
              'Drawdown is measured only at four-interval endpoints. No closed predecessor was rerun.', '']
    with (output/'REPORT.md').open('x') as stream:
        stream.write('\n'.join(lines))
