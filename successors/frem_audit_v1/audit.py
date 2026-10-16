"""Read-only S8 saved-ledger attribution with bounded memory and failure receipts."""

import argparse
import hashlib
import json
import platform
import resource
import signal
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from .ledger import audit_ledger
from .stream import finite_float, invalid_constant, iter_array, iter_object, unique_object


def read(path):
    with Path(path).open() as stream:
        return json.load(stream, object_pairs_hook=unique_object,
                         parse_constant=invalid_constant, parse_float=finite_float)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(262144), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def rss_mib():
    divisor = 2**20 if platform.system() == 'Darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def verify(root, identities):
    for name, expected in identities.items():
        if digest(root / name) != expected:
            raise ValueError(f'preserved source/evidence identity changed: {name}')


def audit(root, manifest, output):
    original = read(manifest)['preserved_identities']
    verify(root, original)
    run = root / 'outputs/frem_s8_2020_2022_20261014_attempt1'
    summary = read(root / 'research/frem_s8_20261014/SUMMARY.json')
    assets, dates = None, []
    for row in iter_array(run / 'ensembles.json'):
        if row['variant'] == 's8':
            if assets is None:
                assets = row['assets']
            if row['assets'] != assets:
                raise ValueError('inconsistent primary asset grid')
            dates.append(row['date'])
    if (assets is None or len(set(assets)) != len(assets) or len(dates) != 153
            or len(set(dates)) != 153 or dates != sorted(dates)):
        raise ValueError('primary asset identity unavailable')
    results, max_error, count, periods = {}, 0.0, 0, 0
    for variant, costs in iter_object(run / 'economics.json'):
        if set(costs) != {'0', '10', '20', '40'}:
            raise ValueError('cost family incomplete')
        results[variant] = {}
        for cost, policies in costs.items():
            if set(policies) != {'signal', 'cost_aware', 's8_h4', 's8_h1'}:
                raise ValueError('policy family incomplete')
            results[variant][cost] = {}
            for policy, ledger in policies.items():
                try:
                    item = audit_ledger(ledger, cost_bps=float(cost), decision_assets=assets)
                except (ValueError, KeyError) as error:
                    raise ValueError(f'{variant}/{cost}/{policy}: {error}') from error
                if item['periods'] != 153:
                    raise ValueError('incomplete continuous weekly path')
                expected = summary['economics'][variant][cost][policy]
                for key in ('terminal_wealth', 'fees'):
                    if abs(item[key] - expected[key]) > 1e-6:
                        raise ValueError(f'summary mismatch: {variant}/{cost}/{policy}/{key}')
                if abs(item['max_drawdown'] - expected['max_drawdown']) > 1e-12:
                    raise ValueError(f'summary drawdown mismatch: {variant}/{cost}/{policy}')
                results[variant][cost][policy] = item
                max_error = max(max_error, item['max_absolute_accounting_error'])
                count += 1
                periods += item['periods']
        if rss_mib() > 64:
            raise RuntimeError('bounded audit exceeded 64 MiB peak RSS')
    if set(results) != set(summary['economics']) or len(results) != 15:
        raise ValueError('forecast family incomplete')
    verify(root, original)
    primary = results['s8']['10']['s8_h4']
    point_intervals = summary['paired_prediction']['equal']
    result = {
        'status': 'PASS', 'saved_ledgers_verified': count, 'period_records_verified': periods,
        'preserved_file_identities': len(original), 'max_absolute_accounting_error': max_error,
        'primary': primary, 'all_forecast_policy_cost_attributions': results,
        'predictive_intervals_vs_equal': point_intervals,
        'both_predictive_upper_bounds_below_0_01': all(v['interval95'][1] < .01 for v in point_intervals.values()),
        'equal_asset_wealth_10bps': summary['benchmarks']['equal_weight']['10']['terminal_wealth'],
        'primary_zero_cost_wealth': results['s8']['0']['s8_h4']['terminal_wealth'],
        'primary_zero_cost_asset_contributions': results['s8']['0']['s8_h4']['asset_contributions'],
        'scope': 'Saved-output arithmetic and descriptive attribution only; no policy replay, new model fit, protected access, or independent retraining.',
    }
    write(output / 'ATTRIBUTION.json', result)
    lines = [
        '# Saved S8 loss attribution', '',
        f'Verified {count} saved ledgers and {periods:,} interval records. No strategy was replayed.', '',
        '| Asset | Primary dollar contribution before fees | Average weight |',
        '|---|---:|---:|',
    ]
    for asset in assets:
        lines.append(f"| {asset} | ${primary['asset_contributions'][asset]:,.2f} | {primary['average_asset_weights'][asset]:.2%} |")
    lines += [
        '', (f"Gross realized contribution: **${primary['gross_dollar_contribution']:,.2f}**. "
        f"Fees including liquidation: **${primary['fees']:,.2f}**. "
        f"Ending wealth: **${primary['terminal_wealth']:,.2f}** from $100,000."), '',
        '| Origin year | SPY contribution | TLT contribution | GLD contribution | Fees | Net return |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for year, part in primary['annual'].items():
        row = ' | '.join(f"${part['asset_contributions'][a]:,.2f}" for a in assets)
        lines.append(f"| {year} | {row} | ${part['fees']:,.2f} | {part['net_return']:.3%} |")
    lines += [
        '', ('Contributions telescope in dollars on the carried path. They are not reset-year backtests, '
        'causal interventions, evidence that the missed rally was predictable, or an authorization to trade the past differently.'), '',
        (f"The primary zero-cost path ended at ${result['primary_zero_cost_wealth']:,.2f}. "
        'Its difference from the 10-bps path combines cost-dependent decisions, fees and compounding; '
        'it is not a pure fee estimate.'), '',
        ('Both saved primary intervals against fixed averaging have upper bounds below the 0.01-nat practical margin. '
        'Their lower bounds also include negative effects. More optimization seeds on the same path would not create independent market histories.'), '',
        f"Maximum accounting residual: {max_error:.3g}. Original sources and saved evidence unchanged.", '',
    ]
    (output / 'REPORT.md').write_text('\n'.join(lines))
    return {k: v for k, v in result.items() if k not in ('all_forecast_policy_cost_attributions', 'primary')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def timeout(_signal, _frame):
        raise TimeoutError('90-second audit limit exceeded')

    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90)
    code = 0
    try:
        result = audit(args.root, args.manifest, args.output)
    except Exception as error:  # noqa: BLE001 -- CLI boundary retains unexpected failures and returns nonzero.
        code = 1
        result = {'status': 'FAILED', 'error': str(error), 'exception': type(error).__name__}
        (args.output / 'failure.log').write_text(traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    result.update(completed_at=datetime.now(UTC).isoformat(),
                  seconds=time.monotonic()-started, peak_rss_mib=rss_mib(),
                  market_fits=0, new_policy_replays=0, protected_access=False)
    write(args.output / 'RECEIPT.json', result)
    print(json.dumps(result, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
