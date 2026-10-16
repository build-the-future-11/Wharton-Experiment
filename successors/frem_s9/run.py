"""Guarded one-shot S9 execution; blocked and failed attempts exit nonzero."""

import argparse
import json
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from successors.frem_audit_v1.audit import digest, read, rss_mib, write

from .data import prepare
from .experiment import report, study


def assess(samples):
    return (len(samples) == 3 and all(s['free_plus_speculative_mib'] >= 512 for s in samples)
            and all(a['pageouts'] == b['pageouts'] for a, b in pairwise(samples)))


def admission():
    samples = []
    for _ in range(3):
        raw = subprocess.check_output(['vm_stat'], text=True, timeout=5)
        size = int(re.search(r'page size of (\d+) bytes', raw).group(1))
        counts = {k: int(v) for k, v in re.findall(r'^([^:\n]+):\s*(\d+)\.', raw, re.MULTILINE)}
        samples.append({'free_plus_speculative_mib': (counts['Pages free']+counts['Pages speculative'])*size/2**20,
                        'pageouts': counts['Pageouts']})
        time.sleep(.5)
    return {'samples': samples, 'admitted': assess(samples), 'floor_mib': 512}


def verify(root, frozen):
    freeze = read(frozen)
    if freeze['kind'] != 'S9_RETROSPECTIVE_DEVELOPMENT_FREEZE' or freeze['confirmation'] is not False:
        raise ValueError('wrong freeze scope')
    for name, expected in freeze['identities'].items():
        if digest(root/name) != expected:
            raise ValueError(f'frozen identity changed: {name}')
    if freeze['python'] != sys.version or freeze['platform'] != platform.platform():
        raise ValueError('frozen environment mismatch')
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started, code, market_fits = time.monotonic(), 0, 0
    result = {'status': 'FAILED_BEFORE_ADMISSION'}
    old = signal.getsignal(signal.SIGALRM)

    def guard(*_):
        if time.monotonic()-started > 600 or rss_mib() > 128:
            raise RuntimeError('600-second/128-MiB S9 ceiling exceeded')

    try:
        verify(args.root, args.freeze)
        gate = admission(); write(args.output/'ADMISSION.json', gate)
        if not gate['admitted']:
            code = 3
            result = {'status': 'BLOCKED_RESOURCE_ADMISSION', 'market_fits': 0}
        else:
            signal.signal(signal.SIGALRM, guard)
            signal.setitimer(signal.ITIMER_REAL, .25, .25)
            guard()
            data, receipt = prepare(args.root/'data/s6_real_etf_20261014')
            write(args.output/'DATA_ADMISSION.json', receipt)
            write(args.freeze.parent/f'consumed_{digest(args.freeze)}.json', {
                'status': 'CONSUMED_BEFORE_FIT', 'freeze_sha256': digest(args.freeze),
                'output': str(args.output.resolve()), 'created_at': datetime.now(UTC).isoformat(),
                'rule': 'Retain after failure; a validity repair requires a new recorded freeze.'})
            # The marker distinguishes a failure during fitting from no fitting.
            write(args.output/'FIT_START.json', {'maximum_fits': 2, 'started_at': datetime.now(UTC).isoformat()})
            market_fits = None
            summary = study(data, args.output)
            market_fits = summary['fits']
            report(summary, args.output)
            verify(args.root, args.freeze); guard()
            result = {'status': summary['status'], 'ledgers': summary['ledgers'],
                      'horizon_disposition': summary['horizon_disposition'],
                      'investment_gate_6pct_both_folds': summary['investment_gate_6pct_both_folds']}
    except Exception as exc:  # noqa: BLE001 -- retain unexpected CLI failures with a nonzero exit.
        signal.setitimer(signal.ITIMER_REAL, 0)
        code = 1
        result = {'status': 'FAILED', 'error': str(exc), 'type': type(exc).__name__}
        (args.output/'failure.log').write_text(traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    result.update(completed_at=datetime.now(UTC).isoformat(), exit_code=code,
                  market_fits=market_fits, seconds=time.monotonic()-started, peak_rss_mib=rss_mib(),
                  protected_access=False, confirmatory=False, independent_reproduction=False)
    write(args.output/'RECEIPT.json', result)
    print(json.dumps(result, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
