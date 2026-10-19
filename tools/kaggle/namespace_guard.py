"""Explicit namespace-only admission for the fixed generated Kaggle smoke.

Hidden ancestors are UNKNOWN. This never returns production host PASS.
The unchanged production/full-root contract remains in the sealed package.
"""
import hashlib
import json
import os
import platform
import resource
import runpy
import sys
import time
from pathlib import Path

CONTRACT = 's10-kaggle-namespace-smoke-v1'
ADDRESS_BYTES = 1024 * 2**20
RESERVE_BYTES = 512 * 2**20
COMMANDS = {'smoke', 'smoke-resume', 'smoke-evaluate', 'smoke-export'}
MANIFEST_SHA256 = 'e576f5fa916730ea1f1a8d6e29772455e7b0f64803395643cba1afae75e93eba'


def text(path):
    if path.is_symlink():
        raise RuntimeError('resource symlink rejected')
    with path.open('rb') as f:
        raw = f.read(65537)
    if len(raw) > 65536:
        raise RuntimeError('oversized resource file')
    return raw.decode('ascii')


def integer(raw):
    raw = raw.strip()
    if not raw.isascii() or not raw.isdigit() or len(raw) > 20:
        raise RuntimeError('finite nonnegative resource counter required')
    return int(raw)


def pairs(raw):
    result = {}
    for line in raw.splitlines():
        key, value = line.split()
        if key in result:
            raise RuntimeError('duplicate counter')
        result[key] = integer(value)
    return result


def sample(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    membership = text(proc / 'self/cgroup').strip()
    if membership != '0::/':
        raise RuntimeError('only observed namespace-root layout supported')
    mounts = [line for line in text(proc / 'self/mountinfo').splitlines() if ' - cgroup2 ' in line]
    if len(mounts) != 1 or mounts[0].split()[3:5] != ['/', str(cgroup)]:
        raise RuntimeError('exact namespace mount required')
    if 'memory' not in text(cgroup / 'cgroup.controllers').split():
        raise RuntimeError('memory controller required')
    maximum = integer(text(cgroup / 'memory.max'))
    current = integer(text(cgroup / 'memory.current'))
    high_raw = text(cgroup / 'memory.high').strip()
    high = maximum if high_raw == 'max' else integer(high_raw)
    events = pairs(text(cgroup / 'memory.events'))
    if not {'high', 'max', 'oom', 'oom_kill'} <= events.keys():
        raise RuntimeError('pressure event counters required')
    vm = pairs(text(proc / 'vmstat'))
    rows = [line.split() for line in text(proc / 'meminfo').splitlines() if line.startswith('MemAvailable:')]
    if len(rows) != 1 or len(rows[0]) != 3 or rows[0][2] != 'kB':
        raise RuntimeError('MemAvailable unavailable')
    available = min(integer(rows[0][1]) * 1024, max(0, min(maximum, high) - current))
    if text(proc / 'self/cgroup').strip() != membership:
        raise RuntimeError('namespace changed during sampling')
    return {'monotonic': time.monotonic(), 'membership': membership, 'mount': mounts[0],
            'maximum': maximum, 'high': high, 'current': current, 'available_bytes': available,
            'pswpout': vm['pswpout'], 'events': events}


def assess(samples):
    if len(samples) != 3:
        raise RuntimeError('three samples required')
    for row in samples:
        if row['available_bytes'] < ADDRESS_BYTES + RESERVE_BYTES:
            raise RuntimeError('visible headroom must cover 1GiB worker plus 512MiB reserve')
    for a, b in zip(samples, samples[1:]):
        if b['monotonic'] - a['monotonic'] < 1:
            raise RuntimeError('samples must be spaced one second')
        for key in ('membership', 'mount', 'maximum', 'high', 'pswpout', 'events'):
            if a[key] != b[key]:
                raise RuntimeError('namespace or pressure changed: ' + key)
    return {'status': 'SMOKE_ONLY_ADMITTED', 'contract': CONTRACT, 'samples': samples,
            'worker_address_bytes': ADDRESS_BYTES, 'reserve_bytes': RESERVE_BYTES,
            'hidden_ancestor_constraints': 'UNKNOWN', 'production_admission': False,
            'scientific_status': 'NOT_EVALUATED'}


def admission():
    if platform.system() != 'Linux' or Path('/dev/nvidia0').exists():
        raise RuntimeError('Linux CPU only')
    rows = []
    for index in range(3):
        if index:
            time.sleep(1)
        rows.append(sample())
    return assess(rows)


def validate_command(args):
    if not args or args[0] not in COMMANDS:
        raise RuntimeError('only fixed generated smoke commands permitted')
    if args[0] in {'smoke', 'smoke-resume'}:
        if args.count('--manifest') != 1:
            raise RuntimeError('exact manifest required')
        path = Path(args[args.index('--manifest') + 1])
        # The sealed package independently rejects duplicate/nonfinite keys.
        value = json.loads(path.read_text())
        canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != MANIFEST_SHA256:
            raise RuntimeError('only original 12-origin manifest admitted')
    return args


def hard_limits():
    for key, ceiling in ((resource.RLIMIT_AS, ADDRESS_BYTES), (resource.RLIMIT_CPU, 75)):
        soft, hard = resource.getrlimit(key)
        bound = min([ceiling] + [v for v in (soft, hard) if v != resource.RLIM_INFINITY])
        resource.setrlimit(key, (bound, bound))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main():
    if sys.argv[1:] == ['check']:
        print(json.dumps(admission()))
        return
    if sys.argv[1:2] != ['exec']:
        raise RuntimeError('check or exec required')
    args = validate_command(sys.argv[2:])
    hard_limits()
    # A fresh check inside the worker closes the gap after the wrapper check.
    check = admission()
    print(json.dumps({'worker_resource_contract': check, 'limits_applied': True}), flush=True)
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.argv = ['frem_s10', *args]
    runpy.run_module('frem_s10', run_name='__main__')


if __name__ == '__main__':
    main()
