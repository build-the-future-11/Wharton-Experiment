"""Portable, resource-gated execution of the frozen real-data S6 matrix.

Keeps the original S6 implementation immutable. Runs sequentially with a hard
parent timeout and cooperative RSS checks. Never bypasses source/data guards.
"""
import argparse
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from .prepare import sha, write


def snapshot():
    if platform.system() == 'Darwin':
        from scripts.public_equity_backtests import memory_snapshot
        return memory_snapshot()
    if platform.system() != 'Linux':
        raise RuntimeError('Only macOS and Linux resource admission supported')
    mem = {k:int(v.strip().split()[0]) for k,v in
           (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    available = mem['MemAvailable']/1024
    # Walk the current unified cgroup hierarchy, including constrained ancestors.
    relative = next((x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines()
                     if x.startswith('0::')), None)
    if relative is None:
        raise RuntimeError('cgroup v1 requires an explicit supported memory-limit adapter')
    base = Path('/sys/fs/cgroup')
    current = base / relative.lstrip('/')
    if not current.exists():
        current = base  # container cgroup namespace may already be rooted here
    while True:
        limit, used = current/'memory.max', current/'memory.current'
        if limit.exists():
            value = limit.read_text().strip()
            if value != 'max':
                available = min(available,(int(value)-int(used.read_text()))/2**20)
        if current == base:
            break
        if base not in current.parents:
            raise RuntimeError('invalid cgroup location')
        current = current.parent
    return {'available_mib':available,'pageouts':int(vm['pswpout']),
            'scope':'Linux MemAvailable capped by cgroup v2 headroom; pswpout pages'}


def admission():
    samples=[]
    for _ in range(3):
        samples.append(snapshot())
        time.sleep(1)
    return {'checked_at':datetime.now(timezone.utc).isoformat(),'samples':samples,
            'admitted':min(s['available_mib'] for s in samples)>=512
                       and samples[-1]['pageouts']<=samples[0]['pageouts']}


def verify(root, frozen):
    validation=json.loads((frozen/'validation.json').read_text())
    if sha(root/'windows.json')!=validation['data_sha256'] or sha(frozen/'protocol.json')!=validation['protocol_sha256']:
        raise ValueError('data or protocol changed after freeze')
    for name,digest in json.loads((frozen/'source_hashes.json').read_text()).items():
        if sha(name)!=digest:
            raise ValueError('frozen source changed: '+name)
    manifest=json.loads((root/'windows.json.manifest.json').read_text())
    if manifest['sha256']!=validation['data_sha256']:
        raise ValueError('manifest digest mismatch')


def worker(root,frozen,output):
    execution=json.loads((output/'execution_freeze.json').read_text())
    if sha(__file__)!=execution['runner_sha256'] or sha(root/'windows.json.manifest.json')!=execution['manifest_sha256']:
        raise ValueError('execution identity changed')
    verify(root,frozen)
    gate=admission()
    write(output/'worker_admission.json',gate)
    if not gate['admitted']:
        raise RuntimeError('fresh worker resource admission failed')
    from successors.frem_s6.data import load_dataset
    from successors.frem_s6.experiment import (planned_cells,validate_protocol,run_cell,
         ensemble_records,economics_from_records,paired_date_interval)
    data=load_dataset(root/'windows.json')
    data={**data['metadata'],**data['partitions']}
    protocol=json.loads((frozen/'protocol.json').read_text())
    validate_protocol(protocol,data,with_xgboost=True)
    import xgboost
    import numpy
    write(output/'environment.json',{'python':sys.version,'platform':platform.platform(),
          'numpy':numpy.__version__,'xgboost':xgboost.__version__})
    # Runtime integrity first, in a bounded subprocess on the admitted host.
    result=subprocess.run([sys.executable,'-m','successors.frem_s6.check','--output',str(output/'integrity')],
                          timeout=90,check=False)
    if result.returncode:
        raise RuntimeError('S6 numerical integrity tests failed')
    families,receipts={},{}
    for name,config in planned_cells(1,with_xgboost=True):
        rows,receipts[name]=run_cell(data,config,output/name)
        families.setdefault(name.split('_',1)[1],[]).append(rows)
    combined,comparisons,economics={},{},{}
    for name,rows in families.items():
        if len(rows)!=3:
            raise ValueError('incomplete seed family')
        combined[name]=ensemble_records(rows)
        economics[name]=economics_from_records(combined[name],data['currency'])
        candidate=[r['date_nll'] for r in combined[name] if r['variant']=='memory']
        comparisons[name]={}
        for control in sorted({r['variant'] for r in combined[name]}-{'memory'}):
            baseline=[r['date_nll'] for r in combined[name] if r['variant']==control]
            differences=[b-a for a,b in zip(candidate,baseline,strict=True)]
            comparisons[name][control]={str(b):paired_date_interval(differences,block_dates=b) for b in (3,6)}
    verify(root,frozen)
    write(output/'seed_ensembles.json',combined)
    write(output/'comparisons.json',comparisons)
    write(output/'ensemble_economics.json',economics)
    write(output/'completion.json',{'status':'FIXED_RETROSPECTIVE_DEVELOPMENT_MATRIX_COMPLETE',
          'fits':24,'receipts':receipts,'protected_access':False,'funding_certification':False,
          'scope':'Price-only included-week diagnostics, NOT continuous total-return portfolio performance.'})


def main():
    p=argparse.ArgumentParser()
    for name in ('root','frozen','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.worker:
        started=time.monotonic()
        def peak():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if platform.system()=='Darwin' else 1024)
        def guard(*_):
            if peak()>768 or time.monotonic()-started>7200:
                raise RuntimeError('768MiB/7200s experiment ceiling exceeded')
        signal.signal(signal.SIGALRM,guard)
        signal.setitimer(signal.ITIMER_REAL,.05,.05)
        try:
            worker(a.root,a.frozen,a.output)
            guard()
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            write(a.output/'resources.json',{'peak_rss_mib':peak(),'seconds':time.monotonic()-started})
        return
    a.output.mkdir(parents=True,exist_ok=False)
    verify(a.root,a.frozen)
    gate=admission()
    gate['market_fits_started']=0
    write(a.output/'admission.json',gate)
    if not gate['admitted']:
        print('BLOCKED: resource admission failed; zero market fits')
        return
    write(a.output/'execution_freeze.json',{'runner_sha256':sha(__file__),
          'manifest_sha256':sha(a.root/'windows.json.manifest.json'),
          'protocol_sha256':sha(a.frozen/'protocol.json')})
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    with (a.output/'worker.log').open('x') as log:
        try:
            subprocess.run([sys.executable,'-m','successors.frem_s6_real.run','--worker',
                '--root',str(a.root),'--frozen',str(a.frozen),'--output',str(a.output)],
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200,check=True)
        except Exception as exc:
            write(a.output/'failure.json',{'error':repr(exc)})
            raise


if __name__=='__main__':
    main()
