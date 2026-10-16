"""Validate the real-price dataset and freeze all cells before market fitting."""
import argparse
import json
import math
import resource
import platform
import signal
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from .prepare import stamp, sha, write, DAY


def freeze(root, output):
    # Bounded data-integrity work; this does not authorize any model fit.
    started = time.monotonic()
    def guard(*_):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if platform.system() == 'Darwin' else 1024)
        if rss > 128 or time.monotonic()-started > 60:
            raise RuntimeError('128MiB/60s integrity ceiling exceeded')
    signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, .05, .05)
    try:
        from successors.frem_s6.data import load_dataset
        from successors.frem_s6.experiment import planned_cells, validate_protocol
        from successors.frem_s6.check import source_identities
        import numpy as np
        source = json.loads((root / 'acquisition.json').read_text())
        for item in source['sources']:
            if sha(root / (item['symbol']+'.json')) != item['retained_sha256']:
                raise ValueError('acquisition identity mismatch')
        admitted = load_dataset(root/'windows.json')
        data = {**admitted['metadata'], **admitted['partitions']}
        # An explicit calendar inventory, not a shifted-timestamp workaround.
        periods = [('2007-12-01','2012-01-01'), ('2015-01-01','2017-01-01'),
                   ('2020-01-01','2021-01-01'), ('2022-01-01','2025-01-01')]
        for t in data['evaluation']['origin_times']:
            if any(stamp(a) <= t < stamp(b) for a,b in periods):
                raise ValueError('evaluation overlaps spent calendar interval')
        cal = data['calibration']
        # Precision proxy uses calibration returns only, never evaluation errors.
        daily = [float(np.mean(cal['y'][cal['origin_times']==t])) for t in np.unique(cal['origin_times'])]
        sd = float(np.std(daily, ddof=1))
        n = len(np.unique(data['evaluation']['origin_times']))
        protocol = dict(protocol_id='S6-R1-ETF-PRICE-20261014', status='FROZEN_DEVELOPMENT',
          created_at=datetime.now(timezone.utc).isoformat(), data_sha256=sha(root/'windows.json'),
          cells=24, seeds=[11,29,47], protected_start=stamp('2025-01-01'), with_xgboost=True,
          cell_configurations={name:asdict(config) for name,config in planned_cells(1,with_xgboost=True)},
          fold_dates={p:np.unique(data[p]['origin_times']).tolist() for p in ('train','calibration','evaluation')},
          fold_rationale='Predeclared train 2005-2014, calibration 2015-2016, outer development 2017-2018; future training targets and calibration outcomes purged at next boundary. 2019+ not downloaded. Prior other-asset training exposure means this is not untouched confirmation.',
          precision_analysis={'required_evaluation_dates':48,'block_dates':[3,6],
             'development_effect_analysis':'84 planned qualifying weeks; both fixed blocks required. No NLL power claim from return variance; exploratory intervals only, no efficacy certification. At most 14 six-date blocks: weak precision and regime coverage.',
             'calibration_date_mean_return_sd':sd, 'return_mean_80pct_power_proxy_at_six_date_effective_blocks':2.8*sd/math.sqrt(n/6),
             'proxy_assumption':'Heuristic normal mean-test proxy, six-date dependence, not power for mixture-NLL differences or funding probability.'},
          spent_date_inventory={'source':'configs/public_equity_crises.json; configs/frem_s3_public.json; closed S3/S4/S5 and portfolio successors. Conservatively excludes all 2023-2024, not merely exact old month indices.',
             'calendar_intervals_exclusive_end':periods,
             'dates':[t+12*3600 for a,b in periods for t in range(stamp(a),stamp(b),DAY)]},
          primary_endpoint='mean_date_level_mixture_nll', economic_policy='shared_fixed_80_20_signal',
          primary_family='full_width8', primary_control='equal',
          decision='Report all cells, seeds, controls, block lengths and costs. No selection from outer outcomes. Negative/inconclusive results remain closed. A material improvement requires a new hypothesis and freeze, using an unconsumed evaluation period.',
          limitations=['Current-vintage, not historical PIT-certified.', 'Fixed three surviving ETFs, no global eligibility claim.',
             'Unadjusted price returns exclude dividends; split events rejected.',
             'Weekly opens occur after research issue/maturity coordinates; outcomes available at Monday16UTC, not at noon.',
             'Historical publication timestamps assumed conservatively, not observed.',
             'Holiday gaps produce included-episode diagnostics, not continuous investable wealth.',
             'No causal claim, mathematical profit proof, market manipulation, protected confirmation, or Laura investment recommendation.'])
        validate_protocol(protocol,data,with_xgboost=True)
        output.mkdir(parents=True,exist_ok=False)
        write(output/'protocol.json',protocol)
        identities=source_identities()
        identities.update({str(p):sha(p) for p in sorted(Path('successors/frem_s6_real').glob('*.py'))})
        write(output/'source_hashes.json',identities)
        write(output/'validation.json',{'status':'REAL_DATA_CONTRACT_VALIDATED_AND_PROTOCOL_FROZEN',
          'rows':{p:len(data[p]['assets']) for p in ('train','calibration','evaluation')},
          'protocol_sha256':sha(output/'protocol.json'),'data_sha256':protocol['data_sha256'],
          'market_fits':0,'protected_access':False,'scope':'Structural and provenance-binding checks, not historical source certification or model efficacy.'})
        guard()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    freeze(a.root,a.output)
