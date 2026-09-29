#!/usr/bin/env python3
"""Fresh-process default-seed replay: compares every raw array, not just scores."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
r=a.report; out=r/'replay'
if out.exists(): raise FileExistsError('Refusing to overwrite replay')
with (r/'replay.log').open('w') as f:
    subprocess.run([sys.executable,'scripts/run_portfolio_research.py','--output',str(out),'--worlds','null','--seeds','2026092901'],stdout=f,stderr=subprocess.STDOUT,check=True)
rows=json.loads((out/'results.json').read_text()); checks=[]
for row in rows:
    with np.load(out/row['artifact'],allow_pickle=False) as x,np.load(r/'production'/row['artifact'],allow_pickle=False) as y:
        okay=sorted(x.files)==sorted(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
        checks.append({'cell_id':row['cell_id'],'all_arrays_bitwise_identical':bool(okay)})
report={'replayed_cells':len(checks),'bitwise_identical_cells':sum(x['all_arrays_bitwise_identical'] for x in checks),'all_identical':all(x['all_arrays_bitwise_identical'] for x in checks),'checks':checks}
(r/'replay_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='checks'})
if not report['all_identical']: raise SystemExit(1)
