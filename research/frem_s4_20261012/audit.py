"""Independent standard-library arithmetic and completeness audit of S4."""
import csv
import hashlib
import json
import math
from pathlib import Path

root=Path.cwd();run=root/'outputs/frem_s4_20261012_attempt1'
freeze=json.loads((run/'freeze.json').read_text())
for path,digest in freeze['sources'].items():
 assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
complete=json.loads((run/'completion.json').read_text());count=0;error=0.;scores={}
for record in complete['records']:
 cell=run/record['fold']/str(record['seed']);paired=None;scores[str(cell.relative_to(run))]={}
 for name,result in record['results'].items():
  p=cell/f'{name}.csv';assert hashlib.sha256(p.read_bytes()).hexdigest()==result['sha256']
  rows=list(csv.DictReader(p.open()));assert len(rows)==result['rows'];identities=[];dates={}
  for r in rows:
   y,mu,s,nu=[float(r[k]) for k in ['actual','prediction','scale','student_df']]
   nll=math.log(s)+.5*math.log(nu*math.pi)+math.lgamma(nu/2)-math.lgamma((nu+1)/2)+(nu+1)/2*math.log1p(((y-mu)/s)**2/nu)
   error=max(error,abs(nll-float(r['nll'])));count+=1
   identities.append((r['time'],r['asset_id'],r['actual'],r['label_time']))
   dates.setdefault(r['time'],[]).append(nll)
  assert len(set((x[0],x[1]) for x in identities))==len(identities)
  if paired is None:paired=identities
  assert paired==identities
  score=sum(sum(v)/len(v) for v in dates.values())/len(dates)
  assert abs(score-result['nll'])<1e-12
  scores[str(cell.relative_to(run))][name]=score
  selection=json.loads((cell/f'{name}_selection.json').read_text());chosen=selection['selected']
  assert chosen['weight']==0 or min(chosen['half_gains'])>.01
  state=json.loads((cell/f'{name}_state.json').read_text());assert all(r['version']==state['version'] for r in state['records'])
assert len(complete['records'])==6 and count==3168
assert error<1e-12
receipt={'status':'VERIFIED','rows':count,'max_nll_error':error,'frozen_hashes':len(freeze['sources']),'fits':6,'prediction_cells':24,'paired_outcomes':True,'gate_rule_checked':True,'scope':'Score/source/completeness audit, not independent retraining','scores':scores}
with (root/'research/frem_s4_20261012/audit.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps({k:v for k,v in receipt.items() if k!='scores'}))
