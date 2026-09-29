#!/usr/bin/env python3
"""Independently recompute saved scores and provenance; never edit raw results."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from wharton_lab.models.m11.accounting import simulate_ledger


def tail(loss,alpha=.95):
    # Independent CVaR representation: minimize the piecewise-linear eta formula.
    loss=np.asarray(loss)
    return float(min(eta+np.maximum(loss-eta,0).mean()/(1-alpha) for eta in np.unique(loss)))

def score(metric,p,y,raw):
    if metric=='mean_pinball':
        q=raw['quantiles']; error=y[:,None]-p
        return np.maximum(q*error,(q-1)*error).mean(axis=1)
    if metric=='negative_mean_spearman':
        return -np.array([spearmanr(a,b).statistic for a,b in zip(y,p)])
    if metric=='mse_normalized_return':
        error=(p-y)**2
        return error if error.ndim==1 else error.mean(axis=tuple(range(1,error.ndim)))
    if metric=='frobenius_error_normalized_covariance':
        return np.sqrt(np.square(p-y).sum(axis=(1,2)))
    if metric=='energy_score_normalized_path':
        return np.array([cdist(a,b[None]).mean()-.5*cdist(a,a).sum()/(len(a)*(len(a)-1)) for a,b in zip(p,y)])
    if metric=='cvar95_unpaid_currency':
        return simulate_ledger(p,y,1,np.zeros(5),raw['liabilities'],.0005)['unpaid'].sum(axis=1)
    raise ValueError(metric)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory',type=Path); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); root=args.directory
    rows=json.loads((root/'results.json').read_text()); protocol=json.loads((root/'protocol_snapshot.json').read_text())
    failures=[]; count=0; maximum_error=0
    for row in rows:
        if row['status']!='completed': failures.append({'cell':row['cell_id'],'error':'failed execution'}); continue
        try:
            artifact=root/row['artifact']
            assert hashlib.sha256(artifact.read_bytes()).hexdigest()==row['artifact_sha256'],'artifact hash'
            assert row['source_fingerprint']==protocol['source_fingerprint'],'source mismatch'
            with np.load(root/'datasets'/f'{row["world"]}_{row["seed"]}.npz',allow_pickle=False) as data:
                returns=data['returns']; scale=float(data['scale'])
                assert hashlib.sha256(np.ascontiguousarray(returns).tobytes()).hexdigest()==row['dataset_sha256'],'dataset hash'
            with np.load(artifact,allow_pickle=False) as raw:
                origins=raw['test_origins']; train=raw['train_origins']
                assert train.max()+5-1<240 and origins.min()>=240,'horizon overlap'
                # Independently validate target alignment for each task.
                metric=row['primary_metric']; truth=raw['truth']
                if row['model'] in ('M01','M04','M05','M06','M09','M12'):
                    expected=returns[origins,0]/scale
                elif row['model'] in ('M03','M08'):
                    expected=returns[origins]/scale
                elif row['model']=='M02':
                    normalized=returns[origins]/scale
                    expected=normalized-raw['beta_test']*normalized.mean(axis=1)[:,None]
                    # Recompute exposures using only the actual trailing observed history.
                    beta=[]
                    for t in origins:
                        x=returns[t-20:t]/scale; m=x.mean(axis=1); m-=m.mean()
                        beta.append(((x-x.mean(axis=0))*m[:,None]).sum(axis=0)/max(float(m@m),1e-10))
                    np.testing.assert_allclose(raw['beta_test'],beta,atol=1e-10)
                elif row['model'] in ('M07','M10'):
                    future=np.array([returns[t:t+5]/scale for t in origins])
                    if row['model']=='M10': expected=future.reshape(len(origins),-1)
                    else: expected=np.array([np.cov(x,rowvar=False,ddof=1) for x in future])
                elif row['model']=='M11': expected=np.array([returns[t:t+5] for t in origins])
                np.testing.assert_allclose(truth,expected,rtol=1e-10,atol=1e-10)
                for field,loss_name,value_name in [('prediction','model_losses','primary_value'),('baseline_prediction','baseline_losses','baseline_value')]:
                    losses=score(metric,raw[field],truth,raw)
                    np.testing.assert_allclose(losses,raw[loss_name],rtol=1e-9,atol=1e-10)
                    recomputed=tail(losses) if metric=='cvar95_unpaid_currency' else float(losses.mean())
                    error=abs(recomputed-row[value_name]); maximum_error=max(maximum_error,error)
                    assert error<1e-8,'reported score mismatch'
                if row['model']=='M11':
                    assert np.max(np.abs(raw['identity_residual']))<1e-9,'ledger identity failed'
                    pool={np.ascontiguousarray(returns[t:t+5]).tobytes() for t in train}
                    assert all(np.ascontiguousarray(x).tobytes() in pool for x in raw['scenario_returns']),'non-training bootstrap path'
            count+=1
        except Exception as exc:
            failures.append({'cell':row['cell_id'],'error':f'{type(exc).__name__}: {exc}'})
    expected=protocol['expected_cells']
    report={'verified_cells':count,'expected_cells':expected,'failures':failures,'maximum_score_discrepancy':maximum_error,
            'verified':not failures and count==expected,'source_fingerprint':protocol['source_fingerprint'],
            'checks':['artifact SHA256','dataset SHA256','source fingerprint equality','matured horizon split',
                      'raw-return target alignment','baseline and model score recomputation','M11 wealth identity and train-only bootstrap']}
    args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    return 0 if report['verified'] else 1
if __name__=='__main__': raise SystemExit(main())
