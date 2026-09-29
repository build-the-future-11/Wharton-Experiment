#!/usr/bin/env python3
"""Seed-paired exploratory summaries, preserving all methods and worlds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


def interval(x):
    x=np.asarray(x,dtype=float); n=len(x)
    if n<2: raise ValueError('At least two independent seeds required')
    mean=float(x.mean()); half=float(t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n))
    return mean,mean-half,mean+half


def main():
    p=argparse.ArgumentParser(); p.add_argument('directory',type=Path); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.directory/'results.json').read_text())
    if any(r['status']!='completed' for r in rows): raise ValueError('Failed cells must be resolved or explicitly analysed before summary')
    frame=pd.DataFrame(rows); summary=[]; seeds=[]
    for (mid,world),block in frame.groupby(['model','world'],sort=False):
        full=block[block.variant=='full'].sort_values('seed'); ab=block[block.variant!='full'].sort_values('seed')
        assert np.array_equal(full.seed,ab.seed),'Unpaired ablation seeds'
        dv=full.primary_value.to_numpy()-full.baseline_value.to_numpy()
        av=full.primary_value.to_numpy()-ab.primary_value.to_numpy()
        mean,lo,hi=interval(dv); am,al,ah=interval(av)
        mm,ml,mh=interval(full.primary_value); bm,bl,bh=interval(full.baseline_value); arm,arl,arh=interval(ab.primary_value)
        item={'model':mid,'world':world,'metric':full.iloc[0].primary_metric,'baseline':full.iloc[0].baseline_name,
              'ablation':ab.iloc[0].variant,'n_seeds':len(full),'model_mean':mm,'model_ci_low':ml,'model_ci_high':mh,
              'baseline_mean':bm,'baseline_ci_low':bl,'baseline_ci_high':bh,
              'paired_mean':mean,'paired_ci_low':lo,'paired_ci_high':hi,
              'ablation_mean':arm,'ablation_ci_low':arl,'ablation_ci_high':arh,
              'full_minus_ablation':am,'ablation_effect_ci_low':al,'ablation_effect_ci_high':ah,
              'model_fits_per_cell':60 if mid=='M03' else 1,
              'mean_elapsed_seconds':float(full.elapsed_seconds.mean())}
        for key in sorted(set().union(*(x.keys() for x in full.extra_metrics))):
            values=[x.get(key) for x in full.extra_metrics]
            if all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in values):
                item['extra_'+key]=float(np.mean(values))
        summary.append(item)
        for (_,row),(_,ablrow) in zip(full.iterrows(),ab.iterrows()):
            seeds.append({'model':mid,'world':world,'seed':int(row.seed),'full':row.primary_value,
                          'baseline':row.baseline_value,'ablation':ablrow.primary_value,
                          'full_minus_baseline':row.primary_value-row.baseline_value,
                          'full_minus_ablation':row.primary_value-ablrow.primary_value})
    out=pd.DataFrame(summary); out.to_csv(args.output/'summary.csv',index=False)
    pd.DataFrame(seeds).to_csv(args.output/'paired_seeds.csv',index=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(out[['model','world','paired_mean','full_minus_ablation']].to_string(index=False))
if __name__=='__main__': main()
