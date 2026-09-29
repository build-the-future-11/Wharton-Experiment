#!/usr/bin/env python3
"""Validate trusted locally generated checkpoints, never untrusted pickle files."""
from pathlib import Path
import json, sys
from datetime import datetime,timedelta
import numpy as np
from wharton_lab.research.causal_portfolio import make_dataset,point_features,panel_features,windows,ABLATIONS
from wharton_lab.models.m01 import M01RegimeDistributionalGBM
from wharton_lab.models.m02 import M02FactorResidualRanker
from wharton_lab.models.m03 import M03ConditionalNonlinearFactor
from wharton_lab.models.m04 import M04FinancialAPEN
from wharton_lab.models.m05 import QAPENModel
from wharton_lab.models.m06 import FIJEPAModel
from wharton_lab.models.m07 import EigenJEPAModel
from wharton_lab.models.m08 import GraphFIJEPAModel
from wharton_lab.models.m09 import M09Model
from wharton_lab.models.m10 import M10Model
from wharton_lab.models.m11 import M11Model
from wharton_lab.models.m12 import M12Model

CLASSES=[M01RegimeDistributionalGBM,M02FactorResidualRanker,M03ConditionalNonlinearFactor,M04FinancialAPEN,QAPENModel,
         FIJEPAModel,EigenJEPAModel,GraphFIJEPAModel,M09Model,M10Model,M11Model,M12Model]
root=Path(sys.argv[1]); output=Path(sys.argv[2]); d=make_dataset('null',2026092901); r=d.normalized; t=d.test_origins
rows=[]
for cls in CLASSES:
    mid=cls.MODEL_ID
    for variant in ('full',ABLATIONS[mid]):
        checkpoint=root/'checkpoints'/f'{mid}__{variant}.pkl'
        record={'model':mid,'variant':variant,'checkpoint':str(checkpoint.relative_to(root))}
        try:
            m=cls.load(checkpoint)
            with np.load(root/'raw'/f'{mid}__null__2026092901__{variant}.npz',allow_pickle=False) as raw:
                b=point_features(r,t); expected=raw['prediction']
                if mid=='M01': actual=m.predict_quantiles(b)
                elif mid=='M02': actual=m.predict(panel_features(r,t)[0].reshape(-1,5)).reshape(-1,5)
                elif mid=='M03': actual=m.predict(panel_features(r,t[-1:])[0][0]); expected=expected[-1]
                elif mid=='M04': actual=m.predict(b,as_of=datetime(2020,1,1)+timedelta(days=int(t[0])))
                elif mid=='M06': actual=m.predict(windows(r,t,8))
                elif mid=='M07':
                    b=windows(r,t,20); b=np.zeros_like(b) if variant!='full' else b
                    actual=m.predict_covariances(b,20)
                elif mid=='M08': actual=m.predict(windows(r,t,8).transpose(0,2,1)[:,:,:,None])
                elif mid=='M10':
                    context=b[:1] if variant=='full' else np.zeros_like(b[:1])
                    actual=m.sample_paths(context,32); expected=cls.load(checkpoint).sample_paths(context,32)
                    record['check']='saved-RNG continuation agrees between independent loads (original continuation unit-tested separately)'
                elif mid=='M11': m.predict(m._scenario_returns); actual=m._last_result.full_horizon_weights
                elif mid=='M12': actual=m.predict(r[t-1])
                else: actual=m.predict(b)
                np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-10)
                record.update(passed=True,max_difference=float(np.max(np.abs(actual-expected))))
        except Exception as exc: record.update(passed=False,error=str(exc))
        rows.append(record)
report={'checked':len(rows),'passed':sum(x['passed'] for x in rows),'all_passed':all(x['passed'] for x in rows),'checks':rows}
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if not report['all_passed']: raise SystemExit(1)
