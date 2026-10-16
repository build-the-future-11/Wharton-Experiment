"""Frozen three-seed S7 development execution; no outcome-selected reruns."""
import json
import hashlib
import resource
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from successors.frem_s6.model import S6Forecaster, ModelConfig, MixtureForecast, student_logpdf
from successors.frem_s6.experiment import ensemble_records, paired_date_interval
from successors.frem_portfolio_v1.engine import Forecast, backtest, rebalance, target_weights
from successors.frem_s6_real.prepare import write, sha
from .data import validate_dataset
from .router import MixtureMemory, select_strength


def cost_aware(periods, cost_bps):
    """Finite targets: hold/cash/80%-anchor/signal; marginal log utility pays two-way fees.

    Uses diagonal variance approximation, not a calibrated return expectation.
    Ex ante criterion fixed at mean log return - 4*variance - roundtrip cost.
    """
    names=list(periods[0]['returns']);holdings=dict.fromkeys(names,0.);cash=100000.;rows=[]
    rate=cost_bps/10000
    for p in periods:
        nav=cash+sum(holdings.values());current={a:holdings[a]/nav for a in names}
        forecasts=p['forecasts']
        def utility(w):return sum(w[f.asset]*f.location-4*w[f.asset]**2*f.variance for f in forecasts)
        best=current;best_value=utility(current)
        candidates=[dict.fromkeys(names,0.),dict.fromkeys(names,.8/len(names)),target_weights(forecasts)]
        for w in candidates:
            score=utility(w)-2*rate*sum(abs(w[a]-current[a]) for a in names)
            if score>best_value+1e-12:best,best_value=w,score
        holdings,cash,trade=rebalance(holdings,cash,best,rate,no_trade_band=0)
        contribution={a:holdings[a]*p['returns'][a] for a in names}
        for a in names:holdings[a]*=1+p['returns'][a]
        rows.append({'time':p['time'],'end':p['end'],'nav_start':nav,'nav_end':cash+sum(holdings.values()),
                     'fee':trade['fee'],'notional':trade['notional'],'contributions':contribution,
                     'cash':cash,'holdings':dict(holdings)})
    liquidation=rate*sum(holdings.values());terminal=cash+sum(holdings.values())-liquidation
    return {'terminal_wealth':terminal,'net_return':terminal/100000-1,'fees':sum(r['fee'] for r in rows)+liquidation,
            'liquidation_fee':liquidation,'periods':rows}


def period(r):
    arrays=[np.asarray(r[k]) for k in ('mean','scale','df','weights')]
    f=MixtureForecast(*arrays,tuple(map(str,range(arrays[0].shape[1]))));mu,var=f.log_return_moments()
    return {'time':r['date'],'end':r['end'],'currency':'USD','forecasts':[Forecast(a,float(m),float(v)) for a,m,v in zip(r['assets'],mu,var)],'returns':dict(zip(r['assets'],r['simple_returns']))}


def run(data, output):
    families=[]
    for seed in (11,29,47):
        folder=output/f'seed{seed}';folder.mkdir()
        model=S6Forecaster(ModelConfig(1,seed=seed,use_xgboost=True))
        train,cal,evaluation=(data['partitions'][p] for p in ('train','calibration','evaluation'))
        model.fit(train['windows'],train['future_windows'],train['y'],origin_times=train['origin_times'],label_available=train['available_times'],future_available=train['future_available'].max(1),fit_cutoff=data['metadata']['fit_cutoff'])
        split=1514764800 # 2018-01-01; scale fitting only on fully mature 2017 outcomes
        early=cal['available_times']<split
        model.calibrate(cal['windows'][early],cal['y'][early],origin_times=cal['origin_times'][early],maturity_times=cal['maturity_times'][early],available_times=cal['available_times'][early])
        memory=MixtureMemory(len(model.names))
        for date in np.unique(cal['origin_times'][early]):
            idx=(cal['origin_times']==date)&early
            keys,means,scales=model._experts(cal['windows'][idx])
            memory.add(keys.mean(0),student_logpdf(cal['y'][idx,None],means,scales,model.config.df),cal['available_times'][idx].max())
        gains={a:[] for a in (.25,.5,1.)};raw=[];alpha=None
        for stage,part in [('selection',cal),('evaluation',evaluation)]:
            if stage=='evaluation':
                alpha=select_strength(gains);write(folder/'selection.json',{'alpha':alpha,'gains':gains});model.save(folder/'fit_checkpoint.json')
            dates=np.unique(part['origin_times'])
            if stage=='selection':dates=dates[dates>max(split,model.calibration_cutoff)]
            for date in dates:
                idx=part['origin_times']==date;assets=np.asarray(part['assets'])[idx].tolist()
                old={r:model.forecast(part['windows'][idx],issue_time=float(date),asset_ids=assets,router=r) for r in ('memory','equal','recent')}
                base=old['equal'];keys,_,_=model._experts(part['windows'][idx]);key=keys.mean(0)
                local=memory.weights(key,date);global_w=memory.weights(key,date,contextual=False)
                def dist(w):return MixtureForecast(base.mean,base.scale,base.df,np.broadcast_to(w,base.mean.shape).copy(),base.names)
                if stage=='selection':
                    off=base.nll(part['y'][idx]).mean()
                    for a in gains:gains[a].append(float(off-dist((1-a)/len(local)+a*local).nll(part['y'][idx]).mean()))
                else:
                    forecasts={**old,'s7':dist((1-alpha)/len(local)+alpha*local),'mixture_ungated':dist(local),'mixture_global':dist(global_w)}
                    for j,name in enumerate(model.names):forecasts[name]=MixtureForecast(base.mean[:,j:j+1],base.scale[:,j:j+1],base.df[:,j:j+1],np.ones((len(assets),1)),(name,))
                    for name,f in forecasts.items():
                        nll=f.nll(part['y'][idx]);raw.append({'date':float(date),'end':float(part['maturity_times'][idx][0]),'assets':assets,'y':part['y'][idx].tolist(),'simple_returns':part['simple_returns'][idx].tolist(),'variant':name,'mean':f.mean.tolist(),'scale':f.scale.tolist(),'df':f.df.tolist(),'weights':f.weights.tolist(),'nll':nll.tolist(),'date_nll':float(nll.mean())})
                # Both mechanisms receive the same labels, with strict maturity filtering.
                density=student_logpdf(part['y'][idx,None],base.mean,base.scale,base.df)
                memory.add(key,density,part['available_times'][idx].max())
                model.observe(issue_time=float(date),y=part['y'][idx],asset_ids=assets,maturity_times=part['maturity_times'][idx],available_times=part['available_times'][idx])
            if stage=='evaluation':
                write(folder/'predictions.json',raw);model.save(folder/'online_checkpoint.json')
                write(folder/'router_state.json',{'records':[{'available':t,'key':k.tolist(),'density':v.tolist()} for t,k,v in memory.records]})
        write(folder/'receipt.json',{'status':'COMPLETE','seed':seed,'alpha':alpha,'rows':len(raw),'config':asdict(model.config)})
        families.append(raw)
    combined=ensemble_records(families);write(output/'ensembles.json',combined)
    variants=sorted({r['variant'] for r in combined});comparisons={};economics={}
    selected=[r['date_nll'] for r in combined if r['variant']=='s7']
    for v in variants:
        baseline=[r['date_nll'] for r in combined if r['variant']==v]
        comparisons[v]={str(b):paired_date_interval(np.array(baseline)-selected,block_dates=b) for b in (3,6)}
        periods=[period(r) for r in combined if r['variant']==v]
        economics[v]={str(cost):{'signal':backtest(periods,cost_bps=cost),'cost_aware':cost_aware(periods,cost)} for cost in (0,10,20,40)}
    periods=[period(r) for r in combined if r['variant']=='equal']
    benchmarks={v:{str(cost):backtest(periods,policy=v,cost_bps=cost) for cost in (0,10,20,40)} for v in ('cash','buy_hold','equal_weight')}
    write(output/'comparisons.json',comparisons);write(output/'economics.json',economics);write(output/'benchmarks.json',benchmarks)
    write(output/'completion.json',{'status':'COMPLETE','base_fits':3,'seeds':[11,29,47],'variants':variants,'protected_access':False,'scope':'2019 retrospective continuous-week development; dividends accrued/reinvested weekly, not actual fills or independently unseen history'})


def main():
    import argparse,traceback
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:
        frozen=json.loads(a.freeze.read_text())
        for name,digest in frozen['sources'].items():
            if sha(name)!=digest:raise ValueError('source drift: '+name)
        # Provider data downloaded after protocol freeze, bound before numerical fitting.
        write(a.output/'data_identity.json',{'sha256':sha(a.root/'windows.json'),'protocol_sha256':sha(a.freeze)})
        data=validate_dataset(json.loads((a.root/'windows.json').read_text()))
        if min(data['partitions']['evaluation']['origin_times'])<1546300800 or max(data['partitions']['evaluation']['available_times'])>=1577836800:
            raise ValueError('evaluation must be exclusively 2019')
        run(data,a.output)
        for name,digest in frozen['sources'].items():
            if sha(name)!=digest:raise ValueError('source drift during run')
    except BaseException:
        write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise
    finally:
        write(a.output/'resources.json',{'seconds':time.monotonic()-started,'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__=='__main__':main()
