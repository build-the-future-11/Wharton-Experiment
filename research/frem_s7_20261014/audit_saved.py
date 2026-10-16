"""Independent stdlib density/ledger/raw-price audit; no model imports or fits."""
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path('outputs/frem_s7_2019_20261014_attempt1')
read=lambda path:json.loads(Path(path).read_text())


def main():
    frozen=read('research/frem_s7_20261014/freeze2.json')
    for name,digest in frozen['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    predecessor=read('research/frem_s6_real_20261014/freeze2/source_hashes.json')
    for name,digest in predecessor.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    model_variants={'memory','equal','recent','s7','mixture_ungated','mixture_global','zero','ridge','trend','adaptive_volatility','reservoir_jepa','xgboost_hist'}
    count=0;maxerr=0.
    for rows in [read(ROOT/f'seed{s}/predictions.json') for s in (11,29,47)]+[read(ROOT/'ensembles.json')]:
        assert len(rows)==51*12 and len({(r['date'],r['variant']) for r in rows})==len(rows)
        assert {r['variant'] for r in rows}==model_variants
        for r in rows:
            vals=[]
            for y,mus,scales,dfs,ws,nll in zip(r['y'],r['mean'],r['scale'],r['df'],r['weights'],r['nll'],strict=True):
                assert abs(sum(ws)-1)<1e-10 and all(w>=0 for w in ws)
                logs=[math.log(w)+math.lgamma((df+1)/2)-math.lgamma(df/2)-.5*math.log(df*math.pi)-math.log(s)-(df+1)/2*math.log1p(((y-mu)/s)**2/df) for mu,s,df,w in zip(mus,scales,dfs,ws,strict=True) if w]
                a=max(logs);value=-a-math.log(sum(math.exp(x-a) for x in logs));vals.append(value);maxerr=max(maxerr,abs(value-nll));count+=1
            maxerr=max(maxerr,abs(sum(vals)/len(vals)-r['date_nll']))
    assert maxerr<1e-10
    econ=read(ROOT/'economics.json');bench=read(ROOT/'benchmarks.json');res=0.;ledgers=0
    for cs in list(econ.values())+[{cost:{'signal':l} for cost,l in cs.items()} for cs in bench.values()]:
        for cost,policies in cs.items():
            for policy,l in policies.items():
                previous=100000.
                for r in l['periods']:
                    res=max(res,abs(previous-r['nav_start']),abs(r['fee']-r['notional']*int(cost)/10000),abs(sum(r['holdings'].values())+r['cash']-r.get('liquidation_fee',0)-r['nav_end']))
                    if 'contributions' in r:res=max(res,abs(r['nav_start']-r['fee']+sum(r['contributions'].values())-r['nav_end']))
                    previous=r['nav_end']
                res=max(res,abs(previous-l.get('liquidation_fee',0)-l['terminal_wealth']));ledgers+=1
    assert res<1e-7
    data=read('data/s7_etf_2019_20261014/windows.json');reconciled=0;maxreturn=0.
    def day(t):return datetime.fromtimestamp(t,timezone.utc).date().isoformat()
    for asset in ('SPY','TLT','GLD'):
        raw=read(f'data/s7_etf_2019_20261014/{asset}.json');prices={day(r['timestamp']):r['open'] for r in raw['rows']};divs=[(day(d['date']),d['amount']) for d in raw['dividends'].values()]
        for part in data['partitions'].values():
            for a,start,end,ret,y in zip(part['assets'],part['origin_times'],part['maturity_times'],part['simple_returns'],part['y'],strict=True):
                if a!=asset:continue
                s,e=day(start),day(end);expected=(prices[e]+sum(v for d,v in divs if s<d<=e))/prices[s]-1
                maxreturn=max(maxreturn,abs(expected-ret),abs(math.expm1(y)-ret));reconciled+=1
    assert maxreturn<1e-12
    result={'status':'PASS','source_identities_verified':len(frozen['sources']),'S6_frozen_identities_unchanged':len(predecessor),'nll_values_recomputed':count,'max_nll_error':maxerr,'ledgers_verified':ledgers,'max_ledger_residual':res,'raw_price_dividend_labels_reconciled':reconciled,'max_return_error':maxreturn,'scope':'Arithmetic/provenance reconciliation, not independent retraining, provider verification, or future performance guarantee'}
    path=Path('research/frem_s7_20261014/AUDIT.json')
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
