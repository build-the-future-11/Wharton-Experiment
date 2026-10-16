"""Retrospective continuous common-session weekly adapter; no 2020+ request."""
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from successors.frem_s6_real.prepare import ASSETS, DAY, sha, stamp, write


def date_of(t):
    return datetime.fromtimestamp(t, timezone.utc).date().isoformat()


def acquire(root):
    root.mkdir(parents=True, exist_ok=False)
    receipts = []
    for asset in ASSETS:
        url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{asset}'
               f'?period1={stamp("2019-01-01")}&period2={stamp("2020-01-01")}'
               '&interval=1d&events=div%2Csplits')
        request = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read(4000001)
            server_date = response.headers.get('Date')
        if len(raw)>4000000:
            raise ValueError('response too large')
        result = json.loads(raw)['chart']['result'][0]
        times = result['timestamp']
        if any(t<stamp('2019-01-01') or t>=stamp('2020-01-01') for t in times):
            raise ValueError('out-of-scope history')
        if result.get('events',{}).get('splits'):
            raise ValueError('split requires a separately justified adapter')
        quotes = result['indicators']['quote'][0]
        fresh = {'symbol':asset,'rows':[{'timestamp':t,'open':op,'close':cl} for t,op,cl
                 in zip(times,quotes['open'],quotes['close'],strict=True)],
                 'dividends':result.get('events',{}).get('dividends',{})}
        write(root/f'{asset}_2019.json',fresh)
        oldpath=Path('data/s6_real_etf_20261014')/f'{asset}.json'
        old=json.loads(oldpath.read_text())
        write(root/f'{asset}.json',{'symbol':asset,'rows':old['rows']+fresh['rows'],
              'dividends':{**old['dividends'],**fresh['dividends']}})
        receipts.append({'symbol':asset,'url':url,'http_date':server_date,
                         'retrieved_at':datetime.now(timezone.utc).isoformat(),
                         'old_source_sha256':sha(oldpath),'retained_sha256':sha(root/f'{asset}.json')})
    write(root/'acquisition.json',{'sources':receipts,'rights':'Private research; redistribution unverified',
          'limitations':'Current-vintage survivor ETFs; no PIT certification. Provider current metadata discarded.'})


def interval_return(start, end, prices, dividends):
    # Holder buys at start open: excludes start ex-date, includes end ex-date.
    cash=sum(amount for day,amount in dividends if start<day<=end)
    return (prices[end]+cash)/prices[start]-1


def build(root):
    prices, features, dividends = {}, {}, {}
    for asset in ASSETS:
        raw=json.loads((root/f'{asset}.json').read_text())
        prices[asset],features[asset]={},[]
        previous=None
        for r in raw['rows']:
            day=date_of(r['timestamp']);op,cl=r['open'],r['close']
            if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (op,cl)):
                raise ValueError('missing/invalid common-session quote; do not silently omit')
            if day in prices[asset]:raise ValueError('duplicate day')
            prices[asset][day]=op
            if previous is not None:features[asset].append((stamp(day)+DAY,math.log(cl/previous)))
            previous=cl
        dividends[asset]=[(date_of(x['date']),x['amount']) for x in raw['dividends'].values()]
    dates=sorted(set.intersection(*(set(v) for v in prices.values())))
    if any(set(v)!=set(dates) for v in prices.values()):
        raise ValueError('asset calendar mismatch; no silent calendar intersection')
    weeks={}
    for day in dates:
        dt=datetime.fromisoformat(day);week=dt.isocalendar()[:2]
        weeks.setdefault(week,day)
    anchors=list(weeks.values())
    fields=('assets','origin_times','maturity_times','available_times','window_times','window_available','windows','y','simple_returns')
    parts={p:{k:[] for k in fields} for p in ('train','calibration','evaluation')}
    parts['train'].update({k:[] for k in ('future_times','future_available','future_windows')})
    excluded=[]
    for start,end in zip(anchors,anchors[1:]):
        origin=stamp(start)+12*3600;maturity=stamp(end)+12*3600
        part='train' if start<'2017-01-01' else 'calibration' if start<'2019-01-01' else 'evaluation'
        selected=[]
        for asset in ASSETS:
            past=[x for x in features[asset] if x[0]<origin][-8:]
            future=[x for x in features[asset] if x[0]>origin][:8]
            if len(past)<8 or (part=='train' and (len(future)<8 or future[-1][0]>=stamp('2017-01-01') or maturity+4*3600>=stamp('2017-01-01'))) or (part=='calibration' and maturity+4*3600>=stamp('2019-01-01')):
                break
            selected.append((asset,past,future))
        if len(selected)!=3:excluded.append(start);continue
        for asset,past,future in selected:
            ret=interval_return(start,end,prices[asset],dividends[asset])
            values=dict(assets=asset,origin_times=origin,maturity_times=maturity,available_times=maturity+4*3600,
                        window_times=[x[0] for x in past],window_available=[x[0] for x in past],windows=[[x[1]] for x in past],
                        y=math.log1p(ret),simple_returns=ret)
            if part=='train':values.update(future_times=[x[0] for x in future],future_available=[x[0] for x in future],future_windows=[[x[1]] for x in future])
            for k,v in values.items():parts[part][k].append(v)
    payload={'schema_version':'s7-variable-week-v1','return_units':'log_return','currency':'USD','horizon_seconds':7*DAY,
             'protected_start':stamp('2025-01-01'),'fit_cutoff':stamp('2017-01-01'),'asset_universe':list(ASSETS),'partitions':parts}
    write(root/'windows.json',payload)
    write(root/'coverage.json',{'dates':{k:len(set(v['origin_times'])) for k,v in parts.items()},'excluded_for_warmup_or_purge':excluded,
          'contract':'First common trading session each ISO week to next; actual variable duration. Weekly dividends accrue at ex-date and reinvest at interval end; payment dates and taxes ignored. No missing prices filled. Continuous evaluation anchors required.',
          'limitations':'Retrospective common-session schedule, assumed availability, no PIT or actual-fill certification; 2019 is new to this S6/S7 campaign, not universally unseen market history.'})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--acquire',action='store_true');a=p.parse_args()
    if a.acquire:acquire(a.root)
    build(a.root)
