"""Independent bounded checks against retained real quotes, not simulated results."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from .prepare import ASSETS, DAY, sha


def verify(root):
    payload=json.loads((root/'windows.json').read_text())
    quotes={a:json.loads((root/f'{a}.json').read_text())['rows'] for a in ASSETS}
    opens={a:{datetime.fromtimestamp(r['timestamp'],timezone.utc).date().isoformat():r['open']
              for r in rows} for a,rows in quotes.items()}
    returns={}
    for asset,rows in quotes.items():
        prior=None
        returns[asset]={}
        for row in rows:
            price=row['close']
            if price is None or row['open'] is None:
                prior=None
                continue
            date=datetime.fromtimestamp(row['timestamp'],timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
            if prior is not None:
                returns[asset][int(date.timestamp())+DAY]=math.log(price/prior)
            prior=price
    count=0
    for part,values in payload['partitions'].items():
        for i,asset in enumerate(values['assets']):
            t=values['origin_times'][i]
            date=datetime.fromtimestamp(t,timezone.utc).date().isoformat()
            end=datetime.fromtimestamp(t+7*DAY,timezone.utc).date().isoformat()
            assert math.isclose(values['y'][i],math.log(opens[asset][end]/opens[asset][date]),abs_tol=1e-14)
            for ts,features in zip(values['window_times'][i],values['windows'][i],strict=True):
                assert ts<t and features==[returns[asset][ts]]
            if part=='train':
                for ts,features in zip(values['future_times'][i],values['future_windows'][i],strict=True):
                    assert ts>t and ts<payload['fit_cutoff'] and features==[returns[asset][ts]]
            count+=1
    return {'status':'PASS','real_rows_reconciled':count,'checks':[
      'Every label independently reconciled to retained Monday opens',
      'Every context feature reconciled to daily closes with strict origin ordering',
      'Every JEPA future feature reconciled and purged at fit cutoff'],
      'data_sha256':sha(root/'windows.json'),'market_fits':0}


if __name__=='__main__':
    import argparse
    from .prepare import write
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    write(a.output,verify(a.root))
