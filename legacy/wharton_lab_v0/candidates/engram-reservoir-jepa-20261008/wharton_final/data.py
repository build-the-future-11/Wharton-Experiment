"""Offline admitted return panels and strictly chronological horizon labels."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv, hashlib, json
import numpy as np

LEGACY_SHA = '6c234590d0b4c28901c9919044141e0617ae73b592b37bd510728386e26cfd29'

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def timestamp(s):
    t=datetime.fromisoformat(s.replace('Z','+00:00'))
    if t.tzinfo is None: raise ValueError('timestamps must be timezone-aware')
    return t.timestamp()

@dataclass
class Panel:
    returns: np.ndarray
    decisions: np.ndarray
    observed: np.ndarray
    assets: tuple
    revision: dict

@dataclass
class Samples:
    context: np.ndarray
    future: np.ndarray
    labels: np.ndarray
    decisions: np.ndarray
    maturity: np.ndarray
    rows: np.ndarray

def load_panel(manifest_path):
    path=Path(manifest_path).resolve();m=json.loads(path.read_text())
    required={'csv','sha256','assets','source','license','lawful_use_attested','asset_order_verified','role','return_kind','protected_rows','data_kind'}
    if required-m.keys(): raise ValueError('missing data contract fields: '+str(sorted(required-m.keys())))
    if m['role'] != 'development' or m['lawful_use_attested'] is not True or m['asset_order_verified'] is not True or not m['source'] or not m['license']:
        raise ValueError('data not admitted for lawful development with verified asset order')
    if m['data_kind'] not in ('market','synthetic_fixture'): raise ValueError('data kind must be explicit')
    if m['return_kind'] != 'total': raise ValueError('forecast panel requires total returns; accounting uses a separate explicit event contract')
    p=(path.parent/m['csv']).resolve()
    if digest(p) != m['sha256']: raise ValueError('data checksum mismatch')
    if m.get('upstream_sha256') == LEGACY_SHA:
        # No raw re-labelled subset of spent history admitted without an exact row map.
        rows=m.get('upstream_rows')
        if rows is None or not rows or any(not isinstance(i,int) or isinstance(i,bool) or i<0 or i>=1691 for i in rows) or any(b-a!=1 for a,b in zip(rows,rows[1:])) or any(800 <= i < 960 for i in rows):
            raise ValueError('spent historical lockbox cannot be admitted')
    if m['protected_rows']:
        raise ValueError('input contains protected rows; export an admitted development-only panel first')
    assets=tuple(m['assets'])
    if not assets or len(set(assets)) != len(assets): raise ValueError('invalid asset identities')
    with p.open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['decision_time','observed_at',*assets]: raise ValueError('CSV header/order differs from contract')
        records=list(reader)
    returns=np.array([[float(r[a]) for a in assets] for r in records])
    decisions=np.array([timestamp(r['decision_time']) for r in records])
    observed=np.array([timestamp(r['observed_at']) for r in records])
    validate_panel(returns,decisions,observed)
    if m.get('upstream_sha256')==LEGACY_SHA and len(m['upstream_rows']) != len(returns): raise ValueError('row map mismatch')
    return Panel(returns,decisions,observed,assets,{'manifest_sha256':digest(path),'data_sha256':digest(p),'contract':m})

def validate_panel(r,d,o):
    if r.ndim != 2 or len(r)<2 or r.shape[1]<1 or len(d)!=len(r) or len(o)!=len(r) or not np.isfinite(r).all() or not np.isfinite(d).all() or not np.isfinite(o).all() or np.any(r<=-1):
        raise ValueError('invalid panel dimensions or returns')
    if np.any(np.diff(d)<=0) or np.any(np.diff(o)<=0) or np.any(o<d): raise ValueError('invalid observation clock')

def samples(panel,context,horizon,reporting_delay_seconds):
    r,d,o=panel.returns,panel.decisions,panel.observed
    validate_panel(r,d,o)
    if context<2 or horizon<1 or reporting_delay_seconds<0: raise ValueError('invalid context/horizon/delay')
    xs=[];ys=[];future=[];rows=[];maturity=[]
    for t in range(context,len(r)-horizon+1):
        if o[t-1]>=d[t]: continue
        xs.append(r[t-context:t]);future.append(r[t:t+horizon])
        ys.append(np.expm1(np.log1p(r[t:t+horizon]).sum(axis=0)))
        rows.append(t);maturity.append(o[t+horizon-1]+reporting_delay_seconds)
    if not rows: raise ValueError('no chronologically available samples')
    return Samples(np.array(xs),np.array(future),np.array(ys),d[rows],np.array(maturity),np.array(rows))

def matured_indices(s,as_of):
    if not np.isfinite(as_of) or not np.isfinite(s.maturity).all() or not np.isfinite(s.decisions).all() or np.any(s.maturity<s.decisions):
        raise ValueError('invalid maturity clock')
    return np.flatnonzero((s.maturity < as_of) & (s.decisions < as_of))
