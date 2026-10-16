"""Strict retrospective ETF adapter. No network or protected-tail reads."""

import math
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import pstdev

from successors.frem_audit_v1.audit import digest, read
from successors.frem_audit_v1.contracts import finite

ASSETS = ('SPY', 'TLT', 'GLD')
DAY = 86400


def stamp(day):
    return datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp()


def date_of(value):
    return datetime.fromtimestamp(value, UTC).date().isoformat()


def outcome(prices, dividends, start, end):
    if start >= end:
        raise ValueError('positive interval required')
    g, d = [], []
    for asset in ASSETS:
        initial, final = prices[asset][start], prices[asset][end]
        g.append(final / initial)
        d.append(math.fsum(amount for day, amount in dividends[asset] if start < day <= end) / initial)
    return {'g': g, 'd': d}


def prepare(root):
    root = Path(root)
    sources = read(root / 'acquisition.json')['sources']
    if [row['symbol'] for row in sources] != list(ASSETS):
        raise ValueError('source identity/order mismatch')
    prices, timestamps, dividends, identities = {}, {}, {}, {}
    for asset, source in zip(ASSETS, sources, strict=True):
        path = root / f'{asset}.json'
        identities[str(path)] = digest(path)
        if identities[str(path)] != source['retained_sha256']:
            raise ValueError('retained source hash mismatch')
        raw = read(path)
        if raw['symbol'] != asset or raw['split_events']:
            raise ValueError('wrong symbol or unsupported split')
        prices[asset], timestamps[asset], dividends[asset] = {}, {}, []
        previous = -math.inf
        for row in raw['rows']:
            time = finite(row['timestamp'], 'source timestamp')
            if time <= previous or not stamp('2005-01-01') <= time < stamp('2019-01-01'):
                raise ValueError('source time order/scope violation')
            previous = time
            day = date_of(time)
            if day >= '2017-01-01':
                continue
            op, close = finite(row['open'], 'open'), finite(row['close'], 'close')
            if min(op, close) <= 0 or day in prices[asset]:
                raise ValueError('invalid or duplicate quote')
            if time <= stamp(day) + 12*3600:
                raise ValueError('recorded open must follow declared decision time')
            prices[asset][day], timestamps[asset][day] = op, time
        seen = set()
        for item in raw['dividends'].values():
            time, amount = finite(item['date'], 'dividend timestamp'), finite(item['amount'], 'dividend')
            day = date_of(time)
            if amount < 0 or day in seen:
                raise ValueError('invalid or duplicate dividend day')
            seen.add(day)
            if '2005-01-01' <= day < '2017-01-01':
                if day not in prices[asset]:
                    raise ValueError('dividend ex-date absent from session calendar')
                dividends[asset].append((day, amount))
    days = sorted(prices[ASSETS[0]])
    if not days or any(set(prices[a]) != set(days) for a in ASSETS):
        raise ValueError('asset calendars differ; no silent intersection')
    for day in days:
        if len({timestamps[a][day] for a in ASSETS}) != 1:
            raise ValueError('nonsynchronous asset opens')
    weeks = {}
    for day in days:
        weeks.setdefault(datetime.fromisoformat(day).isocalendar()[:2], day)
    anchors = list(weeks.values())
    weekly = []
    for start, end in pairwise(anchors):
        if not 3*DAY <= stamp(end)-stamp(start) <= 10*DAY:
            raise ValueError('missing/invalid weekly interval')
        result = outcome(prices, dividends, start, end)
        weekly.append({'available': stamp(end)+DAY,
                       'log': [math.log(g+d) for g, d in zip(result['g'], result['d'], strict=True)]})
    samples = []
    for index in range(len(anchors)-4):
        start, end = anchors[index], anchors[index+4]
        origin = stamp(start)+12*3600
        history = [r for r in weekly[:index] if r['available'] < origin][-13:]
        if len(history) < 13:
            continue
        features = []
        for a in range(3):
            logs = [r['log'][a] for r in history]
            features.extend([logs[-1], math.fsum(logs[-4:]), math.fsum(logs), pstdev(logs)])
        samples.append({'index': index, 'origin': origin, 'start': timestamps[ASSETS[0]][start],
                        'end': timestamps[ASSETS[0]][end], 'available': stamp(end)+DAY,
                        'feature_available': history[-1]['available'], 'x': features,
                        'h1': outcome(prices, dividends, start, anchors[index+1]),
                        'h4': outcome(prices, dividends, start, end)})
    return samples, {'scope': 'RETROSPECTIVE_PRE_2017_ONLY_NOT_PIT_CERTIFIED', 'assets': ASSETS,
                     'daily_sessions': len(days), 'weekly_anchors': len(anchors),
                     'samples': len(samples), 'first_origin': samples[0]['origin'],
                     'last_outcome': samples[-1]['end'], 'identities': identities,
                     'source_acquisition_sha256': digest(root/'acquisition.json'),
                     'source_retrieved_at': [s['retrieved_at'] for s in sources],
                     'protected_access': False, 'redistribution_rights_verified': False}


def split(samples, cutoff, end, *, minimum_train=64, minimum_evaluation=24):
    if cutoff >= end or end > stamp('2017-01-01'):
        raise ValueError('invalid diagnostic fold boundaries')
    train = [r for r in samples if r['available'] < cutoff]
    eligible = [r for r in samples if cutoff <= r['origin'] < end and r['available'] < end]
    evaluation = []
    next_index = eligible[0]['index'] if eligible else None
    for row in eligible:
        if row['index'] == next_index:
            if evaluation and evaluation[-1]['end'] != row['start']:
                raise ValueError('evaluation gap')
            evaluation.append(row)
            next_index += 4
    if len(train) < minimum_train or len(evaluation) < minimum_evaluation:
        raise ValueError('insufficient train/evaluation observations')
    if any(r['feature_available'] >= r['origin'] or r['origin'] >= r['start']
           or r['start'] >= r['end'] or r['end'] > r['available'] for r in train+evaluation):
        raise ValueError('sample chronology violation')
    return train, evaluation
