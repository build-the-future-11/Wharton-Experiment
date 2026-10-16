"""Deterministic conditional empirical scenarios, with frozen train-only scales."""

import math
from statistics import mean, pstdev

from successors.frem_audit_v1.contracts import finite


def vector(values, size, name):
    if len(values) != size:
        raise ValueError(f'{name} dimension mismatch')
    return [finite(v, name) for v in values]


def valid_outcome(value):
    g, d = vector(value['g'], 3, 'gross prices'), vector(value['d'], 3, 'dividends')
    if min(g) <= 0 or min(d) < 0:
        raise ValueError('positive prices and nonnegative dividends required')
    return {'g': g, 'd': d}


def fit(rows, cutoff, *, neighbors=32):
    cutoff = finite(cutoff, 'fit cutoff')
    if type(neighbors) is not int or neighbors < 1 or len(rows) < neighbors:
        raise ValueError('insufficient exemplars or invalid neighbor count')
    training = []
    previous = -math.inf
    for r in rows:
        origin = finite(r['origin'], 'origin')
        start, end, available, feature_available = [finite(r[k], k) for k in
                                                   ('start', 'end', 'available', 'feature_available')]
        if not previous < origin < start < end <= available < cutoff:
            raise ValueError('training order or maturity violation')
        if feature_available >= origin:
            raise ValueError('unavailable training feature')
        previous = origin
        training.append({'origin': origin, 'x': vector(r['x'], 12, 'features'),
                         'start': start, 'end': end, 'available': available,
                         'feature_available': feature_available,
                         'h1': valid_outcome(r['h1']), 'h4': valid_outcome(r['h4'])})
    means = [mean(r['x'][j] for r in training) for j in range(12)]
    scales = [pstdev(r['x'][j] for r in training) for j in range(12)]
    scales = [s if s >= 1e-12 else 1.0 for s in scales]
    return {'schema': 's9-empirical-v1', 'cutoff': cutoff, 'neighbors': neighbors,
            'means': means, 'scales': scales, 'training': training}


def validate_checkpoint(model):
    if model['schema'] != 's9-empirical-v1':
        raise ValueError('checkpoint schema mismatch')
    vector(model['means'], 12, 'means')
    if min(vector(model['scales'], 12, 'scales')) <= 0:
        raise ValueError('nonpositive scale')
    if type(model['neighbors']) is not int or not 1 <= model['neighbors'] <= len(model['training']):
        raise ValueError('invalid checkpoint neighbors')
    previous = -math.inf
    for row in model['training']:
        times = [finite(row[k], k) for k in ('origin', 'start', 'end', 'available')]
        origin, start, end, available = times
        if not previous < origin < start < end <= available < finite(model['cutoff'], 'cutoff'):
            raise ValueError('checkpoint training order')
        if finite(row['feature_available'], 'feature availability') >= origin:
            raise ValueError('checkpoint feature leakage')
        previous = row['origin']
        vector(row['x'], 12, 'features')
        valid_outcome(row['h1']); valid_outcome(row['h4'])
    return model


def predict(model, query, *, horizon, local):
    if horizon not in (1, 4) or type(horizon) is not int or type(local) is not bool:
        raise ValueError('invalid forecast configuration')
    origin = finite(query['origin'], 'query origin')
    if not model['cutoff'] < origin or finite(query['feature_available'], 'query availability') >= origin:
        raise ValueError('query chronology violation')
    x = vector(query['x'], 12, 'query')
    training = model['training']
    if local:
        distances = [(math.fsum(((v-t)/s)**2 for v, t, s in zip(x, r['x'], model['scales'], strict=True)),
                      r['origin'], r) for r in training]
        chosen = [r for _, _, r in sorted(distances, key=lambda item: item[:2])[:model['neighbors']]]
    else:
        chosen = training
    scenarios = []
    for row in chosen:
        value = valid_outcome(row[f'h{horizon}'])
        if horizon == 1:
            g, d = value['g'], value['d']
            value = valid_outcome({'g': [v**4 for v in g],
                                   'd': [a*(1+b+b*b+b**3) for a, b in zip(d, g, strict=True)]})
        scenarios.append(value)
    expected = [mean(s['g'][j]+s['d'][j]-1 for s in scenarios) for j in range(3)]
    return {'origins': [r['origin'] for r in chosen], 'scenarios': scenarios, 'mean_total_return': expected}
