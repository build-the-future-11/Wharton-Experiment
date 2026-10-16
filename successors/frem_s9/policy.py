"""Exact finite-action optimization and round-trip cash/dividend accounting."""

import itertools
import math
from statistics import mean

from successors.frem_audit_v1.contracts import finite

from .model import valid_outcome, vector

ACTIONS = tuple(w for w in itertools.product((0., 1/12, 2/12, 3/12, 4/12), repeat=3)
                if math.fsum(w) <= .8+1e-12)


def terminal_factor(weights, value, fee):
    w, outcome = vector(weights, 3, 'weights'), valid_outcome(value)
    fee = finite(fee, 'fee')
    if min(w) < 0 or math.fsum(w) > 1+1e-12 or not 0 <= fee < 1:
        raise ValueError('invalid cash-funded allocation or fee')
    gross = 1-math.fsum(w)+math.fsum(a*((1-fee)*g+d)
                                    for a, g, d in zip(w, outcome['g'], outcome['d'], strict=True))
    return gross/(1+fee*math.fsum(w))


def moments(scenarios, fee):
    if not scenarios or not 0 <= finite(fee, 'fee') < 1:
        raise ValueError('nonempty scenarios and valid fee required')
    rows = []
    for value in scenarios:
        value = valid_outcome(value)
        rows.append([(1-fee)*g+d-1 for g, d in zip(value['g'], value['d'], strict=True)])
    mu = [mean(row[j] for row in rows) for j in range(3)]
    covariance = [[mean((r[i]-mu[i])*(r[j]-mu[j]) for r in rows) for j in range(3)] for i in range(3)]
    return mu, covariance


def objective(weights, mu, covariance, fee, diagonal):
    factor = 1/(1+fee*math.fsum(weights))
    expected = factor*(1+math.fsum(w*m for w, m in zip(weights, mu, strict=True)))-1
    variance = factor**2*math.fsum(weights[i]*covariance[i][j]*weights[j]
                                  for i in range(3) for j in range(3) if not diagonal or i == j)
    result = expected-4*variance
    if not math.isfinite(result) or variance < -1e-12:
        raise ValueError('invalid objective')
    return result


def choose(scenarios, fee, *, diagonal=False):
    if type(diagonal) is not bool:
        raise ValueError('diagonal must be boolean')
    mu, covariance = moments(scenarios, fee)
    scores = [objective(w, mu, covariance, fee, diagonal) for w in ACTIONS]
    best = max(range(len(scores)), key=scores.__getitem__)
    return {'weights': list(ACTIONS[best]), 'objective': scores[best],
            'cash_objective': scores[0], 'actions_checked': len(ACTIONS)}


def round_trip(nav, weights, value, fee):
    nav = finite(nav, 'NAV')
    if nav <= 0:
        raise ValueError('positive NAV required')
    factor = terminal_factor(weights, value, fee)
    post = nav/(1+fee*math.fsum(weights))
    purchased = [post*w for w in weights]
    entry = fee*math.fsum(purchased)
    sale = [a*g for a, g in zip(purchased, value['g'], strict=True)]
    dividends = math.fsum(a*d for a, d in zip(purchased, value['d'], strict=True))
    exit_fee = fee*math.fsum(sale)
    terminal = post-math.fsum(purchased)+math.fsum(sale)+dividends-exit_fee
    pnl = [a*(g+d-1) for a, g, d in zip(purchased, value['g'], value['d'], strict=True)]
    if abs(terminal-nav*factor) > 1e-8 or abs(terminal-nav-math.fsum(pnl)+entry+exit_fee) > 1e-8:
        raise ValueError('round-trip accounting mismatch')
    return {'nav_start': nav, 'nav_end': terminal, 'net_factor': factor,
            'weights': list(weights), 'entry_fee': entry, 'exit_fee': exit_fee,
            'fees': entry+exit_fee, 'dividend_cash': dividends, 'contributions': pnl}


def simulate(periods, targets, fee):
    if not periods or len(periods) != len(targets):
        raise ValueError('complete targets required')
    nav, peak, drawdown, rows = 100000., 100000., 0., []
    previous = None
    for period, target in zip(periods, targets, strict=True):
        if period['start'] >= period['end'] or (previous is not None and period['start'] != previous):
            raise ValueError('noncontinuous ledger')
        row = round_trip(nav, target, period['h4'], fee)
        row.update(start=period['start'], end=period['end'])
        nav = row['nav_end']; peak = max(peak, nav); drawdown = max(drawdown, 1-nav/peak)
        rows.append(row); previous = period['end']
    return {'terminal_wealth': nav, 'net_return': nav/100000-1,
            'fees': math.fsum(r['fees'] for r in rows), 'endpoint_drawdown': drawdown,
            'periods': rows}


def buy_hold(periods, fee):
    if not periods or not 0 <= finite(fee, 'fee') < 1:
        raise ValueError('invalid buy-and-hold inputs')
    entry = 100000*fee/(1+fee)
    holdings, cash, peak, drawdown, rows, previous = [(100000-entry)/3]*3, 0., 100000., 0., [], None
    for index, period in enumerate(periods):
        if period['start'] >= period['end'] or (previous is not None and previous != period['start']):
            raise ValueError('buy-and-hold gap')
        value = valid_outcome(period['h4'])
        before = 100000. if index == 0 else cash+math.fsum(holdings)
        cash += math.fsum(a*d for a, d in zip(holdings, value['d'], strict=True))
        holdings = [a*g for a, g in zip(holdings, value['g'], strict=True)]
        exit_fee = fee*math.fsum(holdings) if index == len(periods)-1 else 0.
        nav = cash+math.fsum(holdings)-exit_fee
        peak = max(peak, nav); drawdown = max(drawdown, 1-nav/peak)
        rows.append({'start': period['start'], 'end': period['end'], 'nav_start': before,
                     'nav_end': nav, 'fees': (entry if index == 0 else 0.)+exit_fee,
                     'net_factor': nav/before})
        previous = period['end']
    return {'terminal_wealth': nav, 'net_return': nav/100000-1,
            'fees': entry+exit_fee, 'endpoint_drawdown': drawdown, 'periods': rows}
