"""Deterministic Decimal ledger. Prices are unadjusted; positions are units."""
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from typing import Literal

D = Decimal
ZERO = D(0)
TOL = D('1e-20')

def dec(x):
    y = D(str(x))
    if not y.is_finite():
        raise ValueError('nonfinite amount')
    return y

@dataclass(frozen=True)
class Obligation:
    id: str
    due_at: int
    amount: Decimal
    def __post_init__(self):
        object.__setattr__(self, 'amount', dec(self.amount))
        if not isinstance(self.id, str) or not self.id or self.amount < 0 or not isinstance(self.due_at, int) or isinstance(self.due_at, bool):
            raise ValueError('invalid obligation')

@dataclass(frozen=True)
class Contribution:
    """External cash received at the beginning of this step, before payments."""
    id: str
    amount: Decimal
    def __post_init__(self):
        object.__setattr__(self, 'amount', dec(self.amount))
        if not isinstance(self.id, str) or not self.id or self.amount < 0:
            raise ValueError('invalid contribution')

@dataclass(frozen=True)
class Policy:
    payment: Literal['cash_only', 'sell_pro_rata']
    unpaid: Literal['carry', 'write_off']
    timing: Literal['before_trade', 'after_market']
    terminal: Literal['mark', 'liquidate']
    fee_rate: Decimal
    def __post_init__(self):
        object.__setattr__(self, 'fee_rate', dec(self.fee_rate))
        if self.payment not in ('cash_only', 'sell_pro_rata') or self.unpaid not in ('carry', 'write_off') or self.timing not in ('before_trade', 'after_market') or self.terminal not in ('mark', 'liquidate') or not 0 <= self.fee_rate < 1:
            raise ValueError('invalid policy; every economic choice must be explicit')

@dataclass(frozen=True)
class State:
    at: int
    units: tuple
    prices: tuple
    cash: Decimal
    liabilities: tuple[Obligation, ...] = ()
    arrears: tuple[Obligation, ...] = ()
    seen_ids: tuple[str, ...] = ()
    closed: bool = False
    def __post_init__(self):
        for name in ('units', 'prices'):
            object.__setattr__(self, name, tuple(dec(x) for x in getattr(self, name)))
        object.__setattr__(self, 'cash', dec(self.cash))
        for name in ('liabilities', 'arrears', 'seen_ids'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.at, int) or isinstance(self.at, bool):
            raise ValueError('ledger clock must be an integer step')
        if len(self.units) != len(self.prices) or not self.units or min(self.units) < 0 or min(self.prices) < 0 or self.cash < 0:
            raise ValueError('invalid portfolio')
        obligations = self.liabilities + self.arrears
        ids = [x.id for x in obligations]
        if len(ids) != len(set(ids)) or any(x.due_at <= self.at for x in self.liabilities) or any(x.due_at > self.at for x in self.arrears):
            raise ValueError('invalid obligation partition')
        if not set(ids).issubset(self.seen_ids):
            object.__setattr__(self, 'seen_ids', tuple(sorted(set(self.seen_ids) | set(ids))))
    @property
    def holdings(self):
        return tuple(u*p for u,p in zip(self.units, self.prices))
    @property
    def wealth(self):
        return self.cash + sum(self.holdings, ZERO)
    @property
    def debt(self):
        return sum((x.amount for x in self.liabilities + self.arrears), ZERO)
    @property
    def equity(self):
        return self.wealth - self.debt

@dataclass(frozen=True)
class Payment:
    id: str
    due: Decimal
    paid: Decimal
    unpaid: Decimal
    written_off: Decimal

@dataclass(frozen=True)
class Receipt:
    state: State
    market_pnl: Decimal
    distributions: Decimal
    cash_interest: Decimal
    fees: Decimal
    payments: tuple[Payment, ...]
    turnover: Decimal
    new_liabilities: Decimal
    starting_wealth: Decimal
    starting_equity: Decimal
    contributions: tuple[Contribution, ...]
    @property
    def contributed(self):
        return sum((c.amount for c in self.contributions), ZERO)
    @property
    def paid(self):
        return sum((p.paid for p in self.payments), ZERO)
    @property
    def write_off(self):
        return sum((p.written_off for p in self.payments), ZERO)
    def verify(self):
        expected = self.starting_wealth + self.contributed + self.market_pnl + self.distributions + self.cash_interest - self.fees - self.paid
        equity = self.starting_equity + self.contributed + self.market_pnl + self.distributions + self.cash_interest - self.fees - self.new_liabilities + self.write_off
        if abs(self.state.wealth-expected) > TOL or abs(self.state.equity-equity) > TOL:
            raise ArithmeticError('ledger conservation failed')


def transition(state: State, *, at: int, returns, policy: Policy,
               target_weights=None, obligations=(), splits=None, dividends=None,
               cash_rate, return_kind, contributions=(), terminal=False) -> Receipt:
    """Order: recognize dues; optional payment; trade; split; market; dividend;
    cash interest; optional payment; optional terminal liquidation.
    Dividend is per POST-split share; returns exclude split mechanics. Total-return
    data cannot have explicit dividends/splits. FIFO payment by (due_at, id).
    """
    with localcontext() as ctx:
        ctx.prec = 40
        return _transition(state, at=at, returns=returns, policy=policy,
            target_weights=target_weights, obligations=obligations, splits=splits,
            dividends=dividends, cash_rate=cash_rate, return_kind=return_kind, contributions=contributions, terminal=terminal)


def _transition(s, *, at, returns, policy, target_weights, obligations, splits,
                dividends, cash_rate, return_kind, contributions, terminal):
    obligations = tuple(obligations)
    contributions = tuple(contributions)
    if not isinstance(at, int) or isinstance(at, bool) or at <= s.at or s.closed:
        raise ValueError('clock must advance on an open portfolio')
    n = len(s.units)
    def vector(x, default):
        a = [dec(v) for v in (x if x is not None else [default]*n)]
        if len(a) != n: raise ValueError('asset dimension mismatch')
        return a
    r = vector(returns, 0); split = vector(splits, 1); dividend = vector(dividends, 0)
    if min(r) < -1 or min(split) <= 0 or min(dividend) < 0 or dec(cash_rate) < -1:
        raise ValueError('invalid market event')
    if return_kind not in ('price', 'total') or (return_kind == 'total' and (any(v != 1 for v in split) or any(dividend))):
        raise ValueError('corporate actions require price returns to avoid double counting')
    units=list(s.units); prices=list(s.prices); cash=s.cash
    fees=ZERO; turnover=ZERO; payments=[]
    seen=set(s.seen_ids)
    for ob in obligations:
        if ob.id in seen: raise ValueError('duplicate obligation id')
        seen.add(ob.id)
    for contribution in contributions:
        if contribution.id in seen: raise ValueError('duplicate cash-flow event id')
        seen.add(contribution.id)
        cash += contribution.amount
    pending=list(s.liabilities)+list(obligations)
    arrears=list(s.arrears)+[ob for ob in pending if ob.due_at <= at]
    pending=[ob for ob in pending if ob.due_at > at]
    def sell(gross):
        nonlocal units,cash,fees,turnover
        value=sum((u*p for u,p in zip(units,prices)),ZERO)
        gross=min(gross,value)
        if gross > 0:
            units=[u*(1-gross/value) for u in units]
            fee=gross*policy.fee_rate
            cash+=gross-fee;fees+=fee;turnover+=gross
    def pay():
        nonlocal cash,arrears
        remaining=[]
        for ob in sorted(arrears,key=lambda o:(o.due_at,o.id)):
            if cash < ob.amount and policy.payment == 'sell_pro_rata':
                sell((ob.amount-cash)/(1-policy.fee_rate))
            paid=min(cash,ob.amount);cash-=paid;unpaid=ob.amount-paid
            written=unpaid if policy.unpaid == 'write_off' else ZERO
            payments.append(Payment(ob.id,ob.amount,paid,unpaid,written))
            if unpaid > 0 and policy.unpaid == 'carry': remaining.append(replace(ob,amount=unpaid))
        arrears=remaining
    if policy.timing == 'before_trade': pay()
    if target_weights is not None:
        w=vector(target_weights,0)
        if min(w)<0 or sum(w)>1: raise ValueError('long-only weights must sum to <=1')
        holdings=[u*p for u,p in zip(units,prices)]
        wealth=cash+sum(holdings,ZERO)
        # Post-fee NAV equation is continuous, strictly increasing for fee<1.
        lo=ZERO;hi=wealth
        for _ in range(140):
            mid=(lo+hi)/2
            cost=policy.fee_rate*sum((abs(wi*mid-h) for wi,h in zip(w,holdings)),ZERO)
            if mid+cost > wealth: hi=mid
            else: lo=mid
        targets=[wi*lo for wi in w]
        traded=sum((abs(t-h) for t,h in zip(targets,holdings)),ZERO)
        fee=traded*policy.fee_rate
        cash=wealth-sum(targets,ZERO)-fee
        if cash < -TOL: raise ArithmeticError('unfunded trade')
        cash=max(cash,ZERO)
        if any(p == 0 and t > 0 for t,p in zip(targets,prices)):
            raise ValueError('cannot buy an asset with zero mark')
        units=[t/p if p else ZERO for t,p in zip(targets,prices)]
        fees+=fee;turnover+=traded
    units=[u*q for u,q in zip(units,split)];prices=[p/q for p,q in zip(prices,split)]
    pnl=sum((u*p*ri for u,p,ri in zip(units,prices,r)),ZERO)
    prices=[p*(1+ri) for p,ri in zip(prices,r)]
    distributions=sum((u*d for u,d in zip(units,dividend)),ZERO);cash+=distributions
    interest=cash*dec(cash_rate);cash+=interest
    if policy.timing == 'after_market': pay()
    if terminal and policy.terminal == 'liquidate': sell(sum((u*p for u,p in zip(units,prices)),ZERO))
    result=Receipt(State(at,tuple(units),tuple(prices),cash,tuple(pending),tuple(arrears),tuple(sorted(seen)),terminal),
        pnl,distributions,interest,fees,tuple(payments),turnover,sum((o.amount for o in obligations),ZERO),s.wealth,s.equity,contributions)
    result.verify()
    return result
