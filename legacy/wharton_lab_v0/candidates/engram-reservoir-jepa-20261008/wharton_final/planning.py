"""Forecast-conditioned scenarios and common-action liability planning."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .accounting import transition, dec


def scenarios(history, forecast, *, horizon, count, seed):
    """Moving-block bootstrap of past log returns, recentered on horizon forecast.

    Forecast conditions expected log growth, not arithmetic expected terminal wealth.
    Finite empirical residual scenarios are exploratory, not calibrated guarantees.
    """
    history=np.asarray(history,float);forecast=np.asarray(forecast,float)
    if history.ndim!=2 or forecast.shape!=(history.shape[1],) or len(history)<horizon or horizon<1 or count<2 or not np.isfinite(history).all() or not np.isfinite(forecast).all() or np.any(history<=-1) or np.any(forecast<=-1) or seed==997:
        raise ValueError('invalid scenario inputs or spent seed')
    logs=np.log1p(history);center=logs.mean(0)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(logs)-horizon+1,size=count)
    shocks=np.array([logs[s:s+horizon]-center for s in starts])
    out=np.expm1(shocks+np.log1p(forecast)[None,None,:]/horizon)
    if not np.isfinite(out).all():raise FloatingPointError('scenario overflow')
    return out

@dataclass
class Plan:
    weights: np.ndarray
    objective: float
    scenario_terminal_cash: np.ndarray
    scenario_terminal_wealth: np.ndarray
    scenario_unpaid_debt: np.ndarray
    scenario_paid: np.ndarray
    scenario_fees: np.ndarray
    scenario_future_liabilities: np.ndarray
    iterations: int


def rollout(state, paths, weights, obligations, policy, *, terminal_cash_target, cash_rates, future_liability_policy):
    if future_liability_policy not in ('reserve_nominal_cash','report_only'):
        raise ValueError('explicit future liability policy required')
    paths=np.asarray(paths,float)
    if paths.ndim!=3 or not np.isfinite(paths).all() or np.any(paths<=-1) or len(obligations)!=paths.shape[1] or paths.shape[2]!=len(state.units) or not np.isfinite(terminal_cash_target) or terminal_cash_target<0 or len(cash_rates)!=paths.shape[1] or not np.isfinite(np.asarray(cash_rates,float)).all() or min(cash_rates)<-1:
        raise ValueError('invalid planner contract')
    weights=[dec(w) for w in weights]
    total=sum(weights)
    if 1 < total <= dec('1.000000000001'):
        # Remove optimizer roundoff only; a material leverage request still fails.
        weights=[w/total for w in weights]
        excess=max(dec(0),sum(weights)-1)
        j=max(range(len(weights)),key=lambda i:weights[i]);weights[j]-=excess
    results=[]
    for path in paths:
        current=state;paid=0.;fees=0.
        for t,returns in enumerate(path):
            at=state.at+t+1
            receipt=transition(current,at=at,returns=returns,policy=policy,
                target_weights=weights,obligations=obligations[t],cash_rate=cash_rates[t],return_kind='total',terminal=t==len(path)-1)
            current=receipt.state;paid+=float(receipt.paid);fees+=float(receipt.fees)
        # A future liability is a reserve requirement, not a missed payment.
        terminal_at=state.at+len(path)
        due=sum(float(o.amount) for o in state.liabilities+state.arrears if o.due_at<=terminal_at)
        due+=sum(float(o.amount) for step in obligations for o in step if o.due_at<=terminal_at)
        unpaid=max(0.,due-paid)  # Retains written-off due amounts as failures.
        future_liabilities=sum(float(o.amount) for o in current.liabilities)
        future_reserve=future_liabilities if future_liability_policy=='reserve_nominal_cash' else 0.
        shortfall=unpaid+max(0.,terminal_cash_target+future_reserve-float(current.cash))
        results.append((float(current.cash),float(current.wealth),unpaid,paid,fees,future_liabilities,shortfall))
    return np.array(results)


def cvar(loss, alpha):
    """Empirical upper CVaR via fractional tail mass (no quantile tie inflation)."""
    x=np.sort(np.asarray(loss,float))[::-1]
    if len(x)==0 or not np.isfinite(x).all() or not 0<=alpha<1:raise ValueError('invalid CVaR input')
    mass=(1-alpha)*len(x);whole=int(np.floor(mass));fraction=mass-whole
    return float((x[:whole].sum()+(fraction*x[whole] if whole<len(x) else 0))/mass)


def plan(state, paths, obligations, policy, *, max_weight, alpha, terminal_cash_target, cash_rates, future_liability_policy, maxiter=60):
    if not 0<max_weight<=1:raise ValueError('invalid cap')
    paths=np.asarray(paths,float);k=len(state.units)
    def objective(w):
        # Optimizer probes can be slightly outside feasible boundaries.
        if np.min(w)<-1e-9 or np.sum(w)>1+1e-9:return 1e30
        w=np.maximum(w,0);w=w/max(1.,w.sum())
        return cvar(rollout(state,paths,w,obligations,policy,terminal_cash_target=terminal_cash_target,cash_rates=cash_rates,future_liability_policy=future_liability_policy)[:,-1],alpha)
    # Cash and feasible equal-weight starts are both assessed, with no asset-0 fallback.
    candidates=[]
    for initial in (np.zeros(k),np.full(k,min(max_weight,1/k))):
        result=minimize(objective,initial,method='SLSQP',bounds=[(0,max_weight)]*k,
            constraints=[{'type':'ineq','fun':lambda w:1-w.sum()}],options={'maxiter':maxiter,'ftol':1e-8})
        if result.success and np.isfinite(result.fun) and result.x.min()>=-1e-8 and result.x.max()<=max_weight+1e-8 and result.x.sum()<=1+1e-8:
            candidates.append(result)
    if not candidates:raise RuntimeError('planner failed; no allocation approved')
    best=min(candidates,key=lambda r:r.fun);w=np.clip(best.x,0,max_weight);w/=max(1,w.sum())
    outcomes=rollout(state,paths,w,obligations,policy,terminal_cash_target=terminal_cash_target,cash_rates=cash_rates,future_liability_policy=future_liability_policy)
    return Plan(w,cvar(outcomes[:,-1],alpha),outcomes[:,0],outcomes[:,1],outcomes[:,2],outcomes[:,3],outcomes[:,4],outcomes[:,5],int(best.nit))
