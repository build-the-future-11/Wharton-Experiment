"""Post-lockbox research MPC with an auditable self-financing ledger.

All horizon weights are shared across scenarios (open-loop nonanticipative
control). No claim of a full scenario-tree policy or global optimum is made.
The historical mpc.py remains solely for archived defect reproduction.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from wharton_lab.models.m11.mpc import PortfolioState, MPCResult
from wharton_lab.models.m11.accounting import simulate_ledger, empirical_cvar


@dataclass
class VerifiedMPCResult(MPCResult):
    method: str = "finite_policy_bank"
    optimizer_success: bool = False
    candidate_count: int = 0


def _canonical_solver_policy(flat, *, horizon, assets, max_weight, budget, allow_cash):
    """Remove only accepted optimizer boundary noise, respecting every cap.

    Feasibility is checked on the attempted controls before any projection.
    The finite policy bank is not passed through this function. A capped
    simplex projection repairs rows that need an investment-budget correction;
    ordinary normalization could exceed an already saturated asset cap.
    """
    attempted=np.asarray(flat,dtype=float)
    if attempted.size!=horizon*assets or not np.isfinite(attempted).all():
        return None
    attempted=attempted.reshape(horizon,assets)
    sums=attempted.sum(axis=1)
    if (np.min(attempted)<-1e-8 or np.max(attempted)>max_weight+1e-8
            or np.any(sums>budget+1e-7)):
        return None
    # Do not add np.allclose's default relative tolerance to the declared
    # absolute equality tolerance for fully invested controls.
    if not allow_cash and np.any(np.abs(sums-1)>1e-7):
        return None
    weights=np.clip(attempted,0,max_weight)
    for t,row in enumerate(weights):
        total=float(row.sum())
        if total<=budget and (allow_cash or total==budget):
            continue
        if assets*max_weight<budget:
            return None
        # clip(row - theta, 0, cap) has a monotone sum. Keep hi on
        # the feasible side, so rounding never returns an overspent row.
        lo=float(np.min(row-max_weight)); hi=float(np.max(row))
        for _ in range(64):
            theta=(lo+hi)*.5
            if np.clip(row-theta,0,max_weight).sum()>budget:
                lo=theta
            else:
                hi=theta
        weights[t]=np.clip(row-hi,0,max_weight)
    sums=weights.sum(axis=1)
    if (np.any(weights<0) or np.any(weights>max_weight) or np.any(sums>budget)
            or (not allow_cash and not np.allclose(sums,1,rtol=0,atol=1e-12))):
        return None
    return weights


def solve_mpc(*, state, scenario_returns, obligations, horizon, cvar_alpha,
              max_weight, tcost_bps, allow_cash=False, min_cash_buffer=0.0, solver="slsqp"):
    r=np.asarray(scenario_returns,dtype=float)
    if r.ndim!=3 or r.shape[1]!=horizon: raise ValueError('Scenario horizon mismatch')
    K=r.shape[2]
    if not 0<max_weight<=1 or not 0<=min_cash_buffer<1: raise ValueError('Invalid caps or cash buffer')
    if not allow_cash and (K*max_weight<1-1e-12 or min_cash_buffer>0): raise ValueError('Infeasible fully invested portfolio constraints')
    if state.cash+np.asarray(state.holdings).sum()<=0: raise ValueError('Nonpositive initial wealth')
    budget=1-min_cash_buffer
    w0=np.full((horizon,K), min(budget/K,max_weight))
    # Validate all ledger inputs once before the optimizer may use tolerance-level steps.
    simulate_ledger(w0,r,state.cash,state.holdings,obligations,tcost_bps*1e-4)
    empirical_cvar(np.zeros(len(r)),cvar_alpha)
    initial=state.cash+float(np.asarray(state.holdings).sum())
    def objective(flat):
        w=flat.reshape(horizon,K)
        # Feasible extension for finite-difference steps at a budget boundary.
        # Actual returned policy is separately checked against the hard constraints.
        over = float(np.maximum(w.sum(axis=1)-1,0).sum())
        w = np.maximum(w,0) / np.maximum(w.sum(axis=1,keepdims=True),1)
        ledger=simulate_ledger(w,r,state.cash,state.holdings,obligations,tcost_bps*1e-4)
        risk=empirical_cvar(ledger['unpaid'].sum(axis=1)/initial,cvar_alpha)
        return risk+1e-4*float(np.square(w).mean())+100*over
    constraints=[]
    for t in range(horizon):
        if allow_cash:
            constraints.append({'type':'ineq','fun':lambda x,t=t: budget-x.reshape(horizon,K)[t].sum()})
        else:
            constraints.append({'type':'eq','fun':lambda x,t=t: x.reshape(horizon,K)[t].sum()-1})
    # Every candidate is checked, and every reported score is a real ledger score.
    # This provides a bounded approximation, never an implicit all-in fallback.
    fractions=(0.,.25,.5,.75,1.) if allow_cash else (1.,)
    candidates=[]
    for frac in fractions:
        total=min(budget*frac,K*max_weight)
        equal=np.full(K,total/K)
        candidates.append(np.tile(equal,(horizon,1)))
        if K>1:
            for j in range(K):
                concentration=min(max_weight,total)
                other=(total-concentration)/(K-1)
                if other<=max_weight+1e-12:
                    vector=np.full(K,other); vector[j]=concentration
                    candidates.append(np.tile(vector,(horizon,1)))
        if allow_cash and frac>0:
            candidates.append(np.linspace(1,.25,horizon)[:,None]*equal)
            candidates.append(np.linspace(.25,1,horizon)[:,None]*equal)
    unique=[]; seen=set()
    for w in candidates:
        key=np.round(w,12).tobytes()
        if key not in seen:
            seen.add(key); unique.append((objective(w.ravel()),w,'finite_policy_bank'))
    numerical_success=False; message='Finite candidate bank; no continuous-optimizer optimum claimed'
    if solver=='slsqp':
        result=minimize(objective,w0.ravel(),method='SLSQP',bounds=[(0,max_weight)]*(horizon*K),
            constraints=constraints,options={'maxiter':40,'ftol':1e-8,'disp':False})
        w=_canonical_solver_policy(result.x,horizon=horizon,assets=K,
            max_weight=max_weight,budget=budget,allow_cash=allow_cash)
        numerical_success=bool(result.success and w is not None)
        message='SLSQP: '+str(result.message)+'; compared against verified finite candidate bank'
        if w is None: message+='; returned controls rejected as infeasible'
        if numerical_success: unique.append((objective(w.ravel()),w,'slsqp'))
    elif solver!='finite_candidates':
        raise ValueError('solver must be slsqp or finite_candidates')
    if not unique: raise ValueError('No feasible candidate plan')
    score,w,method=min(unique,key=lambda x:x[0])
    # Score the exact stored/returned policy, after any solver-only repair.
    score=objective(w.ravel())
    return VerifiedMPCResult(w[0].copy(),w.copy(),True,message,float(score),method,numerical_success,len(unique))
