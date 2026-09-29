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
        w=result.x.reshape(horizon,K)
        feasible=(np.isfinite(w).all() and np.min(w)>=-1e-8 and np.max(w)<=max_weight+1e-8
                  and np.max(w.sum(axis=1))<=budget+1e-7)
        if not allow_cash: feasible=feasible and np.allclose(w.sum(axis=1),1,atol=1e-7)
        numerical_success=bool(result.success and feasible)
        message='SLSQP: '+str(result.message)+'; compared against verified finite candidate bank'
        if numerical_success: unique.append((objective(w.ravel()),w,'slsqp'))
    elif solver!='finite_candidates':
        raise ValueError('solver must be slsqp or finite_candidates')
    if not unique: raise ValueError('No feasible candidate plan')
    score,w,method=min(unique,key=lambda x:x[0])
    return VerifiedMPCResult(w[0].copy(),w.copy(),True,message,float(score),method,numerical_success,len(unique))
