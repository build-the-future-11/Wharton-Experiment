"""Self-financing long-only accounting for synthetic research, not execution.

Returns are simple returns; holdings and obligations are currency amounts.
Weights are fractions of *post-trade* wealth. Both buying/selling and liability
liquidations pay the same proportional fee. Each liability is paid exactly once.
"""
from __future__ import annotations
import numpy as np


def empirical_cvar(losses: np.ndarray, alpha: float=.95) -> float:
    """Exact expected worst (1-alpha) fraction of an equally weighted sample.

Fractional mass at the tail boundary avoids tie/quantile rounding errors.
"""
    x=np.asarray(losses,dtype=float).ravel()
    if not len(x) or not np.isfinite(x).all() or not 0<=alpha<1:
        raise ValueError("Finite losses and 0 <= alpha < 1 are required")
    ordered=np.sort(x)[::-1]; mass=(1-alpha)*len(x)
    whole=int(np.floor(mass)); fraction=mass-whole
    total=float(ordered[:whole].sum())
    if whole<len(x): total+=fraction*ordered[whole]
    return total/mass


def simulate_ledger(weights, returns, cash, holdings, obligations, fee=0.0):
    """Vectorized independent scenarios. No borrowing or negative holdings.

Returns a dictionary of auditable per-scenario, per-period quantities.
An obligation can remain unpaid at exhaustion; it never creates fictitious debt
or gets charged a second time in the next period. Cash earns zero here.
"""
    r=np.asarray(returns,dtype=float); w=np.asarray(weights,dtype=float)
    h0=np.asarray(holdings,dtype=float); L=np.asarray(obligations,dtype=float)
    if r.ndim!=3 or 0 in r.shape: raise ValueError("returns must be nonempty (scenarios,horizon,assets)")
    S,H,K=r.shape
    if w.shape!=(H,K) or h0.shape!=(K,) or L.shape!=(H,): raise ValueError("Accounting shape mismatch")
    if not all(np.isfinite(x).all() for x in (r,w,h0,L)) or not np.isfinite(cash): raise ValueError("Nonfinite ledger input")
    if cash<0 or np.any(h0<0) or np.any(L<0) or np.any(r < -1): raise ValueError("Invalid long-only state, obligation, or simple return")
    if np.min(w)<-1e-8 or np.any(w.sum(axis=1)>1+1e-8) or not 0<=fee<1:
        raise ValueError("Weights must be long-only with sum <= 1; fee must be in [0,1)")
    # Project only floating-point tolerances, never a materially infeasible policy.
    w=np.maximum(w,0); w=w/np.maximum(w.sum(axis=1,keepdims=True),1)
    h=np.repeat(h0[None],S,axis=0); c=np.full(S,float(cash))
    names=('wealth_before','wealth_after','cash_after','trade_fee','liquidation_fee','market_gain','paid','unpaid','identity_residual')
    out={name:np.zeros((S,H)) for name in names}
    for t in range(H):
        before=c+h.sum(axis=1)
        # Unique root: V + fee * ||w V - h||_1 = before.
        if fee <= .2:
            # This fixed-point map is a contraction with factor <= fee.
            v=before.copy()
            for _ in range(32):
                updated=before-fee*np.abs(v[:,None]*w[t]-h).sum(axis=1)
                error=np.max(np.abs(updated-v))
                v=updated
                if error<=1e-13*max(1.,float(np.max(before))): break
            else:
                raise ArithmeticError("Post-trade wealth root did not meet tolerance")
        else:
            lo=np.zeros(S); hi=before.copy()
            for _ in range(50):
                mid=(lo+hi)*.5
                val=mid+fee*np.abs(mid[:,None]*w[t]-h).sum(axis=1)-before
                hi=np.where(val>0,mid,hi); lo=np.where(val<=0,mid,lo)
            v=(lo+hi)*.5
        new_h=v[:,None]*w[t]
        trade_fee=fee*np.abs(new_h-h).sum(axis=1)
        c=v*(1-w[t].sum())
        gain=(new_h*r[:,t,:]).sum(axis=1)
        h=new_h*(1+r[:,t,:])
        cash_paid=np.minimum(c,L[t]); c-=cash_paid
        need=L[t]-cash_paid; value=h.sum(axis=1)
        gross_sold=np.minimum(value,need/(1-fee))
        fraction=np.divide(gross_sold,value,out=np.zeros_like(value),where=value>0)
        h*=1-fraction[:,None]
        liquidation_fee=fee*gross_sold
        sale_paid=gross_sold-liquidation_fee
        paid=cash_paid+sale_paid; unpaid=np.maximum(L[t]-paid,0)
        after=c+h.sum(axis=1)
        residual=after-(before-trade_fee+gain-liquidation_fee-paid)
        vals=(before,after,c,trade_fee,liquidation_fee,gain,paid,unpaid,residual)
        for name,val in zip(names,vals): out[name][:,t]=val
    out['final_holdings']=h
    return out
