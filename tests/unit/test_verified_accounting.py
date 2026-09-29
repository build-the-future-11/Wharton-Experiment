import numpy as np
import pytest
from wharton_lab.models.m11.accounting import simulate_ledger, empirical_cvar
from wharton_lab.models.m11 import M11Model, M11Config
from wharton_lab.exceptions import BlockedClientError


def test_obligations_paid_exactly_once():
    x=simulate_ledger(np.zeros((3,2)),np.zeros((1,3,2)),1,np.zeros(2),np.array([.1,.2,.3]))
    np.testing.assert_allclose(x['wealth_after'],[[.9,.7,.4]],atol=1e-12)
    np.testing.assert_allclose(x['paid'],[[.1,.2,.3]],atol=1e-12)
    assert np.max(x['unpaid'])==0

@pytest.mark.parametrize('fee',[0,.0005,.01,.2])
def test_self_financing_identity_and_asset_permutation(fee):
    rng=np.random.default_rng(453)
    r=rng.normal(0,.05,(9,4,3)); w=rng.dirichlet(np.ones(4),size=4)[:,:3]
    h=np.array([.2,.3,.1]); L=np.array([.1,.2,.3,.4])
    x=simulate_ledger(w,r,.4,h,L,fee)
    assert np.max(np.abs(x['identity_residual']))<1e-10
    order=[2,0,1]; z=simulate_ledger(w[:,order],r[:,:,order],.4,h[order],L,fee)
    np.testing.assert_allclose(x['wealth_after'],z['wealth_after'],atol=1e-12)
    np.testing.assert_allclose(x['paid']+x['unpaid'],np.broadcast_to(L,(9,4)),atol=1e-12)
    assert np.min(x['wealth_after'])>=0


def test_exhaustion_never_creates_wealth():
    x=simulate_ledger(np.full((2,2),.5),np.zeros((1,2,2)),1,np.zeros(2),np.ones(2),.01)
    assert x['wealth_after'][0,-1]<1e-10
    assert x['unpaid'].sum()>1


def test_exact_fractional_tail_cvar():
    assert empirical_cvar(np.array([0,0,0,10]),.625)==pytest.approx(10/1.5)
    assert empirical_cvar(np.array([2,2,2]),.95)==pytest.approx(2)

@pytest.mark.parametrize('bad',[-.1,1,np.nan])
def test_reject_invalid_fee(bad):
    with pytest.raises(ValueError):
        simulate_ledger(np.zeros((1,1)),np.zeros((1,1,1)),1,np.zeros(1),np.zeros(1),bad)


def test_m11_fail_closed_instead_of_cap_violating_fallback():
    x=np.zeros((3,2,2)); m=M11Model(M11Config(max_weight=.4))
    m.fit(x,np.zeros(3),liability_schedule=np.ones(2)*.1)
    with pytest.raises(BlockedClientError,match='failing closed'):
        m.predict(x)


def test_wealth_scale_invariance():
    args=(np.full((2,2),.4),np.full((3,2,2),.05))
    a=simulate_ledger(*args,1,np.zeros(2),np.array([.2,.3]),.002)
    b=simulate_ledger(*args,100,np.zeros(2),np.array([20,30]),.002)
    np.testing.assert_allclose(a['wealth_after']*100,b['wealth_after'],atol=1e-9)
