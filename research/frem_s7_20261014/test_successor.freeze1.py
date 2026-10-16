import numpy as np
import pytest
from successors.frem_s7.router import MixtureMemory, mixture_weights, select_strength
from successors.frem_s7.prepare import interval_return
from successors.frem_s7.run import cost_aware
from successors.frem_portfolio_v1.engine import Forecast


def test_delayed_labels_do_not_change_prediction():
    m=MixtureMemory(2);m.add([0],[[0,-5]],10)
    before=m.weights([0],20)
    m.add([0],[[-100,100]],21)
    assert np.array_equal(before,m.weights([0],20))
    assert not np.array_equal(before,m.weights([0],22))


def test_complementary_experts_not_rejected_for_poor_standalone_loss():
    x=np.array([[[0,-20]], [[-20,0]]]*10)
    assert np.allclose(mixture_weights(x),[.5,.5])


def test_identical_experts_stay_equal_and_shift_invariance():
    x=np.zeros((10,3,6));assert np.allclose(mixture_weights(x),np.ones(6)/6)
    x=np.random.default_rng(2).normal(size=(10,3,6))
    assert np.allclose(mixture_weights(x),mixture_weights(x+np.arange(10)[:,None,None]))


def test_gate_rejects_one_half_failure():
    assert select_strength({a:[.1]*5+[-.01]*5 for a in (.25,.5,1.)})==0
    assert select_strength({a:[.02]*10 for a in (.25,.5,1.)})==.25


def test_dividend_entitlement_boundary():
    prices={'a':100,'b':99}
    assert interval_return('a','b',prices,[('a',5),('b',1)])==0


def test_no_signal_no_costly_trade_and_outcomes_not_used():
    p={'time':1,'end':2,'currency':'USD','returns':{'A':.5},'forecasts':[Forecast('A',0,.01)]}
    a=cost_aware([p],10);b=cost_aware([{**p,'returns':{'A':-.9}}],10)
    assert a['terminal_wealth']==b['terminal_wealth']==100000
    assert a['fees']==0


def test_cost_aware_conservation():
    p={'time':1,'end':2,'currency':'USD','returns':{'A':.1},'forecasts':[Forecast('A',.2,.001)]}
    r=cost_aware([p],10);row=r['periods'][0]
    assert abs(row['nav_start']-row['fee']+sum(row['contributions'].values())-row['nav_end'])<1e-8
    assert abs(r['terminal_wealth']-(sum(row['holdings'].values())+row['cash']-r['liquidation_fee']))<1e-8


def test_invalid_density_rejected():
    with pytest.raises(ValueError):mixture_weights([[[float('nan')]]])


def weekly_payload():
    def part(days,train=False):
        t=np.repeat(days,2)*86400.;n=len(t);times=t[:,None]+np.array([-3,-2,-1])*86400
        r=dict(assets=['A','B']*len(days),origin_times=t.tolist(),maturity_times=(t+5*86400).tolist(),available_times=(t+5*86400+1).tolist(),window_times=times.tolist(),window_available=times.tolist(),windows=np.zeros((n,3,1)).tolist(),y=[0.]*n,simple_returns=[0.]*n)
        if train:r.update(future_times=(times+5*86400).tolist(),future_available=(times+5*86400).tolist(),future_windows=np.zeros((n,3,1)).tolist())
        return r
    return dict(schema_version='s7-variable-week-v1',return_units='log_return',currency='USD',horizon_seconds=7*86400,protected_start=1000*86400,fit_cutoff=40*86400,asset_universe=['A','B'],partitions={'train':part([10,20],True),'calibration':part([50,60]),'evaluation':part([80,85])})


def test_actual_horizon_and_continuity():
    from successors.frem_s7.data import validate_dataset
    p=weekly_payload();validate_dataset(p)
    p['partitions']['evaluation']['maturity_times'][0]+=86400
    with pytest.raises(ValueError,match='synchronized'):validate_dataset(p)
    p=weekly_payload();p['partitions']['evaluation']['maturity_times'][:2]=[84*86400]*2
    with pytest.raises(ValueError,match='continuous'):validate_dataset(p)


def test_protected_guard_precedes_outcome_conversion():
    from successors.frem_s7.data import validate_dataset
    p=weekly_payload();p['partitions']['evaluation']['available_times'][0]=p['protected_start'];p['partitions']['evaluation']['y']=['DO NOT READ']*4
    with pytest.raises(ValueError,match='protected'):validate_dataset(p)
