import numpy as np
import pytest

from wharton_lab.models.m11.mpc import PortfolioState, _scenario_cash_paths
from wharton_lab.models.m11.mpc_v2 import empirical_cvar_v2, simulate_cash_ledger_v2, solve_mpc_v2


def test_legacy_failure_is_retained_but_v2_pays_obligations_once():
    returns = np.zeros((1, 2, 2))
    weights = np.zeros((2, 2))
    obligations = np.array([0.1, 0.1])
    old = _scenario_cash_paths(weights, returns, 1.0, np.zeros(2), obligations, 0.0)
    np.testing.assert_allclose(old, [[0.8, 0.8]])
    ledger = simulate_cash_ledger_v2(state=PortfolioState(np.zeros(2), 1.0),
        scenario_returns=returns, obligations=obligations, weights=weights)
    np.testing.assert_allclose(ledger.cash, [[0.9, 0.8]])
    np.testing.assert_allclose(ledger.payments, [[0.1, 0.1]])
    np.testing.assert_array_equal(ledger.arrears, np.zeros((1, 2)))


def test_unrealized_returns_stay_in_holdings():
    ledger = simulate_cash_ledger_v2(state=PortfolioState(np.zeros(2), 100.0),
        scenario_returns=np.array([[[0.1, -0.2]]]), obligations=np.zeros(1),
        weights=np.array([[0.5, 0.25]]))
    np.testing.assert_allclose(ledger.holdings, [[[55, 20]]])
    np.testing.assert_allclose(ledger.cash, [[25]])
    np.testing.assert_allclose(ledger.wealth, [[100]])


def test_trade_costs_depend_on_existing_holdings_and_real_trades():
    kwargs = dict(scenario_returns=np.zeros((1, 1, 2)), obligations=np.zeros(1),
                  weights=np.array([[0.5, 0.5]]), cost_bps=100)
    held = simulate_cash_ledger_v2(state=PortfolioState(np.array([50., 50.]), 0), **kwargs)
    bought = simulate_cash_ledger_v2(state=PortfolioState(np.zeros(2), 100), **kwargs)
    assert held.fees.sum() == pytest.approx(0, abs=1e-12)
    assert bought.fees.sum() == pytest.approx(100 / 101)
    assert bought.wealth[0, 0] + bought.fees[0, 0] == pytest.approx(100)


def test_liquidation_cost_and_wealth_conservation():
    ledger = simulate_cash_ledger_v2(state=PortfolioState(np.array([100.]), 0),
        scenario_returns=np.zeros((1, 2, 1)), obligations=np.array([10., 10.]),
        weights=np.ones((2, 1)), cost_bps=100)
    np.testing.assert_allclose(ledger.payments, [[10, 10]])
    np.testing.assert_allclose(ledger.turnover, [[10/0.99, 10/0.99]])
    assert ledger.wealth[0, -1] + ledger.fees.sum() + ledger.payments.sum() == pytest.approx(100)


def test_all_scenarios_keep_cash_and_liability_conservation():
    rng = np.random.default_rng(41)
    returns = rng.uniform(-0.3, 0.2, (4, 5, 3))
    obligations = np.array([.1, .1, .1, .7, .4])
    ledger = simulate_cash_ledger_v2(state=PortfolioState(np.array([.2, .1, .1]), .6),
        scenario_returns=returns, obligations=obligations,
        weights=np.tile([.3, .2, .1], (5, 1)), cost_bps=25)
    np.testing.assert_allclose(ledger.wealth[:, -1] + ledger.payments.sum(axis=1)
        + ledger.fees.sum(axis=1), 1 + ledger.asset_gains.sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(ledger.payments.sum(axis=1) + ledger.arrears[:, -1], obligations.sum())
    assert np.all(ledger.cash >= 0) and np.all(ledger.holdings >= 0)


def test_insolvency_keeps_unpaid_obligations_visible():
    ledger = simulate_cash_ledger_v2(state=PortfolioState(np.zeros(1), 1),
        scenario_returns=np.zeros((1, 3, 1)), obligations=np.array([.6, .6, .6]),
        weights=np.zeros((3, 1)))
    np.testing.assert_allclose(ledger.payments, [[.6, .4, 0]])
    np.testing.assert_allclose(ledger.arrears, [[0, .2, .8]])


def test_future_shock_cannot_change_past_ledger():
    kwargs = dict(state=PortfolioState(np.zeros(2), 1), obligations=np.full(4, .1),
                  weights=np.tile([.3, .3], (4, 1)))
    original = np.zeros((2, 4, 2)); changed = original.copy(); changed[:, 2:] = -.9
    a = simulate_cash_ledger_v2(scenario_returns=original, **kwargs)
    b = simulate_cash_ledger_v2(scenario_returns=changed, **kwargs)
    for name in vars(a):
        np.testing.assert_array_equal(getattr(a, name)[:, :2], getattr(b, name)[:, :2])


def test_cvar_uses_fractional_tail_and_does_not_overweight_ties():
    assert empirical_cvar_v2([0, 1, 2, 3], .625) == pytest.approx(8/3)
    assert empirical_cvar_v2([0, 0, 0, 10], .5) == pytest.approx(5)
    assert empirical_cvar_v2([0, 0, 0, 10], 0) == pytest.approx(2.5)
    assert empirical_cvar_v2([1e308, 1e308], 0) == pytest.approx(1e308)


def test_solver_cash_feasibility_and_liability_result():
    result = solve_mpc_v2(state=PortfolioState(np.zeros(2), 1),
        scenario_returns=np.zeros((2, 2, 2)), obligations=np.array([.6, .6]),
        max_weight=.2, min_cash_buffer=.25, tcost_bps=0)
    assert result.success
    assert result.objective == pytest.approx(.2)
    assert np.all(result.full_horizon_weights >= 0)
    assert np.all(result.full_horizon_weights <= .2)
    assert np.all(result.full_horizon_weights.sum(axis=1) <= .75)
    np.testing.assert_allclose(result.ledger.arrears[:, -1], .2)


def test_solver_uses_declared_scenarios_to_reduce_shortfall():
    # An assumption-only, hand-verifiable optimization case: holding cash
    # leaves 0.5 unpaid, whereas a feasible risky allocation funds the payment.
    result = solve_mpc_v2(state=PortfolioState(np.zeros(2), 1),
        scenario_returns=np.ones((2, 1, 2)), obligations=np.array([1.5]),
        max_weight=.5, tcost_bps=0)
    assert result.success and result.objective == pytest.approx(0, abs=1e-8)
    assert result.first_action_weights.sum() >= .5 - 1e-8
    np.testing.assert_allclose(result.ledger.payments, [[1.5], [1.5]])


@pytest.mark.parametrize("change", [
    {"scenario_returns": np.full((1, 2, 2), np.nan)},
    {"scenario_returns": np.full((1, 2, 2), -1.1)},
    {"obligations": np.ones(3)}, {"cost_bps": 10000},
    {"weights": np.ones((2, 2))},
    {"state": PortfolioState(np.zeros(3), 1)},
])
def test_invalid_ledger_inputs_are_rejected(change):
    kwargs = dict(state=PortfolioState(np.zeros(2), 1), scenario_returns=np.zeros((1, 2, 2)),
                  obligations=np.zeros(2), weights=np.zeros((2, 2)))
    kwargs.update(change)
    with pytest.raises(ValueError):
        simulate_cash_ledger_v2(**kwargs)
