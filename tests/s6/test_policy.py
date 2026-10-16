"""Numerical contracts for the isolated S6 conditional policy module."""

import numpy as np
import pytest

from successors.frem_s6.policy import (
    convert_to_base_returns,
    empirical_cvar,
    plan_portfolio,
    sample_forecast_paths,
    sample_joint_paths,
    self_financing_rebalance,
    trailing_shrinkage_covariance,
)


def test_common_currency_compounding_and_exact_total_loss():
    actual = convert_to_base_returns([-1.0, 0.1, -0.2], [0.7, 0.2, -0.1])
    np.testing.assert_allclose(actual, [-1, 0.32, -0.28])
    assert actual[0] == -1
    with pytest.raises(ValueError):
        convert_to_base_returns([-1.01], [0])
    with pytest.raises(ValueError, match="matching shapes"):
        convert_to_base_returns(np.zeros((2, 2)), np.zeros(2))


def test_covariance_uses_only_trailing_available_rows_with_no_future_validation():
    times = [f"2020-01-0{day}T00:00:00Z" for day in range(1, 6)]
    values = np.array([[.9, .9], [.1, -.1], [.2, -.2], [.3, -.3], [np.nan, -99]])
    cutoff = "2020-01-04T12:00:00Z"
    fit = trailing_shrinkage_covariance(values, times, decision_time=cutoff,
                                       lookback=3, shrinkage=.25)
    np.testing.assert_allclose(fit.mean, [.2, -.2])
    np.testing.assert_allclose(fit.covariance, [[.01, -.0075], [-.0075, .01]])
    assert fit.rows_used == 3
    assert fit.available_through == "2020-01-04T00:00:00+00:00"
    values[-1] = [1e100, 1e100]
    same = trailing_shrinkage_covariance(values, times, decision_time=cutoff,
                                        lookback=3, shrinkage=.25)
    np.testing.assert_array_equal(fit.covariance, same.covariance)
    opaque = values.astype(object)
    opaque[-1] = ["PROTECTED", "PROTECTED"]
    unseen = trailing_shrinkage_covariance(opaque, times, decision_time=cutoff,
                                          lookback=3, shrinkage=.25)
    np.testing.assert_array_equal(fit.covariance, unseen.covariance)


def test_covariance_rejects_unmatured_or_impossible_timing_and_missing_history():
    values = [[.1], [.2]]
    times = ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"]
    with pytest.raises(ValueError, match="at least two"):
        trailing_shrinkage_covariance(values, times, decision_time=times[0])
    with pytest.raises(ValueError, match="timezone"):
        trailing_shrinkage_covariance(values, times, decision_time="2020-01-03")
    with pytest.raises(ValueError, match="before its outcome"):
        trailing_shrinkage_covariance(values, times, decision_time=times[1],
                                     observed_at=[times[1], times[1]])
    with pytest.raises(ValueError, match="finite"):
        trailing_shrinkage_covariance([[np.nan], [.2]], times,
                                     decision_time="2020-01-03T00:00:00Z")


def test_covariance_excludes_simultaneously_available_outcome_and_rejects_overflow():
    times = [f"2020-01-0{day}T00:00:00Z" for day in (1, 2, 3)]
    fit = trailing_shrinkage_covariance([[.1], [.3], ["PROTECTED"]], times,
                                       decision_time=times[2])
    assert fit.rows_used == 2
    np.testing.assert_allclose(fit.mean, [.2])
    with pytest.raises(ValueError, match="overflowed"):
        trailing_shrinkage_covariance([[1e308], [0], [0]], times,
                                     decision_time=times[2])


def test_policy_rejects_compounding_and_objective_overflow():
    with pytest.raises(ValueError, match="compounding overflowed"):
        plan_portfolio([[[1e308], [1e308]]], [0])
    with pytest.raises(ValueError, match="objective overflowed"):
        plan_portfolio([[[0]]], [0], terminal_target=1e308, shortfall_penalty=10)


def test_covariance_singular_history_remains_psd_without_invented_ridge():
    result = trailing_shrinkage_covariance(
        [[0, 0], [0, 0]], ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
        decision_time="2020-01-03T00:00:00Z")
    np.testing.assert_array_equal(result.covariance, np.zeros((2, 2)))


def test_scenarios_reproducible_joint_heavy_tail_and_explicit_assumptions():
    kwargs = {"location": [0, .01], "covariance": [[.04, .02], [.02, .04]],
              "horizon": 3, "paths": 96, "seed": 27, "currency": "USD", "asset_ids": ["A", "B"]}
    first, second = sample_joint_paths(**kwargs), sample_joint_paths(**kwargs)
    np.testing.assert_array_equal(first.simple_returns, second.simple_returns)
    assert first.simple_returns.shape == (96, 3, 2)
    assert np.all(first.simple_returns >= -1)
    assert first.assumptions["probability_status"].endswith("not calibrated")
    assert first.assumptions["covariance_scope"] == "pre-floor and pre-jump t shocks"


def test_scenario_total_loss_is_absorbing_and_student_t_not_exponentiated():
    paths = sample_joint_paths([-1], [[0]], horizon=3, paths=4, jump_probability=0)
    np.testing.assert_array_equal(paths.simple_returns[:, 0, 0], -np.ones(4))
    np.testing.assert_array_equal(paths.simple_returns[:, 1:, 0], np.zeros((4, 2)))
    defaults = sample_joint_paths([.1], [[0]], horizon=2, paths=4,
                                 jump_probability=1, jump_mean=-2, jump_scale=0)
    np.testing.assert_array_equal(defaults.simple_returns[:, 0, 0], -np.ones(4))
    for kwargs in ({"df": 2}, {"covariance": [[-1]]}, {"currency": "usd"}):
        options = {"location": [0], "covariance": [[1]]}
        options.update(kwargs)
        with pytest.raises(ValueError):
            sample_joint_paths(**options)


def _mixture():
    return {
        "mean": np.array([[.1, -.1], [.2, -.2]]),
        "scale": np.full((2, 2), .03),
        "df": np.array([[5., 8.], [5., 8.]]),
        "weights": np.array([[.4, .6], [.4, .6]]),
    }


def test_forecast_bridge_bounded_reproducible_and_policy_compatible():
    options = {"lower_log": -.3, "upper_log": .4, "paths": 32, "horizon": 3,
               "currency": "USD", "asset_ids": ["A", "B"], "seed": 17}
    first = sample_forecast_paths(_mixture(), [[1, .5], [.5, 1]], **options)
    second = sample_forecast_paths(_mixture(), [[1, .5], [.5, 1]], **options)
    np.testing.assert_array_equal(first.simple_returns, second.simple_returns)
    assert first.simple_returns.shape == (32, 3, 2)
    assert np.all(first.simple_returns >= np.expm1(-.3))
    assert np.all(first.simple_returns <= np.expm1(.4))
    assert sum(first.assumptions["component_counts"]) == 96
    assert first.assumptions["expectation_scope"].startswith("bounded approximation")
    plan = plan_portfolio(first, [0, 0], random_candidates=0)
    assert plan.utility >= plan.no_trade_utility
    assert plan.assumptions["currency"] == "USD"


def test_forecast_bridge_shared_components_and_clipping_are_explicit():
    forecast = _mixture()
    forecast["scale"][:] = 1e-300
    result = sample_forecast_paths(forecast, np.eye(2), lower_log=-.15, upper_log=.15,
                                   horizon=2, paths=64, seed=19)
    logs = np.log1p(result.simple_returns)
    # Near-deterministic expert means have common signs across both assets;
    # independent per-asset expert choices would violate this invariant.
    np.testing.assert_array_equal(np.sign(logs[:, :, 0]), np.sign(logs[:, :, 1]))
    np.testing.assert_allclose(np.abs(logs[:, :, 0]), .1)
    np.testing.assert_allclose(np.abs(logs[:, :, 1]), .15)
    assert result.assumptions["clipped_below"] + result.assumptions["clipped_above"] == 128


def test_forecast_bridge_perfect_joint_correlation_matches_normalized_shocks():
    forecast = _mixture()
    forecast["weights"][:] = [1, 0]
    forecast["scale"][1] = [.06, .06]
    result = sample_forecast_paths(forecast, np.ones((2, 2)), lower_log=-10, upper_log=10,
                                   horizon=1, paths=16, seed=20)
    logs = np.log1p(result.simple_returns[:, 0])
    np.testing.assert_allclose((logs[:, 0] - .1) / .03, (logs[:, 1] - .2) / .06,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("field,replacement,message", [
    ("weights", [[.4, .6], [.5, .5]], "identical expert weights"),
    ("df", [[5, 8], [6, 8]], "identical df"),
    ("scale", [[0, 1], [1, 1]], "positive scales"),
])
def test_forecast_bridge_rejects_incompatible_joint_mixtures(field, replacement, message):
    forecast = _mixture()
    forecast[field] = replacement
    with pytest.raises(ValueError, match=message):
        sample_forecast_paths(forecast, np.eye(2), lower_log=-1, upper_log=1, horizon=1)


@pytest.mark.parametrize("lower,upper", [(float("nan"), 1), (-1, float("inf")),
                                         (1, -1), (-100, 1), (-1, 1000)])
def test_forecast_bridge_rejects_undefined_or_unrepresentable_support(lower, upper):
    with pytest.raises(ValueError):
        sample_forecast_paths(_mixture(), np.eye(2), lower_log=lower, upper_log=upper, horizon=1)


def test_forecast_bridge_rejects_indefinite_correlation():
    with pytest.raises(ValueError, match="positive semidefinite"):
        sample_forecast_paths(_mixture(), [[1, 2], [2, 1]], lower_log=-1, upper_log=1, horizon=1)


def test_exact_empirical_cvar_fractional_tail_and_boundary():
    assert empirical_cvar([1, 2, 3], .5) == pytest.approx(8 / 3)
    assert empirical_cvar([1, 2, 3], 0) == pytest.approx(2)
    assert empirical_cvar([1, 2, 3], .99) == 3
    assert empirical_cvar([1e308, 1e308], 0) == 1e308
    with pytest.raises(ValueError):
        empirical_cvar([1], 1)


def test_self_financing_fee_equation_and_purchase_fee_denominator():
    remaining, turnover = self_financing_rebalance([1], [0], 100)
    assert remaining == pytest.approx(1 / 1.01)
    assert turnover == pytest.approx(remaining)
    remaining, turnover = self_financing_rebalance([.1, .6], [.7, .1], 30)
    assert remaining + .003 * turnover == pytest.approx(1)
    assert turnover == pytest.approx(abs(.1 * remaining - .7) + abs(.6 * remaining - .1))
    assert self_financing_rebalance([.4], [.4], 100) == (1, 0)


def test_multiperiod_policy_values_rebound_path_over_greedy_first_period():
    # Asset A loses 20% then gains 50% (net 20%); B gains 10% then loses 10%.
    values = np.broadcast_to([[-.2, .1], [.5, -.1]], (4, 2, 2)).copy()
    plan = plan_portfolio(values, [0, 0], max_weight=1, min_cash=0, cost_bps=0,
                          random_candidates=0, cvar_penalty=0, shortfall_penalty=0)
    np.testing.assert_array_equal(plan.weights, [1, 0])
    assert plan.mean_terminal_wealth == pytest.approx(1.2)
    assert plan.utility == pytest.approx(np.log(1.2))
    assert plan.assumptions["solver"].endswith("no global optimum guarantee")


def test_total_loss_retained_as_zero_wealth_and_negative_infinite_log():
    plan = plan_portfolio(np.array([[[-1]], [[-1]]]), [1], max_weight=1, min_cash=0,
                          cost_bps=0, turnover_cap=0, random_candidates=0)
    assert plan.selected_candidate == "no_trade"
    np.testing.assert_array_equal(plan.terminal_wealth, [0, 0])
    assert plan.utility == -np.inf
    assert plan.cvar_loss == 1
    assert plan.mean_terminal_shortfall == 1


def test_transaction_cost_makes_no_trade_preferable_and_is_reported():
    values = np.full((4, 1, 1), .001)
    plan = plan_portfolio(values, [0], max_weight=1, min_cash=0, cost_bps=100,
                          random_candidates=0)
    assert plan.selected_candidate == "no_trade"
    assert plan.utility == plan.no_trade_utility == 0
    assert plan.turnover == 0


def test_constraints_apply_to_actual_trades_and_post_fee_weights():
    values = np.full((5, 2, 2), .1)
    plan = plan_portfolio(values, [.1, .1], max_weight=.4, min_cash=.3,
                          turnover_cap=.16, cost_bps=100, random_candidates=6)
    assert plan.weights.sum() <= .7 + 1e-12
    assert np.all(plan.weights <= .4 + 1e-12)
    assert plan.turnover <= .16 + 1e-12
    assert plan.post_cost_wealth_fraction + .01 * plan.turnover == pytest.approx(1)
    assert plan.utility >= plan.no_trade_utility
    with pytest.raises(ValueError, match="mandate repair"):
        plan_portfolio(values, [.6, 0], max_weight=.4)


def test_tail_penalty_rejects_mean_winner_and_cash_accrues_over_horizon():
    values = np.array([[[.5]], [[.5]], [[.5]], [[-1]]])
    plan = plan_portfolio(values, [0], max_weight=1, min_cash=0, cost_bps=0,
                          random_candidates=0, cvar_penalty=100, shortfall_penalty=0)
    np.testing.assert_array_equal(plan.weights, [0])
    cash = plan_portfolio(np.zeros((4, 2, 1)), [0], min_cash=1, cost_bps=0,
                          cash_returns=[.02, .03], random_candidates=0)
    assert cash.mean_terminal_wealth == pytest.approx(1.02 * 1.03)


def test_buy_hold_accounting_matches_independent_asset_cash_compounding():
    returns = np.array([[[.1, -.2], [-.1, .3]], [[-.2, .4], [.1, -.3]]])
    plan = plan_portfolio(returns, [.3, .2], max_weight=.5, min_cash=.2,
                          turnover_cap=0, cost_bps=17, cash_returns=.01, random_candidates=0)
    expected = np.array([.3 * 1.1 * .9 + .2 * .8 * 1.3 + .5 * 1.01**2,
                         .3 * .8 * 1.1 + .2 * 1.4 * .7 + .5 * 1.01**2])
    np.testing.assert_allclose(plan.terminal_wealth, expected)
    assert plan.post_cost_wealth_fraction == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.001])
def test_policy_rejects_invalid_return_contract(bad):
    with pytest.raises(ValueError):
        plan_portfolio([[[bad]]], [0])
