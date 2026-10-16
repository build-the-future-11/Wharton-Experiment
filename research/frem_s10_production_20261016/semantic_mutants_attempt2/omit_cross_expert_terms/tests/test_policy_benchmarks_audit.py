"""Constructed independent accounting and finite-decision verification."""

from itertools import product
from typing import cast

import numpy as np
import pytest

from frem_s10.arrays import FloatArray
from frem_s10.domain import ForecastDistribution
from frem_s10.errors import ContractError, NumericalError
from frem_s10.portfolio.actions import Action, Valuation, enumerate_actions, grid_weights
from frem_s10.portfolio.benchmarks import (
    BENCHMARK_NAMES,
    project_capped_simplex,
    select_benchmark,
    unconditional_covariance,
)
from frem_s10.portfolio.policy import evaluate_actions, rank_candidates, select_policy


def forecast(
    g: FloatArray,
    d: FloatArray | None = None,
    p: FloatArray | None = None,
) -> ForecastDistribution:
    return ForecastDistribution(
        "constructed-forecast",
        g,
        np.zeros_like(g) if d is None else d,
        np.full(len(g), 1.0 / len(g)) if p is None else p,
    )


def reference_fee(valuation: Valuation, weights: FloatArray, cost: float) -> tuple[float, float]:
    """Solve the piecewise-linear root by enumerating all eight sign regions."""
    for signs in product((-1.0, 1.0), repeat=3):
        q = (
            valuation.nav
            + cost * sum(s * float(h) for s, h in zip(signs, valuation.risky_dollars, strict=True))
        ) / (1.0 + cost * sum(s * float(w) for s, w in zip(signs, weights, strict=True)))
        deltas = q * weights - valuation.risky_dollars
        if all(s * float(delta) >= -1e-10 for s, delta in zip(signs, deltas, strict=True)):
            return q, float(np.sum(np.abs(deltas)))
    raise AssertionError("no consistent fee sign region")


@pytest.mark.parametrize("cost", [0.0, 0.001, 0.002, 0.004])
@pytest.mark.parametrize("robust", [False, True])
def test_all_policy_candidates_match_scalar_reference(cost: float, robust: bool) -> None:
    valuation = Valuation(np.array([137.0, 213.0, 41.0]), 609.0)
    law = forecast(
        np.array([[1.08, 0.99, 1.02], [0.95, 1.04, 0.98], [1.01, 1.0, 1.03]]),
        np.array([[0.0, 0.006, 0.0], [0.001, 0.0, 0.002], [0.002, 0.001, 0.003]]),
        np.array([0.2, 0.5, 0.3]),
    )
    registry = enumerate_actions(valuation)
    evaluation = evaluate_actions(valuation, law, registry.actions, one_way_cost=cost)
    reference: list[float] = []
    for i, action in enumerate(registry.actions):
        q, trade = (valuation.nav, 0.0) if action.is_hold else reference_fee(valuation, action.weights, cost)
        risky = valuation.risky_dollars if action.is_hold else q * action.weights
        cash = valuation.cash if action.is_hold else q * (1 - float(np.sum(action.weights)))
        wealth = (cash + ((1 - cost) * law.g + law.d) @ risky) / valuation.nav
        utility = (wealth**-3 - 1) / -3
        mean = float(utility @ law.probabilities)
        value = 0.95 * mean + 0.05 * float(np.min(utility)) if robust else mean
        reference.append(value)
        assert evaluation.solutions[i].post_fee_nav == pytest.approx(q, abs=1e-7)
        assert evaluation.solutions[i].turnover == pytest.approx(trade, abs=1e-7)
    values = evaluation.robust if robust else evaluation.nominal
    np.testing.assert_allclose(values, reference, atol=1e-10, rtol=0)
    decision = select_policy(valuation, law, one_way_cost=cost, robust=robust)
    assert len(registry.actions) == 610
    assert decision.certificate.candidate_count == 610
    assert max(reference) - decision.objective <= 1e-10 + 1e-10
    assert 0 <= decision.certificate.optimality_gap <= 1e-10
    assert decision.certificate.constraint_residual <= 1e-12
    assert decision.cash_objective is not None and decision.hold_objective is not None


def test_exact_hold_keeps_actual_dollars_and_cash() -> None:
    valuation = Valuation(np.array([12.3456789, 23.4567891, 0.1234567]), 80.12345)
    law = forecast(np.ones((1, 3)))
    decision = select_policy(valuation, law, one_way_cost=0.0)
    assert decision.action.is_hold
    assert decision.fee_solution.fee == decision.fee_solution.turnover == 0.0
    np.testing.assert_array_equal(decision.fee_solution.risky_dollars, valuation.risky_dollars)
    assert decision.fee_solution.cash.hex() == valuation.cash.hex()
    assert decision.objective == pytest.approx(0, abs=1e-15)
    without_hold = select_policy(valuation, law, one_way_cost=0.0, include_hold=False)
    assert not without_hold.action.is_hold
    assert without_hold.certificate.candidate_count == 609


def test_cash_is_exact_on_flat_zero_cost_path_and_infeasible_hold_is_excluded() -> None:
    law = forecast(np.ones((1, 3)))
    cash = select_policy(Valuation(np.zeros(3), 100_000.0), law, one_way_cost=0.004)
    assert cash.action.is_hold
    assert cash.objective == cash.cash_objective == 0.0
    overweight = select_policy(Valuation(np.array([90.0, 0.0, 0.0]), 10.0), law, one_way_cost=0.0)
    assert overweight.hold_objective is None and not overweight.action.is_hold
    assert overweight.certificate.candidate_count == 609


def test_every_candidate_has_exact_zero_utility_on_zero_cost_flat_atom() -> None:
    valuation = Valuation(np.array([13.37249, 22.67894, 1.000043]), 54.93209)
    registry = enumerate_actions(valuation)
    evaluation = evaluate_actions(valuation, forecast(np.ones((1, 3))), registry.actions, one_way_cost=0.0)
    np.testing.assert_array_equal(evaluation.nominal, np.zeros(len(registry.actions)))
    np.testing.assert_array_equal(evaluation.robust, np.zeros(len(registry.actions)))


def test_zero_mass_adverse_atom_changes_robust_decision() -> None:
    law = forecast(np.array([[1.1, 1.1, 1.1], [0.01, 0.01, 0.01]]), p=np.array([1.0, 0.0]))
    valuation = Valuation(np.zeros(3), 100.0)
    nominal = select_policy(valuation, law, one_way_cost=0.0, robust=False)
    robust = select_policy(valuation, law, one_way_cost=0.0, robust=True)
    assert float(np.sum(nominal.action.weights)) == pytest.approx(0.8)
    assert float(np.sum(robust.action.weights)) < 0.8
    assert nominal.worst_atom_index == 1
    assert robust.robust_utility == robust.objective
    assert nominal.nominal_utility == nominal.objective


def test_price_only_exit_fee_preserves_dividend_cash() -> None:
    valuation = Valuation(np.array([40.0, 0.0, 0.0]), 60.0)
    actions = (Action(valuation.weights, is_hold=True),)
    cost = 0.004
    law = forecast(np.array([[1.1, 1.0, 1.0]]), np.array([[0.2, 0.0, 0.0]]))
    evaluation = evaluate_actions(valuation, law, actions, one_way_cost=cost)
    wealth = (60 + 40 * ((1 - cost) * 1.1 + 0.2)) / 100
    assert evaluation.nominal[0] == pytest.approx((wealth**-3 - 1) / -3, abs=1e-15)
    assert evaluation.solutions[0].fee == 0.0


def test_global_ties_do_not_chain_and_turnover_then_lexicographic_is_order_independent() -> None:
    actions = (
        Action(np.array([0.0, 0.0, 0.0]), is_hold=True),
        Action(np.array([0.1, 0.0, 0.0])),
        Action(np.array([0.0, 0.1, 0.0])),
    )
    values = np.array([0.0, 0.75e-10, 1.5e-10])
    trades = np.array([0.0, 5.0, 5.0 + 0.5e-8])
    chosen, count, trade_count = rank_candidates(values, trades, actions, nav=100.0)
    assert chosen == 2 and count == trade_count == 2
    order = np.array([2, 0, 1])
    permutation = tuple(actions[int(i)] for i in order)
    other, _, _ = rank_candidates(values[order], trades[order], permutation, nav=100.0)
    np.testing.assert_array_equal(permutation[other].weights, actions[chosen].weights)
    hold_values = np.array([0.6e-10, 0.75e-10, 1.5e-10])
    assert rank_candidates(hold_values, trades, actions, nav=100.0)[0] == 0


def reference_projection(target: FloatArray) -> FloatArray:
    candidates: list[FloatArray] = []
    for statuses in product((0, 1, 2), repeat=3):
        free = [i for i, status in enumerate(statuses) if status == 1]
        upper_count = statuses.count(2)
        if free:
            multiplier = (sum(float(target[i]) for i in free) + 0.4 * upper_count - 0.8) / len(free)
            weights = np.array(
                [
                    0.0 if status == 0 else 0.4 if status == 2 else float(target[i]) - multiplier
                    for i, status in enumerate(statuses)
                ]
            )
        else:
            weights = np.array([0.4 if status == 2 else 0.0 for status in statuses])
        if (
            np.all(weights >= -1e-12)
            and np.all(weights <= 0.4 + 1e-12)
            and abs(float(np.sum(weights)) - 0.8) < 1e-12
        ):
            candidates.append(weights)
    return min(candidates, key=lambda weights: float(np.sum((weights - target) ** 2)))


@pytest.mark.parametrize(
    "target",
    [
        (0.8, 0.0, 0.0),
        (0.6, 0.1, 0.1),
        (0.8 / 3, 0.8 / 3, 0.8 / 3),
        (0.0, 0.4, 0.4),
    ],
)
def test_projection_matches_independent_active_face_enumeration(target: tuple[float, float, float]) -> None:
    z = np.asarray(target)
    projection = project_capped_simplex(z)
    np.testing.assert_allclose(projection.weights, reference_projection(z), atol=1e-12, rtol=0)
    assert abs(projection.sum_residual) <= 1e-12
    assert projection.kkt_residual <= 1e-10
    assert projection.iterations <= 100
    with pytest.raises(ValueError):
        projection.weights.setflags(write=True)


@pytest.mark.parametrize(
    "name,total", [("cash", 0.0), ("equal80", 0.8), ("equal100", 1.0), ("buy_hold100", 1.0)]
)
def test_fixed_benchmarks_and_entry_fees(name: str, total: float) -> None:
    decision = select_benchmark(
        name,
        Valuation(np.zeros(3), 100.0),
        forecast(np.ones((1, 3))),
        one_way_cost=0.004,
        initial_entry=True,
    )
    np.testing.assert_array_equal(decision.action.weights, np.full(3, total / 3))
    assert decision.fee_solution.post_fee_nav == pytest.approx(100 / (1 + 0.004 * total), abs=1e-8)
    assert decision.certificate.direction == "fixed"
    assert decision.certificate.candidate_count == 1


def test_buy_hold_preserves_dividends_and_drift_above_policy_caps() -> None:
    valuation = Valuation(np.array([85.0, 10.0, 5.0]), 7.25)
    decision = select_benchmark(
        "buy_hold100",
        valuation,
        forecast(np.ones((1, 3))),
        one_way_cost=0.004,
        initial_entry=False,
    )
    assert decision.action.is_hold
    np.testing.assert_array_equal(decision.fee_solution.risky_dollars, valuation.risky_dollars)
    assert decision.fee_solution.cash == 7.25 and decision.fee_solution.fee == 0
    assert decision.action.weights[0] > 0.4
    assert decision.certificate.constraint_residual == 0


def benchmark_law() -> ForecastDistribution:
    returns = np.array([[0.02, 0.03, 0.09], [-0.02, -0.03, -0.09], [0.0, 0.04, -0.06], [0.0, -0.04, 0.06]])
    return forecast(1.0 + returns)


def test_population_covariance_and_fixed_shrink_grid_are_exact() -> None:
    law = benchmark_law()
    covariance = unconditional_covariance(law)
    reference = sum((np.outer(r, r) for r in law.r), start=np.zeros((3, 3))) / 4
    np.testing.assert_allclose(covariance, reference, atol=1e-17, rtol=0)
    decision = select_benchmark(
        "shrink_min_variance80",
        Valuation(np.zeros(3), 100.0),
        law,
        one_way_cost=0.001,
        initial_entry=True,
    )
    shrink = 0.5 * reference + 0.5 * np.diag(np.diag(reference))
    values = [float(w @ shrink @ w) for w in grid_weights(total_risky=0.8)]
    assert decision.objective == pytest.approx(min(values), abs=1e-10)
    assert decision.certificate.candidate_count == 45
    assert decision.certificate.direction == "min"
    assert decision.certificate.statistic_id is not None


def test_minimum_variance_flat_tie_uses_actual_turnover() -> None:
    valuation = Valuation(np.array([40.0, 40.0, 0.0]), 20.0)
    decision = select_benchmark(
        "shrink_min_variance80",
        valuation,
        forecast(np.ones((1, 3))),
        one_way_cost=0.004,
        initial_entry=False,
    )
    np.testing.assert_array_equal(decision.action.weights, valuation.weights)
    assert decision.fee_solution.turnover == 0
    assert decision.certificate.objective_tie_count == 45


def test_variance_uses_cl16_exact_objective_ties_not_utility_tolerance() -> None:
    actions = (Action(np.array([0.4, 0.4, 0.0])), Action(np.array([0.0, 0.4, 0.4])))
    values, trades = np.array([1e-12, 2e-12]), np.array([80.0, 0.0])
    selected, count, _ = rank_candidates(
        values,
        trades,
        actions,
        nav=100.0,
        maximize=False,
        prefer_hold=False,
        objective_tolerance=0.0,
    )
    assert selected == 0 and count == 1
    reverse, _, _ = rank_candidates(
        values[::-1],
        trades[::-1],
        actions[::-1],
        nav=100.0,
        maximize=False,
        prefer_hold=False,
        objective_tolerance=0.0,
    )
    assert reverse == 1
    deviations = np.diag(np.array([1e-6, 2e-6, 3e-6]))
    law = forecast(1 + np.concatenate((deviations, -deviations)))
    covariance = unconditional_covariance(law)
    shrink = 0.5 * covariance + 0.5 * np.diag(np.diag(covariance))
    targets = grid_weights(total_risky=0.8)
    reference = np.sum((targets @ shrink) * targets, axis=1)
    valuation = Valuation(np.array([0.0, 40.0, 40.0]), 20.0)
    decision = select_benchmark(
        "shrink_min_variance80", valuation, law, one_way_cost=0.004, initial_entry=False
    )
    assert decision.objective == float(np.min(reference))
    assert not np.array_equal(decision.action.weights, valuation.weights)


def test_inverse_volatility_floor_and_projection() -> None:
    law = forecast(np.array([[1.0, 1.01, 1.1], [1.0, 0.99, 0.9]]))
    decision = select_benchmark(
        "inverse_vol80",
        Valuation(np.zeros(3), 100.0),
        law,
        one_way_cost=0.002,
        initial_entry=True,
    )
    inverse = 1 / np.array([1e-4, 0.01, 0.1])
    target = 0.8 * inverse / float(np.sum(inverse))
    np.testing.assert_allclose(decision.action.weights, reference_projection(target), atol=1e-12, rtol=0)
    assert decision.action.weights[0] == 0.4
    assert decision.certificate.projection_kkt_residual is not None
    assert decision.certificate.projection_kkt_residual <= 1e-10


def test_all_six_names_and_immutable_decision_arrays() -> None:
    assert len(BENCHMARK_NAMES) == 6 and len(set(BENCHMARK_NAMES)) == 6
    valuation = Valuation(np.zeros(3), 100.0)
    for name in BENCHMARK_NAMES:
        decision = select_benchmark(name, valuation, benchmark_law(), one_way_cost=0.0, initial_entry=True)
        with pytest.raises(ValueError):
            decision.action.weights.setflags(write=True)
        with pytest.raises(ValueError):
            decision.fee_solution.risky_dollars.setflags(write=True)


def test_failures_are_not_silently_cash_success() -> None:
    valuation = Valuation(np.zeros(3), 100.0)
    law = forecast(np.ones((2, 3)), p=np.array([0.2, 0.8]))
    with pytest.raises(ContractError, match="uniform"):
        select_benchmark("inverse_vol80", valuation, law, one_way_cost=0.0, initial_entry=True)
    with pytest.raises(ContractError, match="unknown"):
        select_benchmark("unknown", valuation, law, one_way_cost=0.0, initial_entry=True)
    with pytest.raises(ContractError, match="boolean"):
        select_policy(valuation, law, one_way_cost=0.0, robust=cast("bool", 1))
    with pytest.raises(ContractError, match="hold"):
        evaluate_actions(valuation, law, (Action(np.full(3, 0.1), is_hold=True),), one_way_cost=0.0)
    with pytest.raises(ContractError):
        project_capped_simplex(np.array([0.5, 0.5, 0.5]))
    with pytest.raises(NumericalError):
        select_policy(Valuation(np.zeros(3), 1e308), forecast(np.full((1, 3), 1e308)), one_way_cost=0.0)
