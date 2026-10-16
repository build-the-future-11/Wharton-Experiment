"""Remaining small contract boundaries; no market values or maximum-scale profiling."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
from test_data_audit import _burnin_records, _panel
from test_model_audit import BASE, WEEK, feedback_records, first_transition, record, request, transform

from frem_s10.data.features import build_origin_records, fit_burnin_transform
from frem_s10.domain import ForecastDistribution
from frem_s10.errors import ExecutionStateError
from frem_s10.experiment.matrix import campaign_matrix
from frem_s10.model.checkpoint import dump_state
from frem_s10.model.config import ForecastConfig
from frem_s10.model.engine import advance, initial_state, state_identity
from frem_s10.model.experts import issue_experts
from frem_s10.model.library import empty_library, update_library
from frem_s10.model.records import IssueRequest
from frem_s10.portfolio.actions import Valuation, feasible_policy_weights
from frem_s10.portfolio.policy import select_policy
from frem_s10.runtime.consumption import consume


@pytest.mark.parametrize("scale", (0.999e-12, 1e-12, 1.001e-12))
def test_burnin_nonzero_feature_scale_floor_is_strict(scale: float) -> None:
    """TEST-009: threshold equality and sides separated beyond summation roundoff."""
    records = tuple(
        replace(item, x=np.full(12, scale if i % 2 else -scale)) for i, item in enumerate(_burnin_records())
    )
    fitted = fit_burnin_transform(records)
    expected = 1.0 if scale < 1e-12 else scale
    np.testing.assert_allclose(fitted.feature_scale, np.full(12, expected), rtol=1e-15, atol=0)
    assert fitted.record_ids == tuple(item.record_id for item in records)


def test_thirteen_irregular_anchor_intervals_preserve_exact_feature_order() -> None:
    """TEST-013: Tuesday holiday substitutes change elapsed days, not interval membership."""
    dates = tuple(date(2005, 1, 3) + timedelta(weeks=i, days=int(i % 3 == 0)) for i in range(20))
    prices = np.array([[100 + 3 * i, 80 + i * i, 70 + 2 * i] for i in range(20)], dtype=np.float64)
    panel = _panel(dates, prices)
    first = build_origin_records(panel)[0]
    expected: list[float] = []
    for asset in range(3):
        values = [float(np.log(prices[i + 1, asset] / prices[i, asset])) for i in range(13)]
        expected.extend((values[-1], sum(values[-4:]), sum(values), float(np.std(values))))
    np.testing.assert_allclose(first.x, expected, atol=1e-14, rtol=1e-14)


def test_simultaneous_maturities_are_ordered_and_checkpoint_replay_is_identical() -> None:
    """TEST-018/022: input order cannot reorder two delayed issued outcomes."""
    first_outcome = replace(record(70), available_us=BASE + 80 * WEEK)
    second_outcome = replace(record(71), available_us=BASE + 80 * WEEK)

    def issue(index: int) -> IssueRequest:
        return replace(request(index), available_us=BASE + 80 * WEEK)

    first = advance(initial_state(transform()), issue(70), tuple(record(i) for i in range(66)))
    second = advance(first.state, issue(71), (record(66),))
    matured = tuple(
        first_outcome if i == 70 else second_outcome if i == 71 else record(i) for i in range(67, 77)
    )
    forward = advance(second.state, request(81), matured)
    reverse = advance(second.state, request(81), matured[::-1])
    assert len(forward.matured_feedback) == 2
    assert [item.origin_us for item in forward.matured_feedback] == [
        first_outcome.origin_us,
        second_outcome.origin_us,
    ]
    assert dump_state(forward.state) == dump_state(reverse.state)


def test_library_and_feedback_eviction_keep_exact_declared_recent_membership() -> None:
    """TEST-020/061: bounded membership construction, not maximum-size model profiling."""
    records = tuple(record(i) for i in range(521))
    for window in (260, 520):
        library = update_library(
            empty_library(),
            records,
            decision_us=BASE + 530 * WEEK,
            config=ForecastConfig(library_window=window),
        )
        assert library.record_ids == tuple(item.record_id for item in records[-window:])
    first = first_transition()
    feedback = tuple(
        replace(item, origin_us=BASE - WEEK + i, available_us=BASE + i)
        for i, item in enumerate(feedback_records(104))
    )
    state = replace(first.state, feedback=feedback)
    state = replace(state, record_id=state_identity(state))
    next_step = advance(state, request(75), tuple(record(i) for i in range(66, 71)))
    assert len(next_step.state.feedback) == 104
    assert tuple(item.record_id for item in next_step.state.feedback[:-1]) == tuple(
        item.record_id for item in feedback[1:]
    )
    assert next_step.state.feedback[-1].issued_id == first.batch.get("contextual_stack").record_id


@pytest.mark.parametrize(
    "neighbors,count,fallback",
    ((16, 64, None), (64, 64, "INSUFFICIENT_LEAVE_ONE_OUT_CONTEXTS"), (64, 65, None)),
)
def test_neighbor_ablation_keeps_exact_k_and_matched_fallback(
    neighbors: int, count: int, fallback: str | None
) -> None:
    """TEST-061: 16/64-neighbor differences propagate without changing shared atoms."""
    config = ForecastConfig(neighbors=neighbors, concentration_floor=neighbors // 2)
    records = tuple(replace(record(i), x=np.zeros(12)) for i in range(count))
    library = update_library(empty_library(), records, decision_us=BASE + 100 * WEEK, config=config)
    experts = issue_experts(
        library, np.zeros(12), decision_us=BASE + 100 * WEEK, transform=transform(), config=config
    )
    assert experts.fallback_reason == fallback and len(experts.neighbor_indices) == neighbors
    assert np.count_nonzero(experts.probabilities[2]) == (count if fallback else neighbors)


@pytest.mark.parametrize("excess,expected", ((0.0, True), (0.5e-12, True), (2e-12, False)))
def test_asset_and_total_caps_on_both_sides_of_tolerance(excess: float, expected: bool) -> None:
    """TEST-044: tolerance does not silently clip actual drifted holdings."""
    assert feasible_policy_weights(np.array([0.4 + excess, 0.0, 0.0])) is expected
    assert feasible_policy_weights(np.array([0.3, 0.3, 0.2 + excess])) is expected


def test_all_adverse_atoms_select_exact_cash_even_with_zero_nominal_mass() -> None:
    """TEST-047: every nonzero exposure is dominated on a loss-only law."""
    forecast = ForecastDistribution(
        "constructed-adverse",
        np.array([[0.5, 0.6, 0.7], [0.2, 0.3, 0.4]]),
        np.zeros((2, 3)),
        np.array([1.0, 0.0]),
    )
    for robust in (False, True):
        chosen = select_policy(Valuation(np.zeros(3), 100.0), forecast, one_way_cost=0.001, robust=robust)
        np.testing.assert_array_equal(chosen.action.weights, np.zeros(3))
        assert chosen.objective == chosen.fee_solution.fee == 0


def test_cartesian_matrix_is_independently_exact() -> None:
    """TEST-057: explicit scientific axes independently define all192 tuples."""
    forecasts = (
        "unconditional",
        "recent",
        "context",
        "equal",
        "global_stack",
        "contextual_stack",
        "individual_score_mix",
        "zero",
    )
    benchmarks = ("cash", "equal100", "equal80", "buy_hold100", "inverse_vol80", "shrink_min_variance80")
    ablations = (
        "repeat_h1",
        "frozen_state",
        "no_shrink",
        "force_liquidate",
        "no_exact_hold",
        "window260",
        "neighbors16",
        "neighbors64",
    )
    expected = {
        (fold, "strategy", arm, cost, policy)
        for fold, arm, cost, policy in product(("A", "B"), forecasts, (0, 10, 20, 40), ("nominal", "robust"))
    }
    expected |= {
        (fold, "benchmark", arm, cost, "robust")
        for fold, arm, cost in product(("A", "B"), benchmarks, (0, 10, 20, 40))
    }
    expected |= {(fold, "ablation", arm, 10, "robust") for fold, arm in product(("A", "B"), ablations)}
    actual = [(cell.fold_id, cell.kind, cell.arm, cell.cost_bps, cell.policy) for cell in campaign_matrix()]
    assert len(actual) == len(set(actual)) == len(expected) == 192 and set(actual) == expected


def test_concurrent_consumption_claim_has_exactly_one_winner(tmp_path: Path) -> None:
    """TEST-075: concurrent starts cannot both consume an immutable identity."""
    barrier = Barrier(2)

    def attempt(number: int) -> str:
        barrier.wait(timeout=10)
        try:
            consume(tmp_path, operation="precision", freeze_id="e" * 64, attempt_id=f"attempt-{number}")
        except ExecutionStateError:
            return "REFUSED"
        return "CONSUMED"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(attempt, (0, 1)))
    assert sorted(outcomes) == ["CONSUMED", "REFUSED"]
    assert len(tuple(tmp_path.iterdir())) == 1
