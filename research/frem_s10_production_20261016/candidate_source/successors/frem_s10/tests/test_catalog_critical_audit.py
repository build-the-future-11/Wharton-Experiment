"""Bounded adversarial closure cases; execution remains resource gated."""

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_experiment_audit import generated_panel
from test_ledger_audit import DAY, OPEN, close, fill, opening
from test_model_audit import BASE, WEEK, feedback_records, record, request, transform

from frem_s10.analysis.bootstrap import campaign_indices, resampled_gains
from frem_s10.arrays import FloatArray, QuantityArray, frozen_f64
from frem_s10.data.calendar import build_calendar_schedule
from frem_s10.data.features import build_origin_records, fit_burnin_transform
from frem_s10.domain import ForecastDistribution
from frem_s10.errors import ChronologyError, ContractError, IntegrityError, NumericalError
from frem_s10.experiment.artifacts import _encode
from frem_s10.experiment.forecasts import build_forecast_tape
from frem_s10.experiment.matrix import MODEL_VARIANTS, CellSpec
from frem_s10.experiment.readback import _decode
from frem_s10.experiment.replay import ReplayInputs, ReplayWindow, prepare_campaign, replay_cell
from frem_s10.identities import JSONObject, canonical_bytes, object_json, parse_json, scientific_digest
from frem_s10.math.fees import solve_rebalance
from frem_s10.math.simplex import DEFAULT_TOLERANCES, _face_candidate, solve_simplex
from frem_s10.metadata import Unit
from frem_s10.model.checkpoint import dump_state, load_state
from frem_s10.model.config import ForecastConfig
from frem_s10.model.engine import advance, initial_state
from frem_s10.model.records import ForecastState, ForecastTransition, scientific_id
from frem_s10.model.stacking import _fit, fit_mixtures
from frem_s10.portfolio.actions import Valuation, enumerate_actions, grid_weights
from frem_s10.portfolio.benchmarks import select_benchmark
from frem_s10.portfolio.ledger import LedgerState, forced_liquidation_open, initial_ledger
from frem_s10.portfolio.policy import evaluate_actions, select_policy


@pytest.mark.parametrize("unit", tuple(Unit))
def test_quantity_domains_reject_nonfinite_and_invalid_values(unit: Unit) -> None:
    """TEST-003: every typed unit rejects nonfinite observations before arithmetic."""
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ContractError):
            QuantityArray(np.array([bad]), unit)
    nonnegative = {
        Unit.DIVIDEND_PER_START_DOLLAR,
        Unit.USD,
        Unit.SHARES,
        Unit.WEIGHT,
        Unit.PROBABILITY,
    }
    if unit in nonnegative | {Unit.PRICE_GROSS, Unit.PRICE_USD}:
        with pytest.raises(ContractError):
            QuantityArray(np.array([-1.0]), unit)
    if unit in (Unit.PRICE_GROSS, Unit.PRICE_USD):
        with pytest.raises(ContractError):
            QuantityArray(np.zeros(1), unit)
    if unit in (Unit.WEIGHT, Unit.PROBABILITY):
        with pytest.raises(ContractError):
            QuantityArray(np.array([1.0 + 2e-12]), unit)


def test_real_bordered_condition_sides_and_original_scale_multiplier() -> None:
    """TEST-033/036: analytic bordered eigenvalues straddle the fixed1e12 limit."""
    a = np.zeros(3)
    epsilon = 2 * math.sqrt(3) * 1e-12
    alpha, certificate = _face_candidate(
        epsilon * np.eye(3), a, (0, 1, 2), scale=1.0, tolerances=DEFAULT_TOLERANCES
    )
    expected_condition = (epsilon + math.sqrt(epsilon**2 + 12)) / (2 * epsilon)
    assert certificate.condition == pytest.approx(expected_condition, rel=2e-4)
    np.testing.assert_allclose(alpha, np.full(3, 1 / 3), atol=1e-5, rtol=0)
    with pytest.raises(NumericalError, match="condition"):
        _face_candidate(epsilon / 4 * np.eye(3), a, (0, 1, 2), scale=1.0, tolerances=DEFAULT_TOLERANCES)
    for constant in (1e-10, 1.0, 1e6):
        fitted = solve_simplex(np.full(3, constant), np.zeros((3, 3)), np.full(3, 1 / 3))
        np.testing.assert_allclose(fitted.alpha, np.full(3, 1 / 3), atol=2e-8, rtol=0)
        chosen = next(face for face in fitted.faces if face.face == fitted.active_face)
        assert chosen.multiplier == pytest.approx(-constant, rel=1e-9, abs=1e-12)


def test_condition_equality_and_singular_solve_fail_without_candidate_skipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TEST-036: exact limit is allowed; next value and solve failure are explicit."""
    for measured in (1e12, float(np.nextafter(1e12, np.inf))):
        monkeypatch.setattr(np.linalg, "cond", lambda *args, value=measured, **kwargs: value)
        if measured == 1e12:
            _, certificate = _face_candidate(
                np.eye(3), np.zeros(3), (0, 1, 2), scale=1.0, tolerances=DEFAULT_TOLERANCES
            )
            assert certificate.certified
        else:
            with pytest.raises(NumericalError, match="condition"):
                solve_simplex(np.zeros(3), np.zeros((3, 3)), np.full(3, 1 / 3))

    def singular(system: FloatArray, rhs: FloatArray) -> FloatArray:
        raise np.linalg.LinAlgError("constructed singular solve")

    monkeypatch.setattr(np.linalg, "cond", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(np.linalg, "solve", singular)
    with pytest.raises(NumericalError, match="singular bordered") as caught:
        solve_simplex(np.zeros(3), np.zeros((3, 3)), np.full(3, 1 / 3))
    assert isinstance(caught.value.__cause__, np.linalg.LinAlgError)


def test_fee_root_near_brackets_and_forced_hundred_iteration_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TEST-043: both certificates bind; a stalled interval never yields a trade."""
    for capital in (1e-4, 100.0, 1e6):
        tolerance = 1e-10 * max(1.0, capital)
        sold = solve_rebalance(np.array([capital, 0.0, 0.0]), 0.0, np.zeros(3), one_way_cost=0.999999)
        assert abs(sold.post_fee_nav - capital * (1 - 0.999999)) <= tolerance
        bought = solve_rebalance(np.zeros(3), capital, np.array([1e-8, 0.0, 0.0]), one_way_cost=0.004)
        assert abs(bought.post_fee_nav - capital / (1 + 0.004e-8)) <= tolerance
        for certificate in (sold, bought):
            assert abs(certificate.residual) <= tolerance and certificate.bracket_width <= tolerance
    calls = 0

    def stalled(condition: object, value: FloatArray, original: FloatArray) -> FloatArray:
        nonlocal calls
        calls += 1
        return original

    monkeypatch.setattr(np, "where", stalled)
    with pytest.raises(NumericalError, match="bracket and accounting"):
        solve_rebalance(np.zeros(3), 100.0, np.array([0.4, 0.0, 0.0]), one_way_cost=0.01)
    assert calls == 200


def test_fixed_shrinkage_reverses_raw_covariance_ranking() -> None:
    """TEST-056: covariance shrinkage has a discriminating portfolio consequence."""
    returns = np.array([[0.01, -0.02, 0.0], [-0.01, 0.02, 0.0], [0.0, 0.0, 0.005], [0.0, 0.0, -0.005]])
    law = ForecastDistribution("joint-shrink-reference", 1 + returns, np.zeros((4, 3)), np.full(4, 0.25))
    covariance = np.array([[1.0, -2.0, 0.0], [-2.0, 4.0, 0.0], [0.0, 0.0, 0.25]]) * 0.00005
    shrink = 0.5 * covariance + 0.5 * np.diag(np.diag(covariance))
    first, second = np.array([0.4, 0.2, 0.2]), np.array([0.35, 0.2, 0.25])
    assert first @ covariance @ first < second @ covariance @ second
    assert first @ shrink @ first > second @ shrink @ second
    targets = grid_weights(total_risky=0.8)
    raw = [sum(float(w[i] * covariance[i, j] * w[j]) for i in range(3) for j in range(3)) for w in targets]
    changed = [sum(float(w[i] * shrink[i, j] * w[j]) for i in range(3) for j in range(3)) for w in targets]
    chosen = select_benchmark(
        "shrink_min_variance80", Valuation(np.zeros(3), 100.0), law, one_way_cost=0.0, initial_entry=True
    )
    assert chosen.action.weights @ shrink @ chosen.action.weights == pytest.approx(min(changed), abs=1e-18)
    assert chosen.action.weights @ covariance @ chosen.action.weights > min(raw) + 1e-8


def test_splitting_or_reordering_atoms_preserves_every_policy_objective() -> None:
    """TEST-025: joint means, covariance and all609 policy objectives survive splitting."""
    g = np.array([[1.03, 0.97, 1.01], [0.98, 1.04, 1.02], [1.01, 1.02, 0.96]])
    d = np.array([[0.001, 0.002, 0.0], [0.0, 0.003, 0.001], [0.002, 0.0, 0.0]])
    laws = (
        ForecastDistribution("original", g, d, np.array([0.2, 0.3, 0.5])),
        ForecastDistribution("split", g[[0, 0, 1, 2]], d[[0, 0, 1, 2]], np.array([0.1, 0.1, 0.3, 0.5])),
        ForecastDistribution("permuted", g[[2, 0, 1]], d[[2, 0, 1]], np.array([0.5, 0.2, 0.3])),
    )
    valuation = Valuation(np.zeros(3), 100.0)
    actions = enumerate_actions(valuation)
    original = evaluate_actions(valuation, laws[0], actions.actions, one_way_cost=0.004)
    mean = laws[0].probabilities @ laws[0].r
    covariance = (laws[0].r - mean).T @ (laws[0].probabilities[:, None] * (laws[0].r - mean))
    for law in laws[1:]:
        candidate = evaluate_actions(valuation, law, actions.actions, one_way_cost=0.004)
        np.testing.assert_allclose(candidate.nominal, original.nominal, atol=1e-15, rtol=1e-13)
        np.testing.assert_allclose(candidate.robust, original.robust, atol=1e-15, rtol=1e-13)
        np.testing.assert_allclose(law.probabilities @ law.r, mean, atol=1e-16, rtol=0)
        np.testing.assert_allclose(
            (law.r - mean).T @ (law.probabilities[:, None] * (law.r - mean)),
            covariance,
            atol=1e-18,
            rtol=1e-14,
        )


def test_feedback_permutation_requires_matched_weights_and_preserves_fit() -> None:
    """TEST-039: matching record/weight permutations preserve the mathematical fit."""
    records = tuple(
        replace(item, a=np.array([0.002 * i, 0.002 * (4 - i), 0.001 * i * i]))
        for i, item in enumerate(feedback_records(5))
    )
    weights = np.array([0.05, 0.1, 0.15, 0.3, 0.4])
    center = np.full(3, 1 / 3)
    original = _fit(records, weights, center, ForecastConfig())
    matched = _fit(records[::-1], weights[::-1], center, ForecastConfig())
    mismatched = _fit(records[::-1], weights, center, ForecastConfig())
    np.testing.assert_allclose(original.alpha, matched.alpha, atol=1e-14, rtol=0)
    assert np.max(np.abs(original.alpha - mismatched.alpha)) > 0.01
    with pytest.raises(ChronologyError, match="ordered"):
        fit_mixtures(records[::-1], np.zeros(12), decision_us=BASE + 100 * WEEK, config=ForecastConfig())


def test_ordinary_hold_and_forced_sale_paths_have_identical_dividend_ownership() -> None:
    """TEST-050: unchanged ordinary shares avoid A4's actual sale/reentry costs."""
    paths: list[LedgerState] = []
    for forced in (False, True):
        state = initial_ledger(
            cell_id="paired", start_us=OPEN - 1, initial_cash=100.0, forced_liquidation_enabled=forced
        )
        state = opening(state).state
        if forced:
            state = forced_liquidation_open(state, event_id="sale0", one_way_cost=0.01).state
        state = fill(state, cost=0.01).state
        entering_shares = float(state.shares[0])
        state = opening(
            close(state).state,
            event_id="open2",
            timestamp_us=OPEN + DAY,
            dividend=0.5,
            dividend_id="paired-dividend",
        ).state
        assert state.dividends[0] == pytest.approx(0.5 * entering_shares)
        if forced:
            state = forced_liquidation_open(state, event_id="sale1", one_way_cost=0.01).state
            state = fill(state, event_id="buy2", cost=0.01).state
        else:
            held = fill(state, event_id="hold2", target=state.risky_dollars / state.nav, cost=0.01, hold=True)
            assert held.event.turnover == held.event.fee == 0
            state = held.state
        state = opening(
            close(state, event_id="close2").state, event_id="open3", timestamp_us=OPEN + 2 * DAY
        ).state
        paths.append(fill(state, event_id="exit", target=(0.0, 0.0, 0.0), cost=0.01, final=True).state)
    ordinary, forced_state = paths
    np.testing.assert_array_equal(ordinary.dividends, forced_state.dividends)
    assert ordinary.nav > forced_state.nav and ordinary.fees < forced_state.fees
    assert ordinary.nav - forced_state.nav == pytest.approx(forced_state.fees - ordinary.fees, abs=5e-8)


def test_future_price_dividend_changes_preserve_all_earlier_states_and_actions() -> None:
    """TEST-017: raw future prices/dividends may change outcomes but not prefix actions."""
    panel, timing = generated_panel(116)
    calendar = build_calendar_schedule(panel.dates)
    origins = build_origin_records(panel, calendar=calendar)
    fitted = fit_burnin_transform(origins)
    selected = tuple(item.anchor_index for item in origins if item.origin_us >= 1199145600000000)
    first, last = selected[0], selected[4]
    opening_prices, closes, dividends = panel.open.copy(), panel.close.copy(), panel.dividends.copy()
    opening_prices[last + 1 :] *= 1.5
    closes[last + 1 :] *= 0.7
    dividends[last + 1 :] *= 3
    changed_panel = replace(panel, open=opening_prices, close=closes, dividends=dividends)
    altered = build_origin_records(changed_panel, calendar=calendar)
    assert fit_burnin_transform(altered).record_id == fitted.record_id
    assert any(not np.array_equal(left.g, right.g) for left, right in zip(origins, altered, strict=True))
    windows = (ReplayWindow("A", (first, last)), ReplayWindow("B", (last,)))
    tapes = tuple(
        build_forecast_tape(records, fitted, retained_anchor_indices=(first, last))
        for records in (origins, altered)
    )
    assert tapes[0].record_id == tapes[1].record_id
    results = []
    for source, records, tape in zip((panel, changed_panel), (origins, altered), tapes, strict=True):
        inputs = ReplayInputs(source, calendar, timing, records, fitted, windows, "GENERATED_ENGINEERING")
        results.append(
            replay_cell(
                inputs,
                replace(tape, evidence_scope="GENERATED_ENGINEERING"),
                CellSpec("A", "strategy", "contextual_stack", 10),
            )
        )
    before, after = results
    assert canonical_bytes(_encode(before.decisions)) == canonical_bytes(_encode(after.decisions))
    cutoff = calendar.anchors[last].quote_us + 2
    assert canonical_bytes(
        _encode(tuple(event for event in before.events if event.timestamp_us <= cutoff))
    ) == canonical_bytes(_encode(tuple(event for event in after.events if event.timestamp_us <= cutoff)))
    assert before.final_state.record_id != after.final_state.record_id


def restore_pair(model: ForecastState, ledger: LedgerState) -> tuple[ForecastState, LedgerState]:
    """Restore a generated pair without an execution/resume route or clock mutation."""
    restored_model = load_state(
        dump_state(model),
        expected_state_id=model.record_id,
        expected_transform_id=model.transform.record_id,
        expected_config=model.config,
    )
    restored_ledger = _decode(parse_json(canonical_bytes(_encode(ledger))))
    assert isinstance(restored_ledger, LedgerState)
    assert restored_ledger.record_id == ledger.record_id
    return restored_model, restored_ledger


def combined_path(*, restore_every_boundary: bool) -> tuple[bytes, tuple[str, ...]]:
    """Use the same generated model and H4 trade schedule with optional restoration."""
    model = initial_state(transform())
    ledger = initial_ledger(cell_id="combined", start_us=BASE + 70 * WEEK - 1, initial_cash=100.0)
    ids: list[str] = []
    for index in range(70, 76):
        incoming = tuple(record(i) for i in range(66)) if index == 70 else (record(index - 5),)
        step = advance(model, request(index), incoming)
        model = step.state
        if restore_every_boundary:
            model, ledger = restore_pair(model, ledger)
            with pytest.raises(ChronologyError):
                advance(model, request(index), ())
        ledger = opening(
            ledger,
            event_id=f"o{index}",
            timestamp_us=BASE + index * WEEK,
            price=10 + (index - 70) / 10,
            dividend=0.2,
            dividend_id=f"d{index}",
        ).state
        if restore_every_boundary:
            model, ledger = restore_pair(model, ledger)
        ids.append(ledger.record_id)
        if index in (70, 74):
            decision = select_policy(
                Valuation(ledger.risky_dollars, ledger.cash),
                step.batch.get("contextual_stack").distribution,
                one_way_cost=0.001,
            )
            ledger = fill(
                ledger,
                event_id=f"t{index}",
                target=decision.action.weights,
                cost=0.001,
                hold=decision.action.is_hold,
            ).state
            if restore_every_boundary:
                model, ledger = restore_pair(model, ledger)
                with pytest.raises(ChronologyError):
                    fill(
                        ledger,
                        event_id=f"t{index}",
                        target=decision.action.weights,
                        cost=0.001,
                        hold=decision.action.is_hold,
                    )
            ids.append(ledger.record_id)
        ledger = close(ledger, event_id=f"c{index}", price=10 + (index - 70) / 10).state
        if restore_every_boundary:
            model, ledger = restore_pair(model, ledger)
            with pytest.raises(ChronologyError):
                close(ledger, event_id=f"c{index}")
        ids.append(ledger.record_id)
    assert len(model.feedback) == 1
    return dump_state(model).encode() + canonical_bytes(_encode(ledger)), tuple(ids)


def test_combined_model_ledger_restore_preserves_every_event_boundary() -> None:
    """TEST-022: exact joint state and duplicate guards survive constructed restores."""
    assert combined_path(restore_every_boundary=False) == combined_path(restore_every_boundary=True)


def test_endianness_canonicalization_and_strict_array_storage_tags() -> None:
    """TEST-072: identity canonicalizes byte order while typed intake rejects guessing."""
    native = np.array([1.25, -0.5, 1e-12])
    swapped = native.astype(">f8")
    assert scientific_id("endian-reference", ("same",), native) == scientific_id(
        "endian-reference", ("same",), swapped
    )
    with pytest.raises(ContractError):
        frozen_f64(swapped, shape=(3,), name="non-native")
    encoded = object_json(_encode(native, owner="LedgerState", name="shares"))
    altered = dict(encoded)
    altered["dtype"] = ">f8"
    with pytest.raises(IntegrityError):
        _decode(altered, owner="LedgerState", name="shares")
    left: JSONObject = {key: index for index, key in enumerate(("a", "b", "c"))}
    right = dict(reversed(tuple(left.items())))
    assert scientific_digest(left) == scientific_digest(right)


def test_separate_process_hash_seed_and_mapping_order_preserve_science(tmp_path: Path) -> None:
    """TEST-088: independent locked subprocesses must emit identical scientific bytes."""
    root = Path(__file__).resolve().parents[1]
    program = """
import json, pathlib, frem_s10
from test_model_audit import first_transition
from frem_s10.model.checkpoint import dump_state
from frem_s10.identities import scientific_digest
import hashlib
assert pathlib.Path(frem_s10.__file__).resolve().is_relative_to(pathlib.Path(__import__('os').environ['S10_TEST_SOURCE']))
step = first_transition()
mapping = {key: key for key in {'alpha', 'beta', 'gamma', 'delta'}}
print(json.dumps({'state': hashlib.sha256(dump_state(step.state).encode()).hexdigest(), 'mapping': scientific_digest(mapping)}, sort_keys=True))
"""
    outputs = []
    for seed in ("17", "913"):
        environment = dict(os.environ)
        environment.update(
            PYTHONHASHSEED=seed,
            PYTHONPATH=os.pathsep.join((str(root / "src"), str(root / "tests"))),
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            S10_TEST_SOURCE=str(root / "src"),
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", program],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]


def test_permuted_asset_feature_payload_cannot_reuse_canonical_origin_identity() -> None:
    """TEST-002: re-signed metadata cannot make a changed feature axis authoritative."""
    panel, timing = generated_panel(574)
    admitted = prepare_campaign(panel, timing)
    first = admitted.origins[0]
    reordered = first.x.reshape(3, 4)[::-1].reshape(12)
    assert not np.array_equal(reordered, first.x)
    with pytest.raises(IntegrityError, match="numeric payload"):
        replace(admitted, origins=(replace(first, x=reordered), *admitted.origins[1:]))


def test_weekly_trace_is_complete_across_retained_fold_boundaries() -> None:
    """TEST-021: independent timestamp sets predict every issue, maturity and pending law."""
    panel, _ = generated_panel(128)
    records = build_origin_records(panel)
    fitted = fit_burnin_transform(records)
    eligible = tuple(
        item
        for item in records
        if item.origin_us >= 1199145600000000
        and sum(prior.available_us < item.origin_us + 1 for prior in records) >= 64
    )
    retained = (eligible[0].anchor_index, eligible[8].anchor_index, eligible[16].anchor_index)
    selected = eligible[:17]
    traces: dict[str, list[int]] = {variant: [] for variant in MODEL_VARIANTS}
    prior_state: dict[str, str] = {}

    def observe(variant: str, step: ForecastTransition) -> None:
        forecast = step.batch.get("contextual_stack")
        current = next(item for item in selected if item.record_id == forecast.request.record_id)
        index = selected.index(current)
        available = tuple(item for item in records if item.available_us < current.origin_us + 1)
        assert step.state.library.record_ids == tuple(
            item.record_id for item in available[-step.state.config.library_window :]
        )
        issued = selected[: index + 1]
        assert step.state.issued_origin_ids == frozenset(item.record_id for item in issued)
        expected_pending = tuple(
            item.record_id for item in issued if item.available_us >= current.origin_us + 1
        )
        assert tuple(item.request.record_id for item in step.state.pending) == expected_pending
        expected_feedback = tuple(
            item.origin_us for item in issued if item.available_us < current.origin_us + 1
        )
        assert tuple(item.origin_us for item in step.state.feedback) == expected_feedback[-104:]
        if variant in prior_state:
            assert step.state.previous_state_id == prior_state[variant]
        prior_state[variant] = step.state.record_id
        traces[variant].append(current.anchor_index)

    tape = build_forecast_tape(records, fitted, retained_anchor_indices=retained, transition_sink=observe)
    assert tape.weekly_origins == 17
    assert len(tape.frames) == 3 * len(MODEL_VARIANTS)
    assert all(trace == [item.anchor_index for item in selected] for trace in traces.values())
    assert dict(tape.final_state_ids) == prior_state


def test_a5_hold_grid_collision_changes_only_special_hold_semantics() -> None:
    """TEST-060: removing hold preserves a coincident grid action and marks the change."""
    valuation = Valuation(np.array([40.0, 0.0, 0.0]), 60.0)
    enabled, disabled = enumerate_actions(valuation), enumerate_actions(valuation, include_hold=False)
    assert enabled.hold_index is not None and disabled.hold_index is None
    assert len(enabled.actions) == len(disabled.actions) == 609
    np.testing.assert_array_equal(
        np.stack([action.weights for action in enabled.actions]),
        np.stack([action.weights for action in disabled.actions]),
    )
    assert enabled.actions[enabled.hold_index].is_hold
    assert not any(action.is_hold for action in disabled.actions)
    law = ForecastDistribution("flat-collision", np.ones((1, 3)), np.zeros((1, 3)), np.ones(1))
    primary = select_policy(valuation, law, one_way_cost=0.001)
    ablation = select_policy(valuation, law, one_way_cost=0.001, include_hold=False)
    np.testing.assert_array_equal(primary.action.weights, ablation.action.weights)
    assert primary.action.is_hold and not ablation.action.is_hold
    assert primary.hold_objective is not None and ablation.hold_objective is None
    assert primary.certificate.action_set_id != ablation.certificate.action_set_id
    assert primary.forecast_id == ablation.forecast_id == law.record_id


def test_labeled_paired_resampling_preserves_asset_method_cost_and_fold_axes() -> None:
    """TEST-062: scalar labeled references expose independently resampled or swapped axes."""
    streams = campaign_indices(8, 9)
    for (fold, block), indices in streams.items():
        count = 8 if fold == "A" else 9
        base = 10000 if fold == "A" else 20000
        labels = np.asarray(
            [
                [
                    [
                        [base + day * 1000 + asset * 100 + method * 10 + cost for cost in range(4)]
                        for method in range(3)
                    ]
                    for asset in range(3)
                ]
                for day in range(count)
            ],
            dtype=np.float64,
        )
        draws = labels[indices[:5]]
        for draw in range(5):
            for position in range(count):
                day = int(indices[draw, position])
                for asset in range(3):
                    for method in range(3):
                        for cost in range(4):
                            assert draws[draw, position, asset, method, cost] == (
                                base + day * 1000 + asset * 100 + method * 10 + cost
                            )
        candidate, control = labels[:, 0, 0, 0], labels[:, 2, 2, 3]
        gains = resampled_gains(candidate, control, indices)
        for draw in range(5):
            chosen = [int(day) for day in indices[draw]]
            reference = 1 - math.fsum(float(candidate[day]) for day in chosen) / math.fsum(
                float(control[day]) for day in chosen
            )
            assert gains[draw] == pytest.approx(reference, abs=2e-15)
        assert block in (2, 4)
