"""Constructed, non-market causal forecasting and replay evidence."""

import hashlib
import json
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from frem_s10.domain import OriginRecord, Transform
from frem_s10.errors import ChronologyError, ContractError, IntegrityError
from frem_s10.identities import JSONObject, JSONValue, object_json, parse_json
from frem_s10.math.scores import energy_score, score_coefficients
from frem_s10.model.checkpoint import dump_state, load_state
from frem_s10.model.config import ForecastConfig
from frem_s10.model.engine import advance, initial_state
from frem_s10.model.feedback import mature_forecast
from frem_s10.model.records import (
    ForecastState,
    ForecastTransition,
    IssuedForecast,
    IssueRequest,
    MaturedFeedback,
)
from frem_s10.model.stacking import fit_mixtures

WEEK = 7 * 86400 * 1_000_000
BASE = 1_000_000_000_000_000


def record(index: int) -> OriginRecord:
    origin = BASE + index * WEEK
    g = np.array([1 + 0.03 * np.sin(index), 1 + 0.02 * np.cos(index), 1 + 0.01 * np.sin(index / 2)])
    return OriginRecord(
        f"origin-{index:04}",
        index,
        origin,
        origin + 4 * WEEK,
        origin + 4 * WEEK + WEEK // 2,
        origin - WEEK // 2,
        np.full(12, np.sin(index / 10)),
        g,
        np.array([0.001, 0.002, 0.0]),
        np.ones(3) + (g - 1) / 4,
        np.array([0.00025, 0.0005, 0.0]),
    )


def request(index: int) -> IssueRequest:
    item = record(index)
    return IssueRequest(
        item.record_id,
        item.origin_us,
        item.endpoint_us,
        item.available_us,
        item.feature_available_us,
        item.x,
        item.anchor_index,
    )


def transform() -> Transform:
    return Transform(
        np.zeros(12),
        np.ones(12),
        np.full(3, 0.05),
        tuple(f"burnin-{i}" for i in range(64)),
        "constructed-transform",
    )


def first_transition() -> ForecastTransition:
    return advance(initial_state(transform()), request(70), tuple(record(i) for i in range(66)))


def test_full_declared_arms_and_owned_immutable_issued_arrays() -> None:
    first = first_transition()
    assert len(first.batch.forecasts) == 8
    assert first.batch.get("zero").g.shape == (1, 3)
    assert first.batch.mixtures.fallback_reason == "INSUFFICIENT_LOCAL_RECORDS"
    with pytest.raises(ValueError):
        first.batch.get("contextual_stack").g.setflags(write=True)
    with pytest.raises(ValueError):
        first.state.library.contexts[0, 0] = 99


def test_current_target_has_no_issuance_input_and_feedback_is_delayed() -> None:
    first = first_transition()
    issued = first.batch.get("contextual_stack")
    target = record(70)
    with pytest.raises(ChronologyError):
        mature_forecast(issued, target, decision_us=target.available_us, transform=transform())
    feedback = mature_forecast(issued, target, decision_us=target.available_us + 1, transform=transform())
    changed = replace(target, g=target.g * 1.4)
    changed_feedback = mature_forecast(
        issued, changed, decision_us=target.available_us + 1, transform=transform()
    )
    assert issued.record_id == first.batch.get("contextual_stack").record_id
    assert not np.array_equal(feedback.a, changed_feedback.a)
    expected = score_coefficients(
        issued.experts.library.r / transform().score_scale,
        issued.experts.probabilities,
        target.r / transform().score_scale,
    )
    np.testing.assert_array_equal(feedback.a, expected.a)
    np.testing.assert_array_equal(feedback.b, expected.b)
    with pytest.raises(IntegrityError):
        mature_forecast(
            issued,
            replace(target, x=np.zeros(12)),
            decision_us=target.available_us + 1,
            transform=transform(),
        )


def test_issued_forecast_is_not_recomputed_after_library_changes() -> None:
    first = first_transition()
    issued = first.batch.get("contextual_stack")
    expected = mature_forecast(
        issued, record(70), decision_us=record(70).available_us + 1, transform=transform()
    )
    current = first.state
    for index in range(71, 76):
        step = advance(current, request(index), (record(index - 5),))
        current = step.state
    actual = step.matured_feedback[0]
    assert actual.issued_id == issued.record_id
    np.testing.assert_array_equal(actual.a, expected.a)
    np.testing.assert_array_equal(actual.b, expected.b)
    assert len(current.library.record_ids) > len(issued.experts.library.record_ids)


def test_duplicate_cursor_immature_and_omitted_outcomes_fail_without_mutation() -> None:
    first = first_transition()
    original = dump_state(first.state)
    with pytest.raises(ChronologyError):
        advance(first.state, request(70), ())
    with pytest.raises(ChronologyError):
        advance(first.state, request(71), (record(0),))
    with pytest.raises(ChronologyError):
        advance(first.state, request(71), (record(70),))
    with pytest.raises(ChronologyError):
        advance(first.state, request(75), ())
    assert dump_state(first.state) == original


def test_checkpoint_replay_matches_every_forecast_byte_and_final_state() -> None:
    current = first_transition().state
    for index in range(71, 111):
        current = advance(current, request(index), (record(index - 5),)).state
    payload = dump_state(current)
    restored = load_state(
        payload,
        expected_state_id=current.record_id,
        expected_transform_id=current.transform.record_id,
        expected_config=current.config,
    )
    assert dump_state(restored) == payload
    uninterrupted = advance(current, request(111), (record(106),))
    resumed = advance(restored, request(111), (record(106),))
    assert dump_state(uninterrupted.state) == dump_state(resumed.state)
    assert uninterrupted.batch.mixtures.global_certificate is not None
    for left, right in zip(uninterrupted.batch.forecasts, resumed.batch.forecasts, strict=True):
        assert left.record_id == right.record_id
        np.testing.assert_array_equal(left.probabilities, right.probabilities)
    with pytest.raises(IntegrityError):
        load_state(
            payload,
            expected_state_id="wrong",
            expected_transform_id="constructed-transform",
            expected_config=current.config,
        )
    with pytest.raises(IntegrityError):
        load_state(
            payload.replace("origin-0000", "origin-0001", 1),
            expected_state_id=current.record_id,
            expected_transform_id="constructed-transform",
            expected_config=current.config,
        )


def feedback_records(count: int = 50) -> tuple[MaturedFeedback, ...]:
    output: list[MaturedFeedback] = []
    for index in range(count):
        output.append(
            MaturedFeedback(
                f"feedback-{index}",
                f"issued-{index}",
                f"outcome-{index}",
                BASE + index * WEEK,
                BASE + (index + 5) * WEEK,
                np.zeros(12),
                np.array([0.1, 0.2, 0.3]),
                np.zeros((3, 3)),
            )
        )
    return tuple(output)


def test_global_local_and_individual_controls_follow_exact_rules() -> None:
    records = feedback_records()
    mixture = fit_mixtures(records, np.zeros(12), decision_us=BASE + 100 * WEEK, config=ForecastConfig())
    assert mixture.fallback_reason is None
    assert mixture.global_certificate is not None
    assert mixture.local_certificate is not None
    expected_beta = min(0.5, mixture.effective_count / (mixture.effective_count + 64))
    assert mixture.beta == expected_beta
    np.testing.assert_allclose(
        mixture.contextual_alpha,
        (1 - expected_beta) * mixture.global_alpha + expected_beta * mixture.local_alpha,
    )
    expected_individual = np.exp(-np.array([0.1, 0.2, 0.3]))
    expected_individual /= expected_individual.sum()
    np.testing.assert_allclose(mixture.individual_alpha, expected_individual)
    unshrunk = fit_mixtures(
        records, np.zeros(12), decision_us=BASE + 100 * WEEK, config=ForecastConfig(no_shrink=True)
    )
    np.testing.assert_array_equal(unshrunk.contextual_alpha, unshrunk.local_alpha)
    extrapolated = fit_mixtures(records, np.ones(12), decision_us=BASE + 100 * WEEK, config=ForecastConfig())
    assert extrapolated.fallback_reason == "LOCAL_EXTRAPOLATION"
    np.testing.assert_array_equal(extrapolated.contextual_alpha, extrapolated.global_alpha)


def test_ablation_combinations_are_not_silently_admitted() -> None:
    with pytest.raises(ContractError):
        ForecastConfig(repeat_h1=True, neighbors=16, concentration_floor=8)
    with pytest.raises(ContractError):
        # Deliberately cross the runtime boundary with a wrongly typed scalar.
        ForecastConfig(library_window=cast(int, 520.0))


def test_duplicate_is_rejected_after_library_eviction() -> None:
    first = advance(
        initial_state(transform(), ForecastConfig(library_window=260)),
        request(400),
        tuple(record(i) for i in range(396)),
    )
    assert record(0).record_id not in first.state.library.record_ids
    assert record(0).record_id in first.state.consumed_event_ids
    with pytest.raises(ChronologyError):
        advance(first.state, request(401), (record(0),))


def test_changed_pending_metadata_invalidates_state_hash() -> None:
    original = first_transition().state
    altered_request = replace(
        original.pending[0].request, available_us=original.pending[0].request.available_us + 1
    )
    forged = replace(original, pending=(replace(original.pending[0], request=altered_request),))
    with pytest.raises(IntegrityError):
        dump_state(forged)


def test_energy_score_cannot_identify_price_dividend_decomposition() -> None:
    # Equal total returns imply identical ES, while price-only liquidation fees differ.
    g1, d1 = np.array([[1.01, 1.0, 1.0]]), np.zeros((1, 3))
    g2, d2 = np.ones((1, 3)), np.array([[0.01, 0.0, 0.0]])
    outcome = np.zeros(3)
    assert energy_score(g1 + d1 - 1, [1.0], outcome) == energy_score(g2 + d2 - 1, [1.0], outcome)
    difference = ((1 - 0.001) * g2 + d2) - ((1 - 0.001) * g1 + d1)
    np.testing.assert_allclose(difference, [[0.001 * 0.01, 0.0, 0.0]], rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "field,value",
    [
        ("transform", None),
        ("library", None),
        ("config", "not-a-config"),
        ("feedback", {"container": "tuple", "items": [None]}),
        ("pending", {"container": "tuple", "items": [None]}),
        ("cursor_us", "not-an-integer"),
        ("previous_state_id", 5),
        ("consumed_event_ids", {"container": "tuple", "items": []}),
    ],
)
def test_checkpoint_rejects_malformed_nested_state_with_typed_error(field: str, value: JSONValue) -> None:
    state = first_transition().state
    envelope, document = checkpoint_documents(state)
    object_json(document["fields"])[field] = value
    reseal_document(envelope, document)
    with pytest.raises(ContractError):
        load_state(
            json.dumps(envelope),
            expected_state_id=state.record_id,
            expected_transform_id=state.transform.record_id,
            expected_config=state.config,
        )


@pytest.mark.parametrize("mutation", ["request", "expert_library", "duplicate_set", "unhashable_set"])
def test_checkpoint_rejects_malformed_pending_members_and_identity_sets(mutation: str) -> None:
    state = first_transition().state
    envelope, document = checkpoint_documents(state)
    fields = object_json(document["fields"])
    pending_items = array_json(object_json(fields["pending"])["items"])
    pending = object_json(object_json(pending_items[0])["fields"])
    consumed_items = array_json(object_json(fields["consumed_event_ids"])["items"])
    if mutation == "request":
        pending["request"] = None
    elif mutation == "expert_library":
        object_json(object_json(pending["experts"])["fields"])["library"] = None
    elif mutation == "duplicate_set":
        consumed_items.append(consumed_items[0])
    else:
        consumed_items.append({"array": "<f8", "shape": [1], "hex": np.zeros(1).tobytes().hex()})
    reseal_document(envelope, document)
    with pytest.raises(ContractError):
        load_state(
            json.dumps(envelope),
            expected_state_id=state.record_id,
            expected_transform_id=state.transform.record_id,
            expected_config=state.config,
        )


def test_checkpoint_rejects_excessive_json_nesting_with_typed_error() -> None:
    with pytest.raises(ContractError):
        load_state(
            '{"nested":' + "[" * 2000 + "0" + "]" * 2000 + "}",
            expected_state_id="unused",
            expected_transform_id="unused",
            expected_config=ForecastConfig(),
        )


def test_state_constructor_rejects_direct_invalid_members() -> None:
    state = first_transition().state
    with pytest.raises(ContractError):
        # Narrow casts let these tests exercise invalid external values at runtime.
        replace(state, pending=cast(tuple[IssuedForecast, ...], (None,)))
    with pytest.raises(ContractError):
        replace(state, feedback=cast(tuple[MaturedFeedback, ...], [None]))
    with pytest.raises(ContractError):
        replace(state, consumed_event_ids=frozenset())


def array_json(value: JSONValue) -> list[JSONValue]:
    assert isinstance(value, list)
    return value


def checkpoint_documents(state: ForecastState) -> tuple[JSONObject, JSONObject]:
    envelope = object_json(parse_json(dump_state(state)))
    payload = envelope["payload"]
    assert isinstance(payload, str)
    return envelope, object_json(parse_json(payload))


def reseal_document(envelope: JSONObject, document: JSONObject) -> None:
    payload = json.dumps(document)
    envelope["payload"] = payload
    envelope["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
