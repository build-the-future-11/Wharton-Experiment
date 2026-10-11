"""Delayed-feedback and retention tests, now covered by runtime verification."""

from __future__ import annotations

import json

import numpy as np
import pytest

from frem.controller import AdaptiveController, ControllerConfig
from frem.memory import EngramMemory, Retrieval


def schedule(
    memory: EngramMemory,
    decision: int,
    label: int,
    *,
    asset: int = 0,
    actual: float | None = 0.04,
    base: float = 0.0,
    prediction: float = 0.0,
    retrieval=None,
    novelty: float = 1.0,
) -> None:
    memory.schedule(
        key=[1, 0],
        asset_id=asset,
        decision_time=decision,
        label_time=label,
        base_prediction=base,
        prediction=prediction,
        memory_prediction=0.04,
        scale=0.02,
        outcome=actual,
        retrieval=retrieval,
        novelty=novelty,
        ngram_code=3,
    )


def test_pending_labels_are_invisible_until_strict_maturity() -> None:
    memory = EngramMemory(key_dim=2)
    schedule(memory, 1, 5)
    assert memory.retrieve([1, 0], asset_id=0, decision_time=4).identifiers == []
    memory.advance(5)
    assert not memory.records
    assert memory.retrieve([1, 0], asset_id=0, decision_time=5).identifiers == []
    memory.advance(6)
    assert len(memory.records) == 1
    with pytest.raises(ValueError, match="historical time"):
        memory.retrieve([1, 0], asset_id=0, decision_time=5)
    assert len(memory.retrieve([1, 0], asset_id=0, decision_time=6).identifiers) == 1


def test_pre_epoch_timestamps_are_valid_chronological_history() -> None:
    memory = EngramMemory(key_dim=2)
    schedule(memory, -100, -50)
    memory.advance(-49)
    assert len(memory.retrieve([1, 0], asset_id=0, decision_time=-40).identifiers) == 1


def test_same_timestamp_assets_share_only_earlier_outcomes() -> None:
    memory = EngramMemory(key_dim=2, same_asset=False)
    schedule(memory, 1, 10, asset=0)
    schedule(memory, 1, 10, asset=1)
    memory.advance(10)
    assert memory.retrieve([1, 0], asset_id=2, decision_time=10).identifiers == []
    memory.advance(11)
    assert len(memory.retrieve([1, 0], asset_id=2, decision_time=11).identifiers) == 2


def test_asset_and_ngram_filters_are_explicit() -> None:
    memory = EngramMemory(key_dim=2, exact_ngram=True)
    schedule(memory, 1, 2)
    memory.advance(3)
    assert not memory.retrieve([1, 0], asset_id=1, decision_time=4, ngram_code=3).identifiers
    assert not memory.retrieve([1, 0], asset_id=0, decision_time=4, ngram_code=2).identifiers
    assert memory.retrieve([1, 0], asset_id=0, decision_time=4, ngram_code=3).identifiers


def test_utility_credit_is_delayed_and_based_on_forecast_improvement() -> None:
    memory = EngramMemory(key_dim=2, capacity=5, utility_rate=1.0)
    schedule(memory, 1, 2)
    memory.advance(3)
    retrieval = memory.retrieve([1, 0], asset_id=0, decision_time=4)
    identifier = retrieval.identifiers[0]
    schedule(memory, 4, 8, actual=0.04, prediction=0.04, retrieval=retrieval)
    memory.advance(8)
    assert memory.records[identifier].utility == 0
    memory.advance(9)
    assert memory.records[identifier].utility > 0
    assert memory.records[identifier].credited_uses == 1


@pytest.mark.parametrize("policy", ["utility", "fifo", "random", "uniform"])
def test_retention_and_pending_are_bounded(policy: str) -> None:
    memory = EngramMemory(key_dim=2, capacity=3, pending_capacity=2, retention=policy)
    for tick in range(1, 15, 2):
        schedule(memory, tick, tick + 1)
        memory.advance(tick + 2)
        assert len(memory.records) <= 3
    assert memory.total_evicted > 0
    schedule(memory, 20, 25, asset=0)
    schedule(memory, 20, 25, asset=1)
    with pytest.raises(RuntimeError, match="capacity"):
        schedule(memory, 20, 25, asset=2)


def test_delayed_credit_metadata_cannot_exceed_neighbor_capacity() -> None:
    memory = EngramMemory(key_dim=2, neighbors=1)
    retrieval = Retrieval(np.zeros(7, dtype=np.float32), [0, 1], [0.5, 0.5])
    with pytest.raises(ValueError, match="neighbor bound"):
        schedule(memory, 1, 3, retrieval=retrieval)
    assert memory.pending == []


def test_surprise_filtered_writes_do_not_discard_feedback_credit() -> None:
    memory = EngramMemory(key_dim=2, surprise_threshold=2.0)
    schedule(memory, 1, 2, novelty=1.0)
    memory.advance(3)
    assert not memory.records
    assert memory.total_matured == 1
    assert memory.total_write_filtered == 1


def test_live_feedback_must_be_published_and_identified() -> None:
    memory = EngramMemory(key_dim=2)
    schedule(memory, 1, 5, actual=None)
    memory.advance(8)
    assert not memory.records and len(memory.pending) == 1
    with pytest.raises(ValueError, match="publication"):
        memory.provide_feedback(decision_time=1, asset_id=0, outcome=0.1, available_at=10, as_of_time=9)
    memory.provide_feedback(decision_time=1, asset_id=0, outcome=0.1, available_at=10, as_of_time=10)
    memory.advance(10)
    assert not memory.records
    memory.advance(11)
    assert len(memory.records) == 1


def test_roundtrip_preserves_rng_and_read_only_retrieval() -> None:
    memory = EngramMemory(key_dim=2, capacity=2, retention="uniform", seed=13)
    schedule(memory, 1, 2)
    memory.advance(3)
    schedule(memory, 4, 8)
    serialized = json.dumps(memory.to_dict(), sort_keys=True)
    memory.retrieve([1, 0], asset_id=0, decision_time=5)
    assert serialized == json.dumps(memory.to_dict(), sort_keys=True)
    restored = EngramMemory.from_dict(json.loads(serialized))
    restored.advance(9)
    memory.advance(9)
    assert restored.to_dict() == memory.to_dict()


def test_controller_cannot_consume_an_unmatured_outcome() -> None:
    controller = AdaptiveController(ControllerConfig())
    arguments = {
        "actual": 0.1,
        "base_prediction": 0,
        "memory_prediction": 0.1,
        "scale": 0.1,
        "label_time": 10,
        "had_memory": True,
    }
    with pytest.raises(ValueError, match="matured"):
        controller.observe_matured(**arguments, as_of_time=10)
    controller.observe_matured(**arguments, as_of_time=11)
    assert controller.memory_weight > 0.5
    old_rate = controller.learning_rate
    controller.observe_validation(1.0)
    controller.observe_validation(1.2)
    controller.observe_validation(1.3)
    assert controller.learning_rate < old_rate


def test_controller_off_has_no_adaptive_changes() -> None:
    controller = AdaptiveController(ControllerConfig(enabled=False))
    for score in (1.0, 2.0, 3.0, 4.0):
        controller.observe_validation(score)
    assert controller.learning_rate == controller.config.base_learning_rate
    assert controller.jepa_weight == controller.config.base_jepa_weight
    assert controller.memory_weight == 1.0
