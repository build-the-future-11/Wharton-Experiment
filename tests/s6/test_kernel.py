"""Executable examples and adversarial checks of S6's declared invariants."""

import copy
import json

import numpy as np
import pytest

from successors.frem_s6.kernel import (
    DelayedEngramMemory,
    QuantizedRippleUpdater,
    ReservoirConfig,
    ReservoirEngramKernel,
    TrainOnlyScaler,
)


def roundtrip(obj):
    return type(obj).from_state_dict(json.loads(json.dumps(obj.state_dict(), allow_nan=False)))


def test_train_scaling_uses_only_declared_training_rows_and_freezes():
    x = np.array([[[1.0, 7], [3, 7]], [[5, 7], [7, 7]]])
    scaler = TrainOnlyScaler().fit(x, row_times=[1, 2], train_end=2)
    np.testing.assert_allclose(scaler.mean_, [4, 7])
    assert scaler.scale_[1] == 1
    transformed = scaler.transform(x)
    np.testing.assert_allclose(transformed.mean(axis=(0, 1)), 0, atol=1e-15)
    np.testing.assert_array_equal(roundtrip(scaler).transform(x), transformed)
    with pytest.raises(ValueError, match="frozen"):
        scaler.fit(x, row_times=[1, 2], train_end=2)
    with pytest.raises(ValueError, match="after train_end"):
        TrainOnlyScaler().fit(x, row_times=[1, 3], train_end=2)


def test_scaler_extreme_values_and_unfitted_roundtrip():
    fresh = roundtrip(TrainOnlyScaler())
    with pytest.raises(ValueError, match="not been fitted"):
        fresh.transform([[1]])
    scale = fresh.fit([[-1e308], [1e308]], row_times=[1, 2], train_end=2)
    np.testing.assert_allclose(scale.transform([[-1e308], [1e308]]), [[-1], [1]])
    assert np.isfinite(scale.transform([[0]])).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_rejected(bad):
    with pytest.raises(ValueError):
        TrainOnlyScaler().fit([[bad]], row_times=[1], train_end=1)
    with pytest.raises(ValueError):
        ReservoirEngramKernel(ReservoirConfig(1)).encode([[[bad]]])
    with pytest.raises(ValueError):
        DelayedEngramMemory(1, 1).add([bad], [1], origin_time=0, maturity_time=1, available_time=1)
    with pytest.raises(ValueError):
        QuantizedRippleUpdater((1,)).observe([bad], step=0)


@pytest.mark.parametrize("kwargs", [
    {"input_dim": True}, {"state_dim": 0}, {"recurrent_norm": 1},
    {"recurrent_norm": -0.1}, {"leaks": ()}, {"leaks": (0,)},
    {"leaks": (1.01,)}, {"rff_dim": -1}, {"seed": -1}, {"input_clip": np.nan},
])
def test_reservoir_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ReservoirConfig(**{"input_dim": 2, **kwargs})


def test_reservoir_boundedness_contraction_and_checkpoint_identity():
    model = ReservoirEngramKernel(ReservoirConfig(3, state_dim=5, seed=9))
    rng = np.random.default_rng(4)
    left = rng.uniform(-1, 1, size=(3, 5))
    right = rng.uniform(-1, 1, size=(3, 5))
    for _ in range(20):
        x = rng.normal(size=3) * 100
        before = np.linalg.norm(left - right, axis=1)
        left, right = model.step(x, left), model.step(x, right)
        assert np.max(np.abs(left)) <= 1
        assert np.all(np.linalg.norm(left - right, axis=1) <= model.contraction_bounds * before + 1e-12)
    windows = rng.normal(size=(4, 7, 3))
    actual = model.encode(windows)
    assert actual.shape == (4, model.output_dim)
    assert np.max(np.abs(actual)) <= max(1, np.sqrt(2 / model.config.rff_dim))
    np.testing.assert_array_equal(roundtrip(model).encode(windows), actual)
    np.testing.assert_array_equal(ReservoirEngramKernel(model.config).encode(windows), actual)


def test_zero_context_is_exactly_input_independent_and_short_context_is_suffix():
    model = ReservoirEngramKernel(ReservoirConfig(2, state_dim=3, rff_dim=0))
    a = np.arange(24).reshape(3, 4, 2)
    np.testing.assert_array_equal(model.encode(a, context="zero"), np.zeros((3, model.output_dim)))
    np.testing.assert_array_equal(model.encode(a * 1000, context=0), model.encode(a, context="zero"))
    np.testing.assert_array_equal(model.encode(a, context=2), model.encode(a[:, -2:]))
    for context in [-1, 5, True, "unknown"]:
        with pytest.raises(ValueError):
            model.encode(a, context=context)
    with pytest.raises(ValueError):
        model.encode(a[0])
    with pytest.raises(ValueError):
        model.step([1, 2], np.full((3, 3), 1.1))


def test_reservoir_checkpoint_cannot_weaken_stability():
    model = ReservoirEngramKernel(ReservoirConfig(1, state_dim=2))
    state = model.state_dict()
    state["recurrent"][0][0][0] = 9
    with pytest.raises(ValueError, match="contraction"):
        ReservoirEngramKernel.from_state_dict(state)


def test_memory_strict_release_boundary_convexity_and_effective_count():
    memory = DelayedEngramMemory(1, 2, bandwidth=0.5)
    memory.add([0], [-2, 1], origin_time=0, maturity_time=1, available_time=2)
    memory.add([0], [4, 9], origin_time=0, maturity_time=2, available_time=3)
    query = [[0], [1]]
    assert memory.retrieve(query, query_time=2).support.tolist() == [0, 0]
    equal_release = memory.retrieve(query, query_time=3)
    np.testing.assert_array_equal(equal_release.values, [[-2, 1], [-2, 1]])
    actual = memory.retrieve(query, query_time=4)
    np.testing.assert_allclose(actual.values, [[1, 5], [1, 5]])
    np.testing.assert_allclose(actual.effective_count, 2)
    np.testing.assert_array_equal(actual.support, [2, 2])
    np.testing.assert_array_equal(roundtrip(memory).retrieve(query, query_time=4).values, actual.values)


def test_memory_fifo_recency_and_rejected_mutations_are_atomic():
    memory = DelayedEngramMemory(1, 1, capacity=2, recency_decay=1)
    for t in range(3):
        memory.add([0], [t], origin_time=t, maturity_time=t + 1, available_time=t + 1)
    read = memory.retrieve([[0]], query_time=4)
    state = copy.deepcopy(memory.state_dict())
    with pytest.raises(ValueError, match="query clock"):
        memory.add([0], [99], origin_time=0, maturity_time=1, available_time=1)
    assert memory.state_dict() == state
    assert 1.5 < read.values[0, 0] < 2
    assert 1 < read.effective_count[0] < 2
    assert read.support[0] == 2
    for maturity, available in [(0, 1), (2, 1), (np.nan, 3)]:
        with pytest.raises(ValueError):
            memory.add([0], [0], origin_time=0, maturity_time=maturity, available_time=available)


def test_future_pending_episode_cannot_evict_mature_memory():
    memory = DelayedEngramMemory(1, 1, capacity=1, max_pending=2)
    memory.add([0], [1], origin_time=0, maturity_time=1, available_time=2)
    np.testing.assert_array_equal(memory.retrieve([[0]], query_time=3).values, [[1]])
    memory.add([0], [99], origin_time=4, maturity_time=5, available_time=6)
    np.testing.assert_array_equal(memory.retrieve([[0]], query_time=3).values, [[1]])
    np.testing.assert_array_equal(memory.retrieve([[0]], query_time=6).values, [[1]])
    np.testing.assert_array_equal(roundtrip(memory).retrieve([[0]], query_time=7).values, [[99]])
    np.testing.assert_array_equal(memory.retrieve([[0]], query_time=7).values, [[99]])
    with pytest.raises(ValueError, match="chronological"):
        memory.retrieve([[0]], query_time=3)


def test_memory_out_of_order_release_pending_capacity_and_checkpoint_validation():
    memory = DelayedEngramMemory(1, 1, capacity=1, max_pending=2)
    memory.add([0], [5], origin_time=0, maturity_time=1, available_time=5)
    memory.add([0], [2], origin_time=1, maturity_time=2, available_time=2)
    before = copy.deepcopy(memory.state_dict())
    with pytest.raises(ValueError, match="pending"):
        memory.add([0], [3], origin_time=2, maturity_time=3, available_time=3)
    assert memory.state_dict() == before
    np.testing.assert_array_equal(memory.retrieve([[0]], query_time=3).values, [[2]])
    # Older origin arriving late does not displace the latest matured origin.
    np.testing.assert_array_equal(memory.retrieve([[0]], query_time=6).values, [[2]])
    bad = memory.state_dict()
    bad["records"][0]["available_time"] = 6
    with pytest.raises(ValueError, match="unreleased"):
        DelayedEngramMemory.from_state_dict(bad)


def test_future_label_changes_do_not_affect_earlier_memory_read():
    first, second = DelayedEngramMemory(1, 1), DelayedEngramMemory(1, 1)
    for memory, future in [(first, 1e9), (second, -1e9)]:
        memory.add([0], [1], origin_time=0, maturity_time=1, available_time=1)
        memory.add([0], [future], origin_time=2, maturity_time=3, available_time=3)
    np.testing.assert_array_equal(first.retrieve([[0]], query_time=3).values,
                                  second.retrieve([[0]], query_time=3).values)


def test_ripple_delay_schedule_and_quantization_conservation():
    updater = QuantizedRippleUpdater((2,), learning_rate=1, quantization_step=0.2,
                                     delay_steps=2, ripple_steps=3, decay=0.5,
                                     max_gradient_norm=100, max_update_norm=100)
    updater.observe([0.73, -0.31], step=0)
    parameters = np.zeros(2)
    np.testing.assert_array_equal(updater.apply(parameters, step=0), parameters)
    np.testing.assert_array_equal(updater.apply(parameters, step=1), parameters)
    for step in range(2, 5):
        parameters = updater.apply(parameters, step=step)
        assert np.max(np.abs(updater.residual)) <= 0.1 + 1e-14
    # Applied quantized sums + final error equal the entire observed gradient.
    np.testing.assert_allclose(-parameters + updater.residual, [0.73, -0.31], atol=1e-15)
    assert not updater.state_dict()["pending"]


def test_ripple_norm_bound_checkpoint_and_time_rejection():
    updater = QuantizedRippleUpdater((2, 2), quantization_step=0.01, max_update_norm=0.02)
    updater.observe(np.full((2, 2), 1e308), step=0)
    updater.apply(np.zeros((2, 2)), step=0)
    restored = roundtrip(updater)
    a = b = np.zeros((2, 2))
    for step in range(1, 5):
        before = a.copy()
        a, b = updater.apply(a, step=step), restored.apply(b, step=step)
        assert np.linalg.norm(a - before) <= 0.02 + 1e-12
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="chronological"):
        updater.apply(a, step=4)
    with pytest.raises(ValueError, match="chronological"):
        updater.observe(np.ones((2, 2)), step=2)


def test_ripple_capacity_and_shape_fail_without_state_mutation():
    updater = QuantizedRippleUpdater((1,), max_pending=3)
    updater.observe([1], step=0)
    before = copy.deepcopy(updater.state_dict())
    with pytest.raises(ValueError, match="capacity"):
        updater.observe([1], step=0)
    with pytest.raises(ValueError, match="shape"):
        updater.apply([1, 2], step=1)
    assert updater.state_dict() == before
    bad = copy.deepcopy(before)
    bad["residual"] = [0.9]
    with pytest.raises(ValueError, match="q/2"):
        QuantizedRippleUpdater.from_state_dict(bad)


def test_ripple_catchup_matches_linear_unclipped_updates():
    sequential = QuantizedRippleUpdater((1,), quantization_step=1e-6, learning_rate=1,
                                        max_update_norm=100)
    sequential.observe([1], step=0)
    catchup = roundtrip(sequential)
    x = np.zeros(1)
    for step in range(1, 4):
        x = sequential.apply(x, step=step)
    np.testing.assert_allclose(x, catchup.apply(np.zeros(1), step=3), atol=1e-6)


def test_immediate_unquantized_ablation_is_actual_gradient_descent():
    updater = QuantizedRippleUpdater((2,), quantization_step=0, delay_steps=0,
                                     ripple_steps=1, learning_rate=0.1, max_update_norm=100)
    updater.observe([1.1, -0.3], step=0)
    np.testing.assert_allclose(updater.apply([1, 2], step=0), [0.89, 2.03])
    np.testing.assert_array_equal(updater.residual, [0, 0])
    assert roundtrip(updater).state_dict() == updater.state_dict()
    with pytest.raises(ValueError, match="chronological"):
        updater.observe([1, 1], step=0)
