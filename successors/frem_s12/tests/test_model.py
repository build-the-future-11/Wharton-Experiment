from __future__ import annotations

import copy

import numpy as np
import pytest

from frem_s12.baselines import CausalMLP
from frem_s12.generated import generated_panel
from frem_s12.model import (
    CausalRouter,
    FourierReservoir,
    ModelConfig,
    issue_experts,
    router_features,
    s11_weights,
)
from frem_s12.scores import Law


def test_reservoir_is_prefix_causal() -> None:
    values = generated_panel(months=200).returns
    config = ModelConfig(reservoir_width=16, fourier_width=16)
    original = FourierReservoir(6, config, 11).states(values)
    changed = values.copy()
    changed[150:] *= 100
    perturbed = FourierReservoir(6, config, 11).states(changed)
    np.testing.assert_array_equal(original[:151], perturbed[:151])


def test_experts_are_joint_and_complete() -> None:
    bundle = issue_experts(generated_panel(months=200).returns, config=ModelConfig(), seed=11)
    assert len(bundle.laws) == 5
    assert all(value.atoms.shape[1] == 6 for value in bundle.laws)


def test_s11_weights_are_capped_simplex() -> None:
    losses = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]), (100, 1))
    weights = s11_weights(losses)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.60 + 1e-12


def test_router_burnin_and_cap() -> None:
    config = ModelConfig(reservoir_width=8, fourier_width=8)
    width = config.reservoir_width + 12
    router = CausalRouter(width, config, 11)
    features = np.ones(width)
    np.testing.assert_array_equal(router.weights(features, 0), np.full(5, 0.2))
    assert router.weights(features, 100).max() <= config.router_weight_cap + 1e-12


def test_router_state_roundtrip_after_update() -> None:
    config = ModelConfig(reservoir_width=8, fourier_width=8)
    width = config.reservoir_width + 12
    router = CausalRouter(width, config, 11)
    features = np.linspace(-1, 1, width)
    laws = tuple(
        Law(np.asarray([[index, 0.0], [index + 0.2, 0.1]]), np.asarray([0.5, 0.5])) for index in range(5)
    )
    weights = router.weights(features, 100)
    router.update(features, laws, weights, np.asarray([0.1, 0.2]), enabled=True)
    clone = CausalRouter(width, config, 999)
    clone.load_state_dict(copy.deepcopy(router.state_dict()))
    np.testing.assert_array_equal(router.weights(features, 100), clone.weights(features, 100))


def test_mlp_state_roundtrip() -> None:
    history = generated_panel(months=200).returns
    model = CausalMLP(6, 8, 0.001, 11)
    _, features, hidden = model.predict(history)
    model.update(features, hidden, history[-1])
    clone = CausalMLP(6, 8, 0.001, 999)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    np.testing.assert_array_equal(model.predict(history)[0].atoms, clone.predict(history)[0].atoms)


def test_mlp_residual_is_against_raw_network_mean() -> None:
    history = generated_panel(months=200).returns
    model = CausalMLP(6, 8, 0.001, 11)
    model.residuals = [np.full(6, 10.0) for _ in range(24)]
    _, features, hidden = model.predict(history)
    raw_mean = model.w2 @ hidden + model.b2
    outcome = np.linspace(-0.03, 0.03, 6)
    expected = outcome - raw_mean
    model.update(features, hidden, outcome)
    np.testing.assert_allclose(model.residuals[-1], expected, rtol=0, atol=1e-15)


def test_router_features_shape() -> None:
    assert router_features(np.ones(8), np.empty((0, 5))).shape == (20,)
