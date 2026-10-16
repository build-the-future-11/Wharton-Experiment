"""Independent S6 numerical and temporal adversarial checks.

Constructed inputs test implementation contracts only. They are not market
observations, trading evidence, a causal identification test, or profitability.
"""

import json

import numpy as np
import pytest

from successors.frem_s6.model import ModelConfig, S6Forecaster, latent_loss_grad


def small_model(context="full"):
    rng = np.random.default_rng(703)
    x = rng.normal(0, .03, size=(8, 5, 2))
    future = rng.normal(0, .03, size=x.shape)
    model = S6Forecaster(ModelConfig(2, state_dim=2, latent_dim=2, rff_dim=2,
                                   epochs=3, context=context))
    model.fit(x, future, np.linspace(-.02, .025, 8),
              origin_times=np.arange(8), label_available=np.arange(8) + 1,
              future_available=np.arange(8) + 1, fit_cutoff=10)
    return model, x


def test_manual_latent_gradients_match_independent_central_differences():
    rng = np.random.default_rng(291)
    features = rng.normal(size=(5, 3))
    future = rng.normal(size=(5, 3))
    target = rng.normal(size=5)
    weights = np.array([.1, .1, .2, .3, .3])
    parameters = [rng.normal(0, .2, size=(3, 2)),
                  rng.normal(0, .2, size=(2, 2)), rng.normal(0, .2, size=2)]
    stopped_target = rng.normal(0, .2, size=(3, 2))
    options = {"jepa_weight": .3, "variance_weight": .2, "variance_floor": .7}

    def evaluate():
        return latent_loss_grad(features, future, target, weights, *parameters,
                                stopped_target, **options)

    _, gradients, std = evaluate()
    assert np.all(std < .7)  # Exercise the nonzero variance-penalty derivative.
    delta = 1e-6
    for parameter, expected in zip(parameters, gradients, strict=True):
        actual = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + delta
            positive = evaluate()[0]
            parameter[index] = original - delta
            negative = evaluate()[0]
            parameter[index] = original
            actual[index] = (positive - negative) / (2 * delta)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-8)


def test_odd_half_context_all_experts_ignore_same_prefix():
    model, windows = small_model("half")
    changed = windows.copy()
    # T=5 implies the declared floor(T/2)=2 suffix; indices 0,1,2 are hidden.
    changed[:, :3] += .9
    for original, modified in zip(model._experts(windows), model._experts(changed), strict=True):
        np.testing.assert_array_equal(original, modified)


def test_zero_window_context_all_experts_are_input_independent():
    model, windows = small_model("zero")
    for original, modified in zip(model._experts(windows), model._experts(windows * 999), strict=True):
        np.testing.assert_array_equal(original, modified)


def test_calibration_accepts_release_order_different_from_origin_order():
    model, windows = small_model()
    # The first origin's label releases later. All releases precede issuance.
    model.calibrate(windows[:3], [.01, -.02, .03], origin_times=[11, 12, 13],
                    maturity_times=[12, 13, 14], available_times=[16, 13, 15])
    prediction = model.forecast(windows[:2], issue_time=17, asset_ids=["A", "B"])
    assert np.isfinite(prediction.nll([.01, -.01])).all()


@pytest.mark.parametrize("name,value", [("target_scale", -1),
                                        ("fit_cutoff", float("nan")),
                                        ("calibration_cutoff", 0)])
def test_checkpoint_rejects_invalid_scale_or_temporal_contract(tmp_path, name, value):
    model, windows = small_model()
    model.calibrate(windows[:2], [.01, -.02], origin_times=[11, 12],
                    maturity_times=[12, 13], available_times=[12, 13])
    valid = tmp_path / "original.json"
    model.save(valid)
    state = json.loads(valid.read_text())
    state[name] = value
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        S6Forecaster.load(invalid)
