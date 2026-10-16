import json

import numpy as np
import pytest

from successors.frem_s6.model import (
    MixtureForecast,
    ModelConfig,
    S6Forecaster,
    date_weights,
    latent_loss_grad,
    student_logpdf,
)


def fixture(context="full", **kwargs):
    rng = np.random.default_rng(101)
    x = rng.normal(0, .02, (20, 5, 2))
    future = np.roll(x, -1, axis=1)
    y = x[:, -1, 0] * .5 + rng.normal(0, .001, 20)
    m = S6Forecaster(ModelConfig(2, state_dim=3, latent_dim=3, rff_dim=3,
                                epochs=5, context=context, **kwargs))
    m.fit(x[:12], future[:12], y[:12], origin_times=np.arange(12),
          label_available=np.arange(12) + 1, future_available=np.arange(12) + 1,
          fit_cutoff=13)
    m.calibrate(x[12:16], y[12:16], origin_times=np.arange(14, 18),
                maturity_times=np.arange(15, 19), available_times=np.arange(15, 19))
    return m, x[16:], y[16:]


def test_latent_gradient_finite_difference():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 4)); future = rng.normal(size=(5, 4))
    y = rng.normal(size=5); w = date_weights([1, 1, 1, 2, 3])
    params = [rng.normal(0, .1, (4, 3)), rng.normal(0, .1, (3, 3)), rng.normal(0, .1, 3)]
    target = rng.normal(0, .1, (4, 3))
    options = {"variance_floor": .7, "jepa_weight": .4, "variance_weight": .3}
    _, gradients, _ = latent_loss_grad(x, future, y, w, *params, target, **options)
    for p, grad in zip(params, gradients, strict=True):
        for index in np.ndindex(p.shape):
            original = p[index]
            p[index] = original + 1e-6
            plus = latent_loss_grad(x, future, y, w, *params, target, **options)[0]
            p[index] = original - 1e-6
            minus = latent_loss_grad(x, future, y, w, *params, target, **options)[0]
            p[index] = original
            assert grad[index] == pytest.approx((plus - minus) / 2e-6, abs=1e-8)


def test_student_density_and_exact_mixture():
    # t_3(0) = 2/(pi sqrt(3)); this is not a matched-variance Gaussian.
    assert float(student_logpdf(0, 0, 1, 3)) == pytest.approx(np.log(2 / (np.pi * np.sqrt(3))))
    f = MixtureForecast(np.array([[0., 1.]]), np.ones((1, 2)), np.full((1, 2), 3),
                        np.array([[.25, .75]]), ("a", "b"))
    pdf = np.exp(student_logpdf(0, np.array([0, 1]), 1, 3))
    assert f.nll([0])[0] == pytest.approx(-np.log(.25 * pdf[0] + .75 * pdf[1]))
    mean, var = f.log_return_moments()
    assert mean[0] == .75
    assert var[0] == pytest.approx(3 + .25 * .75)


def test_fit_serialization_and_delayed_date_feedback(tmp_path):
    model, x, y = fixture()
    first = model.forecast(x, issue_time=20, asset_ids=["a", "b", "c", "d"])
    assert np.isfinite(first.nll(y)).all()
    path = tmp_path / "model.json"
    model.save(path)
    restored = S6Forecaster.load(path)
    second = restored.forecast(x, issue_time=20, asset_ids=["a", "b", "c", "d"])
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.mean, second.mean)
    before = model.forecast(x, issue_time=21, asset_ids=["a", "b", "c", "d"])
    model.observe(issue_time=20, y=y * 100, asset_ids=["a", "b", "c", "d"],
                  maturity_times=[22]*4, available_times=[23]*4)
    after = model.forecast(x, issue_time=21, asset_ids=["a", "b", "c", "d"])
    np.testing.assert_array_equal(before.weights, after.weights)
    with pytest.raises(ValueError, match="once"):
        model.observe(issue_time=20, y=y, asset_ids=["a", "b", "c", "d"],
                      maturity_times=[22]*4, available_times=[23]*4)
    with pytest.raises(FileExistsError):
        model.save(path)


def test_zero_context_really_has_no_sequence_information():
    model, x, _ = fixture(context="zero")
    a = model.forecast(x, issue_time=20, asset_ids=list("abcd"))
    b = model.forecast(x * -700, issue_time=21, asset_ids=list("abcd"))
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.scale, b.scale)
    np.testing.assert_array_equal(a.weights, b.weights)


def test_half_context_odd_window_mutation():
    model, x, _ = fixture(context="half")
    mutated = x.copy(); mutated[:, :3] = 999
    a = model._experts(x); b = model._experts(mutated)
    for left, right in zip(a, b, strict=True):
        np.testing.assert_array_equal(left, right)


def test_fit_rejects_future_target_and_bad_calibration():
    x = np.zeros((4, 3, 1))
    m = S6Forecaster(ModelConfig(1))
    with pytest.raises(ValueError, match="availability"):
        m.fit(x, x, np.zeros(4), origin_times=[1, 2, 3, 4], label_available=[2, 3, 4, 5],
              future_available=[2, 3, 4, 5], fit_cutoff=5)
    m, x, _ = fixture()
    with pytest.raises(ValueError, match="calibration"):
        m.forecast(x, issue_time=18, asset_ids=list("abcd"))


def test_date_weighting_and_config_validation():
    np.testing.assert_allclose(date_weights([1, 1, 2]), [.25, .25, .5])
    for options in ({"df": 2}, {"ema": 1}, {"ripple_delay": -1}, {"epochs": 0}):
        with pytest.raises(ValueError):
            ModelConfig(1, **options)


def test_checkpoint_malformed_scale_rejected(tmp_path):
    m, _, _ = fixture()
    path = tmp_path / "model.json"; m.save(path)
    s = json.loads(path.read_text()); s["target_scale"] = -1
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError):
        S6Forecaster.load(path)
