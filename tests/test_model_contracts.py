"""Authored regression tests. Not executed during delivery preparation."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from frem.baselines import DLinearRegressor, PatchTransformerRegressor, load_baseline, make_baseline
from frem.data import WindowData
from frem.model import (
    CausalFeatureGate,
    CausalSpectralEncoder,
    FixedReservoir,
    ModelConfig,
    predictive_loss,
    representation_losses,
)
from frem.training import Trainer, TrainingConfig, date_balanced_mean, time_groups

DAY = 86_400_000_000_000


def windows(
    dates: int = 16, assets: int = 2, start: int = 0, context: int = 8, horizon: int = 2, features: int = 3
) -> WindowData:
    generator = np.random.default_rng(47 + start)
    rows = dates * assets
    x = generator.normal(size=(rows, context, features)).astype(np.float32)
    future = generator.normal(size=(rows, horizon, features)).astype(np.float32)
    time = np.repeat((np.arange(dates) + start + 1) * DAY, assets).astype(np.int64)
    identifiers = np.tile(np.arange(assets), dates)
    entry = time + DAY
    exit_time = time + (horizon + 1) * DAY
    return WindowData(
        x=x,
        future_x=future,
        y=0.01 * x[:, -1, 0],
        time=time,
        label_time=exit_time + 1,
        asset_id=identifiers,
        feature_names=[f"feature_{index}" for index in range(features)],
        asset_names=[f"asset_{index}" for index in range(assets)],
        entry_time=entry,
        exit_time=exit_time,
    )


def small_trainer(**overrides: object) -> Trainer:
    model = ModelConfig(
        input_dim=3,
        context_length=8,
        horizon=2,
        hidden_dim=8,
        reservoir_dim=8,
        spectral_bands=3,
        memory_ngram=2,
        memory_capacity=16,
        dropout=0,
        **overrides,
    )
    return Trainer(model, TrainingConfig(epochs=1, batch_size=16, seed=3))


def test_reservoir_is_fixed_and_contracts_initial_state() -> None:
    reservoir = FixedReservoir(3, 10, leak=0.4, contraction=0.8, seed=7)
    assert not list(reservoir.parameters())
    assert float(torch.linalg.matrix_norm(reservoir.recurrent, ord=2)) <= 0.80001
    x = torch.randn(4, 1, 3)
    first = torch.randn(4, 10)
    second = torch.randn(4, 10)
    distance_before = torch.linalg.vector_norm(first - second, dim=1)
    distance_after = torch.linalg.vector_norm(reservoir(x, first)[:, -1] - reservoir(x, second)[:, -1], dim=1)
    assert torch.all(distance_after <= reservoir.contraction_bound * distance_before + 1e-5)


def test_feature_gates_are_prefix_causal() -> None:
    gate = CausalFeatureGate(3).eval()
    prefix = torch.randn(2, 5, 3)
    full = torch.cat((prefix, torch.randn(2, 4, 3) * 100), dim=1)
    for short_output, long_output in zip(gate(prefix), gate(full), strict=True):
        torch.testing.assert_close(short_output, long_output[:, :5])


def test_spectral_peak_and_explicit_reference_requirement() -> None:
    length = 32
    spectral = CausalSpectralEncoder(1, length, length // 2 + 1)
    wave = torch.sin(2 * torch.pi * 4 * torch.arange(length) / length)[None, :, None]
    _, power = spectral.raw_descriptor(wave)
    assert int(power[0, :, 0].argmax()) == 4
    with pytest.raises(RuntimeError, match="reference"):
        spectral(wave)
    spectral.fit_reference([wave, 0.8 * wave])
    features, key, surprise = spectral(wave)
    assert features.shape == (1, spectral.output_dim)
    assert key.shape == (1, spectral.descriptor_dim)
    assert torch.isfinite(surprise).all()
    with pytest.raises(RuntimeError, match="already fitted"):
        spectral.fit_reference([wave])


def test_memory_keys_do_not_drift_with_trainable_parameters() -> None:
    model = small_trainer().model
    x = torch.randn(4, 8, 3)
    before, codes_before = model.make_keys(x)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter))
    after, codes_after = model.make_keys(x)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    torch.testing.assert_close(codes_before, codes_after)
    assert not after.requires_grad


def test_target_encoder_is_detached_and_ema_updates() -> None:
    model = small_trainer(use_spectral=False).model.train()
    x, future = torch.randn(4, 8, 3), torch.randn(4, 2, 3)
    output = model(x)
    target = model.encode_target(future)
    regularizers = representation_losses(output["predicted_latent"], target, output["latent"])
    loss = predictive_loss(output, torch.randn(4) * 0.01) + regularizers["jepa"] + regularizers["variance"]
    loss.backward()
    assert not model.target_encoder.training
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in model.target_encoder.parameters()
    )
    assert model.jepa_predictor[0].weight.grad is not None
    old = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(0.5)
    model.update_ema(0.5)
    assert not torch.equal(old, next(model.target_encoder.parameters()))


def test_singleton_regularizers_are_explicitly_skipped() -> None:
    latent = torch.randn(1, 8, requires_grad=True)
    losses = representation_losses(latent, torch.randn(1, 8), latent)
    assert losses["variance"].item() == 0
    assert losses["covariance"].item() == 0
    assert losses["regularizer_samples"].item() == 0
    losses["jepa"].backward()
    assert latent.grad is not None


@pytest.mark.parametrize(
    "ablation",
    [
        "use_reservoir",
        "use_spectral",
        "use_memory",
        "use_ngram",
        "use_jepa",
        "use_feature_gate",
        "use_stochastic",
        "use_adaptive_controller",
    ],
)
def test_each_mechanism_off_has_finite_outputs(ablation: str) -> None:
    model = small_trainer(**{ablation: False}).model.eval()
    x = torch.randn(3, 8, 3)
    if model.spectral is not None:
        model.spectral.fit_reference([x])
    output = model(x)
    assert output["prediction"].shape == (3,)
    assert torch.isfinite(output["prediction"]).all()
    assert torch.all(output["scale"] > 0)
    assert torch.all(output["df"] > 2)
    if ablation == "use_memory":
        torch.testing.assert_close(output["prediction"], output["base_prediction"])


def test_time_groups_keep_cross_section_together() -> None:
    values = np.asarray([10, 10, 10, 12, 12, 15])
    assert [indices.tolist() for indices in time_groups(values)] == [[0, 1, 2], [3, 4], [5]]
    with pytest.raises(ValueError, match="chronological"):
        list(time_groups(values[::-1]))


def test_validation_score_gives_each_decision_date_equal_weight() -> None:
    losses = np.asarray([1.0, 3.0, 6.0])
    dates = np.asarray([DAY + DAY // 3, DAY + DAY // 2, 2 * DAY + DAY // 3])
    assert date_balanced_mean(losses, dates, np.asarray([0, 1, 0])) == 4.0
    # Extra intraday rows for asset zero do not give that asset more date weight.
    repeated_asset = np.asarray([1.0, 1.0, 3.0, 6.0])
    intraday = np.asarray([DAY + 1, DAY + 2, DAY + 3, 2 * DAY + 1])
    assert date_balanced_mean(repeated_asset, intraday, np.asarray([0, 0, 1, 0])) == 4.0
    with pytest.raises(ValueError, match="finite"):
        date_balanced_mean(np.asarray([np.nan]), np.asarray([10]), np.asarray([0]))


def test_fit_rejects_label_overlap_and_refit() -> None:
    trainer = small_trainer()
    train = windows()
    overlapping_validation = windows(start=15)
    with pytest.raises(ValueError, match="Purge"):
        trainer.fit(train, overlapping_validation)
    with pytest.raises(RuntimeError, match="one chronological split"):
        trainer.fit(train)


def test_frozen_forecast_never_reads_future_and_does_not_mutate_memory() -> None:
    trainer = small_trainer()
    trainer.fit(windows())
    future = windows(dates=5, start=30)
    before = json.dumps(trainer.memory.to_dict(), sort_keys=True)
    forecast = trainer.predict(future)
    modified = copy.deepcopy(future)
    modified.y[:] = np.nan

    class ForbiddenFuture:
        @property
        def shape(self) -> tuple[int, ...]:
            raise AssertionError("Prediction accessed the future target")

    modified.future_x = ForbiddenFuture()  # type: ignore[assignment]
    changed_actual = trainer.predict(modified)
    np.testing.assert_array_equal(forecast.prediction, changed_actual.prediction)
    assert changed_actual.actual.isna().all()
    assert json.dumps(trainer.memory.to_dict(), sort_keys=True) == before
    assert {"student_df", "entry_time", "exit_time"} <= set(forecast)


def test_validation_information_is_in_fitted_through_boundary() -> None:
    trainer = small_trainer()
    train, validation = windows(), windows(dates=4, start=25)
    trainer.fit(train, validation)
    assert trainer.fitted_through == int(validation.label_time.max())
    with pytest.raises(ValueError, match="follow all"):
        trainer.predict(validation)


def test_tensor_checkpoint_preserves_predictions_and_pending_state(tmp_path) -> None:
    trainer = small_trainer()
    trainer.fit(windows())
    future = windows(dates=4, start=30)
    trainer.predict(future, online_update=True)
    trainer.save(tmp_path / "checkpoint")
    restored = Trainer.load(tmp_path / "checkpoint", device="cpu")
    later = windows(dates=3, start=50)
    np.testing.assert_array_equal(trainer.predict(later).prediction, restored.predict(later).prediction)
    assert restored.memory.to_dict() == trainer.memory.to_dict()
    assert restored.controller.to_dict() == trainer.controller.to_dict()


def test_online_future_labels_cannot_change_earlier_predictions(tmp_path) -> None:
    trainer = small_trainer()
    trainer.fit(windows())
    trainer.save(tmp_path / "initial")
    first = Trainer.load(tmp_path / "initial")
    second = Trainer.load(tmp_path / "initial")
    data = windows(dates=8, start=30)
    altered = copy.deepcopy(data)
    altered.y[-4:] += 100.0
    original = first.predict(data, online_update=True)
    changed = second.predict(altered, online_update=True)
    # These outcomes cannot mature anywhere inside the available prediction prefix.
    np.testing.assert_array_equal(original.prediction, changed.prediction)
    with pytest.raises(ValueError, match="previous decision"):
        first.predict(data, online_update=True)
    with pytest.raises(ValueError, match="previous decision"):
        first.predict(data, online_update=False)


def test_wall_budget_cannot_report_an_untrained_success(monkeypatch) -> None:
    trainer = small_trainer()
    monkeypatch.setattr(trainer, "_budget_exceeded", lambda: True)
    with pytest.raises(TimeoutError):
        trainer.fit(windows())
    assert not trainer.fitted


@pytest.mark.parametrize("name", ["zero", "ridge", "reservoir_ridge"])
def test_baseline_forecast_contract_and_checkpoint(name: str, tmp_path) -> None:
    baseline = make_baseline(name, reservoir_dim=8)
    baseline.fit(windows())
    forecast = baseline.predict(windows(dates=4, start=30))
    assert {"prediction", "scale", "student_df", "actual", "entry_time", "exit_time"} <= set(forecast)
    assert np.isfinite(forecast.prediction).all() and (forecast.scale > 0).all()
    baseline.save(tmp_path / name)
    restored = load_baseline(tmp_path / name, trust_joblib=True)
    np.testing.assert_allclose(forecast.prediction, restored.predict(windows(dates=4, start=30)).prediction)


@pytest.mark.parametrize("name", ["dlinear", "patch_transformer"])
def test_compact_sequence_controls_have_scalar_forecasts(name: str) -> None:
    module = DLinearRegressor(3, 8) if name == "dlinear" else PatchTransformerRegressor(3, 8, 8, 3)
    assert module(torch.randn(5, 8, 3)).shape == (5,)
