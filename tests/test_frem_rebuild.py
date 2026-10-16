"""Numerical, information-boundary and serialization checks for FREM-S2."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from test_model_contracts import windows

from frem.frozen_residual import FrozenResidualForecaster, choose_calibration, residual_partitions
from frem.model import FREMModel, ModelConfig
from frem.training import Trainer, TrainingConfig
from scripts.public_equity_backtests import monthly_prices


def test_public_zero_quote_quarantines_entire_month_and_keeps_default_rejection():
    dates = pd.to_datetime(["2003-04-30", "2003-05-19", "2003-05-30", "2003-06-30"], utc=True)
    chart = {
        "timestamp": (dates.as_unit("ns").asi8 // 1_000_000_000).tolist(),
        "meta": {"exchangeTimezoneName": "UTC"},
        "indicators": {
            "adjclose": [{"adjclose": [10, 0, 11, 12]}],
            "quote": [
                {
                    **{key: [10, 0, 11, 12] for key in ("open", "high", "low", "close")},
                    "volume": [100, 0, 100, 100],
                }
            ],
        },
    }
    cfg = {"source_start": "2003-01-01", "source_end_exclusive": "2004-01-01"}
    with pytest.raises(ValueError, match="Nonpositive"):
        monthly_prices(chart, cfg)
    price = monthly_prices(chart, cfg, quarantine_zero_quote_months=True)
    assert np.isnan(price.loc["2003-05"])
    assert price.loc["2003-06"] == 12
    assert np.isnan(np.log(price / price.shift(1)).loc["2003-06"])
    assert price.attrs["quarantined_zero_quote_months"] == ["2003-05"]
    chart["indicators"]["quote"][0]["volume"][1] = 1
    with pytest.raises(ValueError, match="Nonpositive"):
        monthly_prices(chart, cfg, quarantine_zero_quote_months=True)


def config(**kwargs):
    return ModelConfig(
        input_dim=3,
        context_length=8,
        horizon=2,
        hidden_dim=8,
        reservoir_dim=8,
        spectral_bands=3,
        memory_capacity=64,
        dropout=0.1,
        **kwargs,
    )


def test_standardized_initial_forecast_matches_training_moments_and_horizon_units():
    model = FREMModel(config(return_parameterization="train_standardized", use_spectral=False))
    x = torch.randn(4, 8, 3)
    with pytest.raises(RuntimeError, match="reference"):
        model(x)
    y = torch.tensor([-0.02, 0.01, 0.03, 0.06])
    model.fit_return_reference(y)
    out = model.eval()(x)
    torch.testing.assert_close(out["base_prediction"], y.mean().expand(4))
    expected = y.std(unbiased=False) * np.sqrt(3 / 5) + model.config.scale_floor
    torch.testing.assert_close(out["scale"], expected.expand(4))
    with pytest.raises(RuntimeError, match="already fitted"):
        model.fit_return_reference(y * 100)


def test_standardized_reference_rejects_invalid_targets_and_handles_constant_targets():
    model = FREMModel(config(return_parameterization="train_standardized", use_spectral=False))
    with pytest.raises(ValueError, match="finite"):
        model.fit_return_reference(torch.tensor([float("nan")]))
    model.fit_return_reference(torch.ones(3))
    assert float(model.return_unit) == pytest.approx(model.config.scale_floor)
    assert torch.isfinite(model.eval()(torch.zeros(1, 8, 3))["scale"]).all()


def test_legacy_state_dict_has_no_new_required_buffers():
    original = FREMModel(config())
    assert not any(
        key.startswith("return_reference") or key in {"return_center", "return_unit"}
        for key in original.state_dict()
    )
    FREMModel(config()).load_state_dict(original.state_dict(), strict=True)


def test_reference_uses_training_targets_only_and_checkpoint_roundtrips(tmp_path):
    train, validation, later = windows(20), windows(8, start=24), windows(5, start=40)
    validation.y[:] = 0.9
    trainer = Trainer(
        config(return_parameterization="train_standardized", use_memory=False),
        TrainingConfig(epochs=1, batch_size=16, seed=11),
    )
    trainer.fit(train, validation)
    assert float(trainer.model.return_center) == pytest.approx(train.y.mean(), abs=1e-8)
    assert float(trainer.model.return_unit) == pytest.approx(train.y.std(), abs=1e-8)
    assert trainer.history[0]["gradient_clipped"] >= 0
    trainer.save(tmp_path / "scaled")
    pd.testing.assert_frame_equal(trainer.predict(later), Trainer.load(tmp_path / "scaled").predict(later))


def test_residual_split_purges_all_label_overlap():
    data = windows(50)
    core, selection, residual = residual_partitions(data)
    assert core.label_time.max() < selection.time.min()
    assert selection.label_time.max() < residual.time.min()
    assert set(core.time).isdisjoint(residual.time)


def test_nll_calibration_can_disable_harmful_memory_and_uses_finite_declared_grid():
    frame = pd.DataFrame(
        {
            "actual": [0.0] * 8,
            "prediction": [0.0] * 8,
            "scale": [0.1] * 8,
            "student_df": [5.0] * 8,
            "time": np.arange(8) * 86_400_000_000_000,
            "asset_id": [0] * 8,
        }
    )
    choice = choose_calibration(frame, np.ones(8), use_memory=True)
    assert choice["selected"]["weight"] == 0.0
    assert len(choice["candidates"]) == 20
    assert len(choose_calibration(frame, np.ones(8), use_memory=False)["candidates"]) == 5


def test_frozen_residual_reference_is_out_of_sample_and_predictions_ignore_future(tmp_path):
    train, validation, later = windows(50), windows(8, start=55), windows(5, start=70)
    model = FrozenResidualForecaster(config(), TrainingConfig(epochs=1, batch_size=16, seed=11))
    model.fit(train, validation)
    _, _, residual = residual_partitions(train)
    base = model.base.predict(residual)
    assert model.base.fitted_through < residual.time.min()
    by_key = {(int(r.time), int(r.asset_id)): r for r in base.itertuples()}
    for record in model.memory.records.values():
        row = by_key[(record.decision_time, record.asset_id)]
        assert record.residual == pytest.approx(row.actual - row.prediction)
        assert record.credited_uses == 0
    before = copy.deepcopy(model.memory.to_dict())
    expected = model.predict(later)
    poisoned = copy.deepcopy(later)
    poisoned.y[:] = 1000
    poisoned.future_x[:] = np.nan
    actual = model.predict(poisoned)
    for column in ("prediction", "scale", "student_df", "base_prediction", "memory_residual"):
        np.testing.assert_array_equal(expected[column], actual[column])
    assert model.memory.to_dict() == before
    off = model.predict(later, use_memory=False)
    np.testing.assert_array_equal(off.prediction, off.base_prediction)
    with pytest.raises(ValueError, match="calibration"):
        model.predict(validation)
    model.save(tmp_path / "successor")
    loaded = FrozenResidualForecaster.load(tmp_path / "successor")
    pd.testing.assert_frame_equal(expected, loaded.predict(later))
    with pytest.raises(FileExistsError):
        model.save(tmp_path / "successor")
