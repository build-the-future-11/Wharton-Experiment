from copy import deepcopy

import pytest
import torch

from rejepa.checkpoint import load_checkpoint, model_state_sha256, save_checkpoint
from rejepa.engram import SpectralEngramMemory
from rejepa.model import REJEPA, REJEPAConfig
from rejepa.training import TrainingBatch, TrainingConfig, fit_model, predict_frozen


def _model(**overrides: object) -> REJEPA:
    values: dict[str, object] = {
        "input_dim": 3,
        "reservoir_dim": 6,
        "latent_dim": 5,
        "spectral_bands": 2,
        "worlds": 2,
        "world_horizon": 2,
        "interpreters": 2,
    }
    values.update(overrides)
    torch.manual_seed(13)
    return REJEPA(REJEPAConfig(**values))


def _batches() -> list[TrainingBatch]:
    generator = torch.Generator().manual_seed(17)
    return [
        TrainingBatch(
            context=torch.randn(4, 7, 3, generator=generator),
            future_target=torch.randn(4, 5, 3, generator=generator),
            realized_return=torch.randn(4, generator=generator) * 0.01,
        )
    ]


def test_phase_one_training_runs_and_returns_best_checkpoint() -> None:
    model = _model()
    before = deepcopy(model.state_dict())
    result = fit_model(
        model,
        _batches(),
        _batches(),
        TrainingConfig(epochs=3, patience=2, learning_rate=1e-3),
    )
    assert result.completed_epochs >= 2
    assert result.best_epoch >= 0
    assert torch.isfinite(torch.tensor(result.best_validation_loss))
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert not model.training


def test_frozen_inference_is_repeatable_and_does_not_adapt() -> None:
    model = _model()
    context = torch.randn(2, 8, 3)
    before = deepcopy(model.state_dict())
    first = predict_frozen(model, context, seed=71)
    second = predict_frozen(model, context, seed=71)
    assert torch.equal(first.return_location, second.return_location)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_hash_bound_checkpoint_round_trip_includes_memory(tmp_path) -> None:
    model = _model()
    memory = SpectralEngramMemory(5, 5)
    memory.write(torch.ones(5), torch.arange(5.0), 10, 20, 21, 0.4)
    save_checkpoint(tmp_path, model, metadata={"seed": 11, "phase": "development"}, memory=memory)
    restored, metadata, restored_memory = load_checkpoint(tmp_path)
    assert metadata == {"phase": "development", "seed": 11}
    assert restored_memory is not None
    assert restored_memory.export() == memory.export()
    context = torch.randn(2, 8, 3)
    expected = predict_frozen(model, context, seed=29, memory=memory, as_of_ns=20)
    actual = predict_frozen(restored, context, seed=29, memory=restored_memory, as_of_ns=20)
    assert torch.equal(expected.return_location, actual.return_location)
    assert torch.equal(expected.loss_probability, actual.loss_probability)
    assert model_state_sha256(restored.state_dict()) == model_state_sha256(model.state_dict())


def test_checkpoint_tampering_fails_closed(tmp_path) -> None:
    save_checkpoint(tmp_path, _model(), metadata={"phase": "development"})
    with (tmp_path / "metadata.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="integrity failed"):
        load_checkpoint(tmp_path)


def test_mechanism_controls_are_executable() -> None:
    context = torch.randn(3, 7, 3)
    memory = SpectralEngramMemory(5, 5)
    memory.write(torch.ones(5), torch.ones(5), 1, 2, 2, 1.0)
    control = _model(use_world_model=False, use_engram=False, use_fourier_key=False)
    output = control(context, memory=memory, as_of_ns=2)
    assert output.retrieved_identifiers == ()
    assert torch.equal(output.world_output.combined, output.world_output.trajectories[:, 0, 0, :])
    assert torch.allclose(output.world_output.gate, torch.full((3, 2), 0.5))


def test_single_joint_head_control_is_executable() -> None:
    model = _model(separate_heads=False)
    output = model(torch.randn(3, 7, 3))
    output.return_location.sum().backward()
    assert model.joint_expert[-1].weight.grad is not None
    assert model.growth_expert[-1].weight.grad is None


def test_training_rejects_nonfinite_and_missing_memory_timestamp() -> None:
    bad = _batches()[0]
    bad.context[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        fit_model(_model(), [bad], _batches(), TrainingConfig(epochs=1))
    memory = SpectralEngramMemory(5, 5)
    with pytest.raises(ValueError, match="as_of_ns"):
        fit_model(_model(), _batches(), _batches(), TrainingConfig(epochs=1), memory=memory)


def test_horizon_conditioning_is_carried_through_training_and_inference() -> None:
    batch = _batches()[0]
    conditioned = TrainingBatch(
        context=batch.context,
        future_target=batch.future_target,
        realized_return=batch.realized_return,
        horizon=torch.tensor([1, 2, 3, 4]),
    )
    model = _model(max_horizon=4)
    result = fit_model(model, [conditioned], [conditioned], TrainingConfig(epochs=1, patience=1))
    assert result.completed_epochs == 1
    first = predict_frozen(model, batch.context, seed=3, horizon=1)
    second = predict_frozen(model, batch.context, seed=3, horizon=2)
    assert not torch.equal(first.return_location, second.return_location)
