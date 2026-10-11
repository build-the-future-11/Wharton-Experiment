import pytest
import torch
from scipy.stats import t as scipy_student_t

from rejepa.engram import SpectralEngramMemory
from rejepa.model import REJEPA, REJEPAConfig, rejepa_loss


def _model() -> REJEPA:
    torch.manual_seed(7)
    return REJEPA(
        REJEPAConfig(
            input_dim=4,
            reservoir_dim=10,
            latent_dim=8,
            spectral_bands=3,
            worlds=4,
            world_horizon=3,
            interpreters=3,
        )
    )


def test_forward_has_multiple_worlds_and_separate_experts() -> None:
    model = _model()
    context = torch.randn(5, 12, 4)
    future = torch.randn(5, 8, 4)
    generator = torch.Generator().manual_seed(19)
    output = model(context, future_target=future, generator=generator)
    assert output.world_output.trajectories.shape == (5, 4, 3, 8)
    assert output.world_output.interpretations.shape == (5, 3, 8)
    assert torch.allclose(output.world_output.gate.sum(dim=-1), torch.ones(5))
    assert output.return_location.shape == (5,)
    assert output.loss_probability.shape == (5,)
    assert output.risk_volatility.shape == (5,)
    assert output.downside_quantile.shape == (5,)
    assert output.drawdown_exposure.shape == (5,)
    assert (output.risk_volatility > 0).all()
    assert model.growth_expert[-1].weight.data_ptr() != model.protection_expert[-1].weight.data_ptr()


def test_world_generation_is_reproducible_from_predeclared_seed() -> None:
    model = _model()
    context = torch.randn(2, 10, 4)
    a = model(context, generator=torch.Generator().manual_seed(31))
    b = model(context, generator=torch.Generator().manual_seed(31))
    assert torch.equal(a.world_output.trajectories, b.world_output.trajectories)


def test_target_encoder_is_frozen_and_ema_updates() -> None:
    model = _model()
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    before = [parameter.clone() for parameter in model.target_encoder.parameters()]
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    model.update_target_encoder()
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.target_encoder.parameters(), strict=True)
    )


def test_end_to_end_loss_backpropagates_without_rewriting_memory() -> None:
    model = _model()
    memory = SpectralEngramMemory(8, 8)
    memory.write(torch.randn(8), torch.randn(8), 1, 2, 2, 1.0)
    original = memory.records[0].value.clone()
    output = model(
        torch.randn(6, 12, 4),
        future_target=torch.randn(6, 8, 4),
        memory=memory,
        as_of_ns=2,
        generator=torch.Generator().manual_seed(5),
    )
    losses = rejepa_loss(output, torch.randn(6) * 0.02)
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.context_encoder.parameters())
    assert torch.equal(memory.records[0].value, original)


def test_model_rejects_malformed_or_nonfinite_inputs() -> None:
    model = _model()
    with pytest.raises(ValueError, match="input_dim"):
        model(torch.randn(2, 8, 3))
    bad = torch.randn(2, 8, 4)
    bad[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        model(bad)


def test_downside_quantile_uses_the_declared_student_t_distribution() -> None:
    model = _model()
    output = model(torch.randn(2, 8, 4), horizon=torch.tensor([1, 3]))
    multiplier = (output.downside_quantile - output.return_location) / output.return_scale
    expected = scipy_student_t.ppf(model.config.downside_quantile_level, df=model.config.student_df)
    assert torch.allclose(multiplier, torch.full_like(multiplier, expected), atol=1e-5)


def test_fractional_horizons_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite integers"):
        _model()(torch.randn(2, 8, 4), horizon=torch.tensor([1.5, 2.0]))
