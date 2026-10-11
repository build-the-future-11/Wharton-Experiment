import torch

from reng.config import ModelConfig
from reng.inference import (
    analytic_forecast,
    calibrated_growth_signal,
    empirical_block_scenarios,
    sample_ensemble_scenarios,
)
from reng.model import RENGJEPA, stage_a_loss


def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    context = torch.randn(2, 4, 128, 18)
    future = torch.randn(2, 4, 21, 18)
    mask = torch.ones(2, 4, dtype=torch.bool)
    scale = torch.full((2, 4), 0.05)
    decision_ns = torch.tensor([100, 200], dtype=torch.int64)
    return context, future, mask, scale, decision_ns


def test_full_model_dimension_contract_and_parameter_ceiling() -> None:
    model = RENGJEPA()
    assert model.config.base_context_dim == 168
    assert model.config.interpreter_dim == 200
    assert model.trainable_parameter_count() <= 1_500_000


def test_stage_a_shapes_and_gradient() -> None:
    model = RENGJEPA()
    context, future, mask, scale, _ = batch()
    output = model.stage_a(context, future, mask, 21)
    assert output.features.base_context.shape == (2, 4, 3, 168)
    assert output.features.target_latent.shape == (2, 4, 32)
    losses = stage_a_loss(output, torch.zeros_like(scale))
    losses["total"].backward()
    assert model.backbone.context_encoder.gru.weight_ih_l0.grad is not None


def test_full_forward_outputs_correlated_return_law() -> None:
    model = RENGJEPA()
    context, future, mask, scale, decision_ns = batch()
    with torch.no_grad():
        outputs = model(
            context,
            future,
            mask,
            scale,
            ("A", "B", "C", "D"),
            decision_ns,
            21,
        )
    assert len(outputs) == 3
    assert outputs[0].gate_logits.shape == (2, 3)
    assert outputs[0].return_mean.shape == (2, 3, 4)
    assert outputs[0].return_factors.shape == (2, 3, 4, 8)
    assert torch.isfinite(outputs[0].return_diagonal).all()
    assert torch.all(outputs[0].return_diagonal > 0)


def test_uniform_gate_ablation_is_exact() -> None:
    config = ModelConfig(uniform_gate=True)
    model = RENGJEPA(config)
    context, future, mask, scale, decision_ns = batch()
    with torch.no_grad():
        output = model(context, future, mask, scale, ("A", "B", "C", "D"), decision_ns, 21)[0]
    assert torch.equal(output.gate_logits, torch.zeros_like(output.gate_logits))
    assert torch.allclose(torch.softmax(output.gate_logits, -1), torch.full((2, 3), 1 / 3))


def test_decoder_projection_enforces_norm() -> None:
    member = RENGJEPA().members[0]
    with torch.no_grad():
        member.decoder.fill_(100)
        member.project_decoder()
    assert torch.linalg.vector_norm(member.decoder, dim=-1).max() <= 5.0 + 1e-6


def test_analytic_quantiles_scenarios_and_growth_fallback() -> None:
    model = RENGJEPA()
    context, future, mask, scale, decision_ns = batch()
    with torch.no_grad():
        outputs = model(context, future, mask, scale, ("A", "B", "C", "D"), decision_ns, 21)
    forecast = analytic_forecast(outputs)
    assert forecast.quantiles.shape == (4, 7)
    assert torch.isfinite(torch.from_numpy(forecast.quantiles)).all()
    scenarios = sample_ensemble_scenarios(outputs, count_per_member=10)
    assert scenarios.shape == (30, 4)
    signal, beta, fallback = calibrated_growth_signal(
        forecast,
        0.1,
        coverage_90=0.5,
        representation_collapsed=False,
        reference_mean=torch.zeros(4).numpy(),
    )
    assert fallback and beta == 0 and (signal == 0).all()
    empirical = empirical_block_scenarios(torch.zeros(100, 4).numpy(), horizon=21, count=20)
    assert empirical.shape == (20, 4)
