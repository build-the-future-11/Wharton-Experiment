"""Independent matrix, semigroup, Monte Carlo, FIFO and gradient contracts."""

import os
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from frem_s14.cli import qualification_config
from frem_s14.dynamics import StochasticField, ring_laplacian
from frem_s14.model import S14, Engram, objective
from frem_s14.training import seed_all

DEVICE = os.environ.get("FREM_S14_TEST_DEVICE", "cpu")


def test_field_matches_independent_matrix_exponential():
    cfg = qualification_config(DEVICE).model
    field = StochasticField(cfg).to(DEVICE)
    z = torch.arange(cfg.latent, device=DEVICE).float()[None] / cfg.latent
    rows, cols = cfg.field_rows, cfg.latent // cfg.field_rows
    d = torch.nn.functional.softplus(field.raw_diffusion).detach().cpu().numpy() + 1e-5
    k = float(torch.nn.functional.softplus(field.raw_damping).detach()) + 1e-4
    matrix = d[0] * np.kron(ring_laplacian(rows).numpy(), np.eye(cols))
    matrix += d[1] * np.kron(np.eye(rows), ring_laplacian(cols).numpy()) + k * np.eye(cfg.latent)
    decay = expm(-matrix * 0.7)
    integral = np.linalg.solve(matrix, np.eye(cfg.latent) - decay)
    force = torch.tanh(field.forcing(z)).detach().cpu().numpy()
    actual, variance = field(z, 0.7)
    expected = z.cpu().numpy() @ decay + force @ integral
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-5, atol=2e-6)
    assert (variance > 0).all()


def test_semigroup_and_ou_variance_composition():
    field = StochasticField(qualification_config(DEVICE).model).to(DEVICE)
    a, _, q1 = field.coefficients(0.3)
    b, _, q2 = field.coefficients(0.7)
    c, _, q = field.coefficients(1.0)
    torch.testing.assert_close(a * b, c)
    torch.testing.assert_close(b.square() * q1 + q2, q)
    assert torch.all(c > 0) and torch.all(c < 1)


def test_stochastic_samples_match_conditional_moments():
    seed_all(91, DEVICE)
    cfg = qualification_config(DEVICE).model
    field = StochasticField(cfg).to(DEVICE)
    z = torch.zeros(1, cfg.latent, device=DEVICE)
    with torch.no_grad():
        mean, variance = field(z)
        draws = field.sample(z, count=12000)[:, 0]
        residual = (draws - mean) @ field.basis
    torch.testing.assert_close(residual.mean(0), torch.zeros_like(variance), atol=0.012, rtol=0)
    torch.testing.assert_close(residual.var(0), variance, atol=0.003, rtol=0.06)


def test_vectorized_fifo_exactly_matches_scalar_reference():
    cfg = qualification_config(DEVICE).model
    bank = Engram(cfg, 3).to(DEVICE)
    slots, cursor = [None] * cfg.memory_capacity, 0
    for count in (3, 80, 0, 7):
        values = torch.arange(count, device=DEVICE).float() + count * 100
        keys = torch.ones(count, 3, device=DEVICE)
        bank.append(
            keys, values, values.long(), torch.zeros(count, device=DEVICE, dtype=torch.int64)
        )
        for value in values.cpu().tolist():
            slots[cursor] = value
            cursor = (cursor + 1) % cfg.memory_capacity
        assert int(bank.cursor) == cursor
        for index, value in enumerate(slots):
            if value is not None:
                assert float(bank.values[index]) == value


def test_graph_cannot_use_future_or_other_asset_values():
    cfg = qualification_config(DEVICE).model
    bank = Engram(cfg, 3).to(DEVICE)
    keys = torch.ones(3, 3, device=DEVICE)
    bank.append(
        keys,
        torch.tensor([2.0, 100.0, -100.0], device=DEVICE),
        torch.tensor([1, 3, 1], device=DEVICE),
        torch.tensor([0, 0, 1], device=DEVICE),
    )
    result = bank(keys[:1], torch.tensor([2], device=DEVICE), torch.tensor([0], device=DEVICE))
    assert float(result[0, 0]) == pytest.approx(2)


def test_jepa_mask_field_gradients_and_growth_statistics():
    seed_all(11, DEVICE)
    cfg = qualification_config(DEVICE)
    model = S14(cfg.model).to(DEVICE).train()
    x = torch.randn(8, cfg.model.context, cfg.model.input_dim, device=DEVICE)
    output = model(
        x, torch.arange(8, device=DEVICE), torch.zeros(8, device=DEVICE, dtype=torch.int64)
    )
    target = torch.stack([x + 0.1, x + 0.2], 1)
    loss, _ = objective(model, output, torch.randn(8, device=DEVICE), target, cfg)
    loss.backward()
    for param in (model.field.raw_diffusion, model.field.raw_noise, model.mask_token):
        assert (
            param.grad is not None
            and torch.isfinite(param.grad).all()
            and param.grad.abs().sum() > 0
        )
    assert all(p.grad is None for p in model.target.parameters())
    model.encoder.observe_growth(output["growth_proposals"], torch.arange(8, device=DEVICE).float())
    assert int(model.encoder.growth_count) == 8
    model.encoder.eval()
    with pytest.raises(ValueError):
        model.encoder.observe_growth(output["growth_proposals"], torch.ones(8, device=DEVICE))


def test_all_new_ablation_laws_are_finite():
    cfg = qualification_config(DEVICE).model
    x = torch.zeros(4, cfg.context, cfg.input_dim, device=DEVICE)
    for arm in ("full", "no_spde", "no_multihorizon", "no_graph", "no_mask"):
        model = S14(replace(cfg, arm=arm)).to(DEVICE).eval()
        out = model(
            x, torch.arange(4, device=DEVICE), torch.zeros(4, device=DEVICE, dtype=torch.int64)
        )
        assert torch.isfinite(out["mean"]).all() and (out["std"] > 0).all()
