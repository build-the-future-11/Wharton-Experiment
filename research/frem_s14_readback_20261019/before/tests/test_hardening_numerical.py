"""Regressions for causal clocks, AMP memory, moments and raw evidence readback."""

import os
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from frem_s14.baselines import combine
from frem_s14.cli import qualification_config
from frem_s14.data import Samples, generated_panel
from frem_s14.dynamics import StochasticField
from frem_s14.evaluation import allocate, score_law
from frem_s14.model import S14, Engram, objective
from frem_s14.training import seed_all
from frem_s14.verification import verify_calendar, verify_suite

DEVICE = os.environ.get("FREM_S14_TEST_DEVICE", "cpu")


def test_delayed_feature_rejects_already_realized_target():
    panel = generated_panel()
    panel.available += 86400000000
    with pytest.raises(ValueError, match="target return"):
        Samples(panel, qualification_config(DEVICE))


@pytest.mark.parametrize("horizon", [1, 2, 5, 20])
def test_vectorized_compounding_matches_scalar_with_terminal_loss(horizon):
    cfg = qualification_config(DEVICE)
    cfg = replace(cfg, model=replace(cfg.model, horizon=horizon), embargo=horizon)
    panel = generated_panel(length=240)
    panel.returns[30, 1] = panel.features[30, 1, 0] = -1
    samples = Samples(panel, cfg)
    expected = np.array(
        [
            np.prod(1 + panel.returns[o + 1 : o + horizon + 1, a]) - 1
            for o, a in zip(samples.origin, samples.asset, strict=True)
        ]
    )
    np.testing.assert_allclose(samples.y, expected, rtol=0, atol=1e-14)
    assert np.any(samples.y == -1)


def test_hybrid_variance_survives_large_common_location():
    def law(mu):
        return {
            "mu": np.array([[mu]]),
            "scale": np.array([[0.01]]),
            "weights": np.ones((1, 3, 1)),
            "mean": np.array([mu]),
            "std": np.array([0.01 * (5 / 3) ** 0.5]),
        }

    result = combine(law(1e8), law(1e8 + 2), 0.25, np.array([1e8]), 5)
    expected = (0.01**2 * 5 / 3 + 0.25 * 0.75 * 4) ** 0.5
    np.testing.assert_allclose(result["std"], [expected], rtol=1e-12)
    with pytest.raises(ValueError):
        combine(law(0), law(1), float("nan"), np.zeros(1), 5)


def test_neural_variance_survives_large_common_location():
    cfg = qualification_config(DEVICE).model
    model = S14(cfg).to(DEVICE).eval()
    with torch.no_grad():
        for expert in model.expert:
            expert[-1].weight.zero_()
            expert[-1].bias.zero_()
            expert[-1].bias[0] = 1e4
        out = model(
            torch.zeros(4, cfg.context, cfg.input_dim, device=DEVICE),
            torch.arange(4, device=DEVICE),
            torch.zeros(4, dtype=torch.int64, device=DEVICE),
        )
    within = (torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-3) * (5 / 3) ** 0.5
    torch.testing.assert_close(
        out["std"], torch.full_like(out["std"], within.item()), atol=1e-5, rtol=1e-5
    )


def test_ineligible_extreme_values_cannot_contaminate_retrieval():
    bank = Engram(qualification_config(DEVICE).model, 3).to(DEVICE)
    keys = torch.ones(3, 3, device=DEVICE)
    bank.append(
        keys,
        torch.tensor([2.0, 1e30, -1e30], device=DEVICE),
        torch.tensor([1, 3, 1], device=DEVICE),
        torch.tensor([0, 0, 1], device=DEVICE),
    )
    result = bank(keys[:1], torch.tensor([2], device=DEVICE), torch.tensor([0], device=DEVICE))
    assert torch.isfinite(result).all() and result[0, 0] == 2
    empty = bank(keys[:1], torch.tensor([1], device=DEVICE), torch.tensor([0], device=DEVICE))
    torch.testing.assert_close(empty, torch.zeros_like(empty), rtol=0, atol=0)


def test_populated_graph_bank_remains_float32_under_autocast():
    bank = Engram(qualification_config(DEVICE).model, 3).to(DEVICE)
    keys = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], device=DEVICE)
    bank.append(
        keys,
        torch.tensor([2.0, 3.0, 9.0], device=DEVICE),
        torch.tensor([1, 1, 3], device=DEVICE),
        torch.zeros(3, dtype=torch.int64, device=DEVICE),
    )
    times, assets = torch.tensor([2], device=DEVICE), torch.tensor([0], device=DEVICE)
    reference = bank(keys[:1], times, assets)
    with torch.autocast(
        device_type=DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16
    ):
        result = bank(keys[:1], times, assets)
    assert result.dtype == torch.float32 and torch.isfinite(result).all()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_field_density_matches_independent_full_gaussian():
    seed_all(19, DEVICE)
    cfg = qualification_config(DEVICE).model
    field = StochasticField(cfg).to(DEVICE)
    z, target = torch.randn(3, cfg.latent, device=DEVICE), torch.randn(3, cfg.latent, device=DEVICE)
    with torch.no_grad():
        mean, variance = field(z, 0.6)
        actual = field.latent_nll(target, mean, variance)
    basis = field.basis.cpu().numpy().astype(np.float64)
    covariance = (basis * variance.cpu().numpy()) @ basis.T
    expected = np.mean(
        [
            -multivariate_normal.logpdf(y, m, covariance) / cfg.latent
            for y, m in zip(target.cpu().numpy(), mean.cpu().numpy(), strict=True)
        ]
    )
    assert float(actual) == pytest.approx(expected, rel=3e-6, abs=3e-6)


def test_optional_metrics_do_not_change_loss_or_gradients():
    seed_all(11, DEVICE)
    cfg = qualification_config(DEVICE)
    model = S14(cfg.model).to(DEVICE)
    x = torch.randn(4, cfg.model.context, cfg.model.input_dim, device=DEVICE)
    out = model(x, torch.arange(4, device=DEVICE), torch.zeros(4, dtype=torch.int64, device=DEVICE))
    future = torch.stack([x] * len(cfg.model.target_offsets), 1)
    y = torch.zeros(4, device=DEVICE)
    a, metrics = objective(model, out, y, future, cfg)
    b, no_metrics = objective(model, out, y, future, cfg, collect_metrics=False)
    assert metrics and no_metrics == {}
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    p = model.router.weight
    ga = torch.autograd.grad(a, p, retain_graph=True)[0]
    gb = torch.autograd.grad(b, p)[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def test_retained_calendar_rejects_reordered_missing_or_wrong_asset():
    raw = {
        "y": np.zeros(4),
        "times": np.array([10, 10, 20, 20], dtype=np.int64),
        "ends": np.array([15, 15, 25, 25], dtype=np.int64),
        "assets": np.array([0, 1, 0, 1], dtype=np.int64),
    }
    split = {"rows": 4, "first_decision": 10, "last_decision": 20, "last_label": 25}
    assert verify_calendar(raw, split) == 2
    for field, value in [
        ("assets", [1, 0, 0, 1]),
        ("assets", [0, 0, 0, 1]),
        ("times", [10, 20, 10, 20]),
        ("ends", [15, 16, 25, 25]),
    ]:
        changed = dict(raw, **{field: np.array(value, dtype=np.int64)})
        with pytest.raises(ValueError):
            verify_calendar(changed, split)


def test_all_routing_gates_are_validated():
    raw = {"mu": np.zeros((2, 1)), "scale": np.ones((2, 1)), "weights": np.ones((2, 3, 1))}
    raw["weights"][0, 2, 0] = np.nan
    with pytest.raises(ValueError):
        score_law(raw, np.zeros(2), 5)


def test_allocation_randomized_differential_reference():
    rng = np.random.default_rng(731)
    for _ in range(8):
        matrix = rng.normal(0, 0.1, (4, 4))
        covariance = matrix @ matrix.T + np.eye(4) * 0.03
        mean = rng.normal(0.02, 0.05, 4)
        previous = rng.dirichlet(np.ones(5))[:4]
        weights, _ = allocate(mean, covariance, previous, 0.4, 3, 0.002)

        def loss(w, mean=mean, covariance=covariance, previous=previous):
            return -mean @ w + 1.5 * w @ covariance @ w + 0.002 * abs(w - previous).sum()

        reference = minimize(
            loss,
            np.full(4, 0.2),
            method="SLSQP",
            bounds=[(0, 0.4)] * 4,
            constraints={"type": "ineq", "fun": lambda w: 1 - w.sum()},
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        assert reference.success
        assert loss(weights) <= loss(reference.x) + 2e-7


def test_suite_verifier_rejects_false_completion_before_cell_readback(tmp_path):
    from frem_s14.cli import source_identity
    from frem_s14.runtime import manifest, write_json
    from frem_s14.suite import FOLD_ENDS, suite_config

    cfg = suite_config(DEVICE)
    write_json(
        tmp_path / "plan.json",
        {
            "config": cfg.to_dict(),
            "fold_ends": list(FOLD_ENDS),
            "expected_cells": 171,
            "protected_access": False,
            "source_identity": source_identity(),
        },
    )
    write_json(tmp_path / "status.json", {"status": "COMPLETE", "folds": []})
    write_json(tmp_path / "cells.json", [])
    write_json(tmp_path / "manifest.json", manifest(tmp_path))
    with pytest.raises(ValueError, match="suite plan"):
        verify_suite(tmp_path)
