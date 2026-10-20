"""Mathematical, causality, gradient, serialization and accounting tests."""

import os
import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.stats import t

from frem_s14.cli import qualification_config
from frem_s14.data import Samples, generated_panel, load_csv
from frem_s14.evaluation import (
    allocate,
    capped_simplex,
    holm,
    paired_bootstrap,
    rebalance,
    score_law,
)
from frem_s14.model import S14, nll, objective
from frem_s14.runtime import digest, write_json
from frem_s14.training import (
    GRUForecast,
    load_checkpoint,
    populate,
    predict,
    save_checkpoint,
    seed_all,
    train_neural,
)

DEVICE = os.environ.get("FREM_S14_TEST_DEVICE", "cpu")


@pytest.fixture
def cfg():
    return qualification_config(DEVICE)


def test_purged_splits_and_scaler_no_future(cfg):
    panel = generated_panel()
    a = Samples(panel, cfg)
    assert a.end[a.train].max() < a.time[a.validation].min()
    assert a.end[a.validation].max() < a.time[a.test].min()
    changed = generated_panel()
    start = a.origin[a.validation].min()
    changed.features[start:, :, 1:] += 100
    b = Samples(changed, cfg)
    np.testing.assert_array_equal(a.center, b.center)
    np.testing.assert_array_equal(a.scale, b.scale)
    np.testing.assert_array_equal(a.batch(a.train)[0], b.batch(b.train)[0])


def test_compounded_labels_and_lazy_windows(cfg):
    samples = Samples(generated_panel(), cfg)
    ix = samples.train[:1]
    o, a = samples.origin[ix[0]], samples.asset[ix[0]]
    expected = np.prod(1 + samples.panel.returns[o + 1 : o + cfg.model.horizon + 1, a]) - 1
    assert samples.y[ix[0]] == expected
    x, future = samples.batch(ix, future=True)
    np.testing.assert_array_equal(x[0, -1], samples.x[o, a])
    np.testing.assert_array_equal(future[0, -1, -1], samples.x[o + cfg.model.horizon, a])


def test_csv_protected_values_never_parsed(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "timestamp,asset,available_at,return,feature_x\n"
        "2000-01-01T00:00:00Z,A,2000-01-01T00:00:00Z,0.01,1\n"
        "2000-01-02T00:00:00Z,A,2000-01-02T00:00:00Z,0.02,2\n"
        "2030-01-01T00:00:00Z,A,invalid,DO_NOT_PARSE,DO_NOT_PARSE\n"
    )
    meta = {
        "schema": 1,
        "sha256": digest(path),
        "source": "generated test",
        "retrieved_at": "2026-10-19T00:00:00Z",
        "license": "generated",
        "universe_policy": "test fixture",
        "purpose": "development",
        "return_definition": "simple_total_return_including_terminal_outcomes",
        "feature_availability": "available_at_bounds_all_features",
    }
    write_json(str(path) + ".manifest.json", meta)
    assert load_csv(path, "2025-01-01T00:00:00Z").returns.shape == (2, 1)


def test_reject_nonfinite_and_invalid_returns():
    panel = generated_panel()
    panel.features[0, 0, 1] = np.nan
    with pytest.raises(ValueError):
        panel.validate()
    panel = generated_panel()
    panel.returns[0, 0] = -1.01
    with pytest.raises(ValueError):
        panel.validate()


def test_reservoir_contraction_and_growth(cfg):
    model = S14(cfg.model).to(DEVICE)
    for matrix in model.encoder.recurrent:
        assert float(torch.linalg.matrix_norm(matrix, 2)) <= cfg.model.contraction + 1e-6
    before = model.encoder.recurrent.clone()
    assert model.encoder.grow() == (4, 6)
    assert model.encoder.grow() == (6, 8)
    assert model.encoder.grow() == (8, 8)
    torch.testing.assert_close(before, model.encoder.recurrent, rtol=0, atol=0)
    assert all(1 - leak + leak * cfg.model.contraction < 1 for leak in cfg.model.leaks)


def test_memory_strict_maturity_asset_and_empty(cfg):
    model = S14(cfg.model).to(DEVICE)
    width = model.memory.keys.shape[1]
    key = torch.ones(1, width, device=DEVICE)
    model.memory.append(
        key,
        torch.tensor([3.0], device=DEVICE),
        torch.tensor([100], device=DEVICE),
        torch.tensor([0], device=DEVICE),
    )
    same = torch.tensor([0], device=DEVICE)
    assert model.memory(key, torch.tensor([100], device=DEVICE), same)[0, 0] == 0
    assert model.memory(key, torch.tensor([101], device=DEVICE), same)[0, 0] == 3
    assert model.memory(key, torch.tensor([101], device=DEVICE), same + 1)[0, 0] == 0
    model.memory.reset()
    assert torch.equal(model.memory(key, same, same), torch.zeros(1, 3, device=DEVICE))


def test_memory_overflow_and_checkpoint_equivalence(cfg, tmp_path):
    model = S14(cfg.model).to(DEVICE)
    n, width = cfg.model.memory_capacity + 5, model.memory.keys.shape[1]
    model.memory.append(
        torch.ones(n, width, device=DEVICE),
        torch.arange(n, device=DEVICE).float(),
        torch.arange(n, device=DEVICE),
        torch.zeros(n, dtype=torch.int64, device=DEVICE),
    )
    assert int(model.memory.count) == cfg.model.memory_capacity
    save_checkpoint(tmp_path / "model.pt", {"binding": {"x": 1}, "model": model.state_dict()})
    restored = S14(cfg.model).to(DEVICE)
    restored.load_state_dict(load_checkpoint(tmp_path / "model.pt", {"x": 1})["model"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "model.pt", {"x": 2})


def test_router_caps_gradient_and_target_stop_gradient(cfg):
    seed_all(11, DEVICE)
    model = S14(cfg.model).to(DEVICE)
    x = torch.randn(4, cfg.model.context, cfg.model.input_dim, device=DEVICE)
    times, assets = torch.arange(4, device=DEVICE), torch.zeros(4, dtype=torch.int64, device=DEVICE)
    out = model(x, times, assets)
    torch.testing.assert_close(out["weights"].sum(-1), torch.ones(4, 3, device=DEVICE))
    assert out["weights"].max() <= cfg.model.router_cap + 1e-6
    loss, _ = objective(
        model,
        out,
        torch.randn(4, device=DEVICE),
        torch.stack([x + 0.1] * len(cfg.model.target_offsets), 1),
        cfg,
    )
    loss.backward()
    assert all(p.grad is None for p in model.target.parameters())
    assert model.router.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    target_before = next(model.target.parameters()).clone()
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1)
    model.ema_update()
    torch.testing.assert_close(next(model.target.parameters()), target_before + (1 - cfg.model.ema))


def test_scipy_mixture_matches_torch_and_single_law():
    raw = {
        "mu": np.array([[0.1, -0.2]]),
        "scale": np.array([[0.3, 0.4]]),
        "weights": np.array([[[0.7, 0.3]] * 3]),
    }
    y = np.array([0.5])
    out = {key: torch.tensor(value) for key, value in raw.items()}
    np.testing.assert_allclose(
        nll(out, torch.tensor(y), 5).numpy(), score_law(raw, y, 5)["nll"], rtol=1e-12
    )
    single = {"mu": np.zeros((1, 1)), "scale": np.ones((1, 1)), "weights": np.ones((1, 3, 1))}
    assert score_law(single, np.zeros(1), 5)["nll"][0] == pytest.approx(-t.logpdf(0, 5))


def test_projection_constraints_and_optimizer_reference():
    p = capped_simplex([9, -5, 3, 2], 0.4)
    assert p.sum() <= 1 + 1e-12 and p.max() <= 0.4 and p.min() >= 0
    mean = np.array([0.02, 0.03, -0.01])
    cov = np.array([[0.04, 0.01, 0], [0.01, 0.06, 0], [0, 0, 0.03]])
    previous = np.array([0.2, 0.3, 0.1])
    weights, _ = allocate(mean, cov, previous, 0.6, 3, 0.001)

    def loss(w):
        return -mean @ w + 1.5 * w @ cov @ w + 0.001 * np.abs(w - previous).sum()

    ref = minimize(
        loss,
        previous,
        method="SLSQP",
        bounds=[(0, 0.6)] * 3,
        constraints={"type": "ineq", "fun": lambda w: 1 - w.sum()},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert ref.success
    assert loss(weights) <= loss(ref.x) + 1e-7


def test_self_financing_fees_and_terminal_loss():
    factor, after, cost = rebalance([0], [1], [0.1], 0.01)
    assert factor == pytest.approx(1.1 / 1.01)
    assert cost == pytest.approx(0.01 / 1.01)
    np.testing.assert_allclose(after, [1])
    assert rebalance([1], [0], [0], 0.01)[0] == pytest.approx(0.99)
    assert rebalance([1], [1], [-1], 0)[0] == 0
    with pytest.raises(ValueError):
        rebalance([0], [1.2], [0], 0)


def test_bootstrap_and_holm():
    result = paired_bootstrap(np.ones(20) * 0.05, 4, 99)
    np.testing.assert_allclose(result["ci95"], [0.05, 0.05])
    assert result["p_two_sided"] == 0.01
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(5), 4, 99)


def test_training_and_epoch_resume_are_equal(cfg, tmp_path):
    cfg = replace(cfg, arms=("full",), seeds=(11,))
    samples = Samples(generated_panel(), cfg)
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    binding = {"test": "generated"}
    full = train_neural(samples, cfg, "full", 11, one, time.monotonic() + 600, binding)
    resumed = train_neural(
        samples, cfg, "full", 11, two, time.monotonic() + 600, binding, one / "epoch_000.pt"
    )
    a, b = [predict(m, samples, samples.test, DEVICE) for m in (full, resumed)]
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    before = full.memory.values.clone()
    populate(full, samples, np.array([], dtype=np.int64), DEVICE)
    torch.testing.assert_close(full.memory.values, before)


def test_gru_predict_uses_shared_non_trace_interface(cfg):
    """Regression for Kaggle Version 3's S14-only keyword leak."""
    samples = Samples(generated_panel(), cfg)
    model = GRUForecast(cfg.model).to(DEVICE)
    result = predict(model, samples, samples.validation[:3], DEVICE)
    assert set(result) == {"mu", "scale", "weights", "mean", "std", "nll"}
    assert all(np.isfinite(value).all() for value in result.values())


@pytest.mark.skipif(DEVICE != "cuda", reason="requires explicitly selected CUDA runtime")
def test_cuda_amp_is_finite(cfg):
    cfg = replace(cfg, amp=True)
    model = S14(cfg.model).cuda()
    x = torch.randn(4, cfg.model.context, cfg.model.input_dim, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(
            x, torch.arange(4, device="cuda"), torch.zeros(4, device="cuda", dtype=torch.int64)
        )
        loss, _ = objective(
            model,
            out,
            torch.zeros(4, device="cuda"),
            torch.stack([x] * len(cfg.model.target_offsets), 1),
            cfg,
        )
    loss.backward()
    assert torch.isfinite(loss)
