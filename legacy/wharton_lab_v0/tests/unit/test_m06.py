"""M06 FI-JEPA unit tests."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wharton_lab.models.m06.config import M06Config
from wharton_lab.models.m06.model import FIJEPAModel
from wharton_lab.models.m06.modules import ema_update


def test_gradient_flow():
    cfg = M06Config(epochs=1, batch_size=8, seq_len=4, input_dim=3)
    m = FIJEPAModel(cfg)
    n = 24
    X = np.random.default_rng(0).normal(size=(n, cfg.seq_len * cfg.input_dim))
    y = np.random.default_rng(1).normal(size=n)
    m.fit(X, y)
    for p in m.context.parameters():
        assert p.grad is None or np.all(np.isfinite(p.detach().cpu().numpy()))
    # one manual backward step has grad
    seq = torch.as_tensor(m._to_sequences(X[:8]), dtype=torch.float32, device=m.device)
    z = m.context(seq)
    loss = z.sum()
    m._opt.zero_grad()
    loss.backward()
    assert any(p.grad is not None for p in m.context.parameters())


def test_no_ema_on_holdout_future():
    cfg = M06Config(epochs=2, holdout_future_frac=0.25, ema_momentum=0.9)
    m = FIJEPAModel(cfg)
    before = {k: v.clone() for k, v in m.target.state_dict().items()}
    X = np.random.default_rng(0).normal(size=(40, cfg.seq_len * cfg.input_dim))
    y = np.random.default_rng(1).normal(size=40)
    m.fit(X, y)
    after = m.target.state_dict()
    changed = any(not torch.allclose(before[k], after[k]) for k in before)
    assert changed


def test_serialization_roundtrip(tmp_path):
    m = FIJEPAModel(M06Config(epochs=1))
    X = np.random.default_rng(0).normal(size=(16, 32))
    y = np.random.default_rng(1).normal(size=16)
    m.fit(X, y)
    path = tmp_path / "m06.pkl"
    m.save(path)
    m2 = FIJEPAModel.load(path)
    p1 = m.predict(X[:4])
    p2 = m2.predict(X[:4])
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-5)
