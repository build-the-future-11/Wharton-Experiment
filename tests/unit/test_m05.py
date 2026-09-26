"""M05 Q-APEN unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from wharton_lab.models.m05.config import M05Config
from wharton_lab.models.m05.model import QAPENModel
from wharton_lab.models.m05.quantized_memory import QuantizedExpertMemory


def test_quantized_memory_nbytes_and_roundtrip():
    mem = QuantizedExpertMemory(cap_bytes=1_000_000, codebook_size=256)
    w = np.linspace(-1, 1, 40)
    assert mem.pack("a", w)
    used = mem.used_bytes()
    assert used > 0
    w2 = mem.unpack("a")
    assert w2 is not None
    assert w2.shape == w.shape
    assert np.mean(np.abs(w2 - w)) < 0.05


def test_top_k_routing_sums_to_one():
    cfg = M05Config(max_experts=4, top_k=2, min_matured_before_lifecycle=1)
    m = QAPENModel(cfg)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 5))
    y = rng.normal(size=16)
    m.fit(X, y)
    weights = m._route(X)
    assert weights.shape == (16, len(m._specs))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    active = (weights > 0).sum(axis=1)
    assert np.all(active <= cfg.top_k)


def test_pending_maturity_before_lifecycle():
    cfg = M05Config(min_matured_before_lifecycle=10, create_loss_threshold=0.0)
    m = QAPENModel(cfg)
    X = np.random.default_rng(1).normal(size=(8, 3))
    y = np.random.default_rng(2).normal(size=8)
    m.fit(X, y)
    assert m.pending_maturity_count() >= 1
    n_after_first = len(m._specs)
    m.fit(X, y)
    assert len(m._specs) == n_after_first


def test_save_load_smoke(tmp_path):
    m = QAPENModel(M05Config(random_state=0))
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.random.default_rng(1).normal(size=20)
    m.fit(X, y)
    p = tmp_path / "m05.pkl"
    m.save(p)
    m2 = QAPENModel.load(p)
    pred = m2.predict(X[:4])
    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))
