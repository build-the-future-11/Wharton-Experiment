"""M08 Graph FI-JEPA unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from wharton_lab.models.m08.config import M08Config
from wharton_lab.models.m08.graph import shuffle_edges_control, shrinkage_correlation
from wharton_lab.models.m08.model import GraphFIJEPAModel


def test_shrinkage_correlation_psd():
    r = np.random.default_rng(0).normal(size=(30, 5))
    corr = shrinkage_correlation(r, shrinkage=0.3)
    assert corr.shape == (5, 5)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)


def test_frozen_graph_and_mask():
    cfg = M08Config(n_nodes=4, seq_len=6, node_dim=2, epochs=2)
    m = GraphFIJEPAModel(cfg)
    flat = cfg.n_nodes * cfg.seq_len * cfg.node_dim
    X = np.random.default_rng(0).normal(size=(10, flat))
    y = np.random.default_rng(1).normal(size=(10, cfg.n_nodes))
    m.fit(X, y)
    A = m.frozen_adjacency()
    p1 = m.predict(X[:2])
    # mask one node
    mask = np.ones((2, cfg.n_nodes))
    mask[:, 0] = 0.0
    p2 = m.predict(X[:2], node_mask=mask)
    assert p1.shape == (2, cfg.n_nodes)
    assert np.allclose(p2[:, 0], 0.0)
    assert not np.allclose(A, np.eye(cfg.n_nodes))


def test_shuffled_edge_control():
    cfg = M08Config(epochs=1)
    m = GraphFIJEPAModel(cfg)
    flat = cfg.n_nodes * cfg.seq_len * cfg.node_dim
    X = np.random.default_rng(0).normal(size=(8, flat))
    y = np.random.default_rng(1).normal(size=(8, cfg.n_nodes))
    m.fit(X, y)
    base = m.predict(X[:2])
    shuf = m.predict(X[:2], shuffled_edges=True)
    assert base.shape == shuf.shape
