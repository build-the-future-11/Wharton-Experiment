import numpy as np
import pytest

from wharton_lab.exceptions import BlockedClientError
from wharton_lab.models.m11 import M11Model, M11Config


def _synthetic_liabilities(horizon: int = 5):
    return np.linspace(0.02, 0.06, horizon)


def test_m11_requires_liabilities():
    m = M11Model()
    X = np.random.default_rng(0).normal(size=(10, 5, 3))
    with pytest.raises(BlockedClientError):
        m.fit(X, np.zeros(10))


def test_m11_mpc_smoke():
    rng = np.random.default_rng(0)
    H, k = 5, 3
    X = rng.normal(scale=0.01, size=(16, H, k))
    m = M11Model(M11Config(horizon=H))
    m.fit(X, np.zeros(16), liability_schedule=_synthetic_liabilities(H), cash=1.0)
    w = m.predict(X)
    assert w.shape == (k,)
    assert np.all(np.isfinite(w))
    assert 0.99 <= w.sum() <= 1.01
