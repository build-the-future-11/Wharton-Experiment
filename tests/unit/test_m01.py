import tempfile
from pathlib import Path

import numpy as np
import pytest

from wharton_lab.models.m01 import (
    M01Config,
    M01RegimeDistributionalGBM,
    causal_regime_ids,
    enforce_non_crossing,
    trailing_volatility,
)


def test_m01_quantile_shapes_and_finite():
    rng = np.random.default_rng(1)
    n, d = 80, 5
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + rng.normal(scale=0.1, size=n)
    past = rng.normal(scale=0.01, size=n)
    cfg = M01Config(n_estimators=20, max_depth=2, use_hist_gb=True)
    m = M01RegimeDistributionalGBM(cfg)
    m.fit(X, y, past_returns=past)
    q = m.predict_quantiles(X[:10], past_returns=past[:10])
    assert q.shape == (10, 5)
    assert np.all(np.isfinite(q))
    assert np.all(q[:, :-1] <= q[:, 1:] + 1e-9)


def test_m01_serialization_roundtrip():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    m = M01RegimeDistributionalGBM(M01Config(n_estimators=15, max_depth=2))
    m.fit(X, y)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m01.pkl"
        m.save(p)
        m2 = M01RegimeDistributionalGBM.load(p)
    pred1 = m.predict(X[:5])
    pred2 = m2.predict(X[:5])
    np.testing.assert_allclose(pred1, pred2)


def test_causal_regime_no_future_leak():
    r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    vol = trailing_volatility(r, window=2)
    assert np.isnan(vol[0])
    assert vol[1] == pytest.approx(0.0)
    regimes = causal_regime_ids(r, vol_window=2, vol_threshold=0.015)
    assert regimes.shape == r.shape


def test_non_crossing():
    q = np.array([[0.5, 0.2, 0.8, 0.3, 0.9]])
    fixed = enforce_non_crossing(q)
    assert np.all(fixed[0, :-1] <= fixed[0, 1:])
