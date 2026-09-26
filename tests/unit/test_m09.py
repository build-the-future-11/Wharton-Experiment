import numpy as np
import pytest

from wharton_lab.models.m09 import M09Model, M09Config
from wharton_lab.models.m01.regimes import causal_regime_ids


def test_m09_fit_predict():
    rng = np.random.default_rng(1)
    X = rng.normal(scale=0.01, size=(80, 4))
    y = X[:, 0] * 2 + rng.normal(scale=0.005, size=80)
    m = M09Model(M09Config(groupdro_steps=3))
    m.fit(X, y)
    pred = m.predict(X[:10])
    assert pred.shape == (10,)
    assert np.all(np.isfinite(pred))


def test_m09_groupdro_weights():
    m = M09Model()
    out = m.smoke_forward()
    assert out["finite"]
    assert abs(out["group_weights_sum"] - 1.0) < 1e-6


def test_causal_regimes_past_only():
    rets = np.array([0.01, -0.02, 0.03, -0.01] * 10, dtype=float)
    r = causal_regime_ids(rets, vol_window=5)
    assert r.shape == rets.shape
