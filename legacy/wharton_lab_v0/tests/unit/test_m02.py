import tempfile
from pathlib import Path

import numpy as np

from wharton_lab.models.m02 import M02Config, M02FactorResidualRanker, date_level_spearman_ic
from wharton_lab.models.m02.factors import residualize_labels, rolling_ols_exposure


def test_m02_shapes_finite_serialization():
    rng = np.random.default_rng(3)
    n, d = 60, 4
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    dates = np.repeat(np.arange(12), 5)
    m = M02FactorResidualRanker(M02Config(rank_epochs=30))
    m.fit(X, y, date_ids=dates)
    s = m.predict(X[:8])
    assert s.shape == (8,)
    assert np.all(np.isfinite(s))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m02.pkl"
        m.save(p)
        m2 = M02FactorResidualRanker.load(p)
    np.testing.assert_allclose(m.predict(X[:8]), m2.predict(X[:8]))


def test_residual_labels_and_rolling_ols():
    T, N, F = 50, 3, 2
    rng = np.random.default_rng(0)
    f = rng.normal(size=(T, F))
    r = rng.normal(size=(T, N))
    betas = rolling_ols_exposure(r, f, window=20)
    assert betas.shape == (T, N, F)
    assert np.isnan(betas[:20]).all()
    y = rng.normal(size=N)
    b = betas[-1, :, :]
    rff = f[-1]
    resid = residualize_labels(y, b, np.tile(rff, (N, 1)))
    assert resid.shape == (N,)


def test_spearman_ic():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([1.1, 2.2, 2.9, 4.5])
    dates = np.array([0, 0, 0, 0])
    ic = date_level_spearman_ic(scores, labels, dates)
    assert ic > 0.9
