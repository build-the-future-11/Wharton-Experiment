import tempfile
from pathlib import Path

import numpy as np

from wharton_lab.models.m03 import M03Config, M03ConditionalNonlinearFactor


def test_m03_fit_predict_serialization():
    rng = np.random.default_rng(4)
    N, d, T = 25, 6, 40
    X = rng.normal(size=(N, d))
    panel = rng.normal(size=(T, N))
    y = panel[-1]
    m = M03ConditionalNonlinearFactor(M03Config(mlp_epochs=40, n_factors=2))
    m.fit(X, y, returns_panel=panel)
    pred = m.predict(X)
    assert pred.shape == (N,)
    assert np.all(np.isfinite(pred))
    f = m.forecast_factors(steps=2)
    assert f.shape == (2, 2)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m03.pkl"
        m.save(p)
        m2 = M03ConditionalNonlinearFactor.load(p)
    np.testing.assert_allclose(pred, m2.predict(X))
