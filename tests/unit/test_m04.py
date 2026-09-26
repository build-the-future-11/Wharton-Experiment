from datetime import datetime, timedelta

import numpy as np
import tempfile
from pathlib import Path

from wharton_lab.models.m04 import M04Config, M04FinancialAPEN, PendingOutcomeQueue


def test_m04_pending_maturity():
    q = PendingOutcomeQueue(horizon_steps=2, maturity_delay=1)
    t0 = datetime(2024, 1, 1)
    key = np.array([1.0, 0.0])
    q.enqueue(t0, key, 0.5)
    assert q.pop_matured(t0 + timedelta(days=2)) == []
    matured = q.pop_matured(t0 + timedelta(days=3))
    assert len(matured) == 1
    assert matured[0].residual == 0.5


def test_m04_fit_predict_serialization():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(50, 4))
    y = X @ np.array([0.5, -0.2, 0.1, 0.3]) + rng.normal(scale=0.05, size=50)
    cfg = M04Config(horizon_steps=1, maturity_delay=1, memory_capacity=32)
    m = M04FinancialAPEN(cfg)
    m.fit(X, y)
    t = datetime(2024, 1, 1)
    m.log_prediction(X, y, t, row_index=0)
    assert m._pending.pending_count() == 1
    n = m.advance_time(t + timedelta(days=3))
    assert n == 1
    pred = m.predict(X[:6], as_of=datetime(2024, 6, 1))
    assert pred.shape == (6,)
    assert np.all(np.isfinite(pred))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m04.pkl"
        m.save(p)
        m2 = M04FinancialAPEN.load(p)
    np.testing.assert_allclose(pred, m2.predict(X[:6], as_of=datetime(2024, 6, 1)))
