"""M07 Eigen-JEPA unit tests."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wharton_lab.models.m07.config import M07Config
from wharton_lab.models.m07.metrics import (
    projector_sign_invariant_distance,
    flag_small_eigengaps,
)
from wharton_lab.models.m07.model import EigenJEPAModel
from wharton_lab.models.m07.psd import PSDCovHead


def test_psd_cholesky_head():
    head = PSDCovHead(4)
    cov = head()
    sym = torch.allclose(cov, cov.T)
    evals = torch.linalg.eigvalsh(cov)
    psd = bool(torch.all(evals > 0))
    assert sym and psd


def test_projector_sign_flip_invariance():
    rng = np.random.default_rng(0)
    U = rng.normal(size=(5, 3))
    V = U.copy()
    V[:, 1] *= -1
    d = projector_sign_invariant_distance(U, V)
    assert d < 1e-10


def test_predict_covariance_and_eigengaps():
    cfg = M07Config(epochs=2, n_assets=4)
    m = EigenJEPAModel(cfg)
    w20 = 20 * cfg.n_assets
    X = np.random.default_rng(0).normal(size=(12, w20))
    y = np.zeros(12)
    m.fit(X, y)
    cov = m.predict_covariance(20)
    evals = np.linalg.eigvalsh(cov)
    assert np.all(evals > -1e-8)
    flags = flag_small_eigengaps(evals, cfg.eigengap_threshold)
    assert flags.dtype == bool
    pred = m.predict(X)
    assert pred.size == 6  # 3 evals x 2 windows
