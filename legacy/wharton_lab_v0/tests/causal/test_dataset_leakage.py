"""Leakage and fold-diversity audit of legacy vs repaired data paths (D-050..D-052)."""

import numpy as np
import pytest

from wharton_lab.audit.leaky import LeakyFeatureError, assert_features_causal, assert_no_feature_leak
from wharton_lab.backtesting.walk_forward import iter_walk_forward
from wharton_lab.data.causal_tracks import causal_return_features, causal_synthetic_panel
from wharton_lab.data.etf_track import ETFTrackResult
from wharton_lab.orchestration import datasets


def _panel(n=300, k=5, seed=3):
    return np.random.default_rng(seed).normal(scale=0.01, size=(n, k))


def _legacy_etf_build(monkeypatch):
    def build(rets):
        datasets._ETF_MEMO.clear()
        monkeypatch.setattr(
            datasets,
            "load_etf_track",
            lambda seed: ETFTrackResult(["A"] * rets.shape[1], rets, "test", False),
        )
        s = datasets.load_tabular("etf_track_a", seed=11, fold=0, n_steps=rets.shape[0])
        X = np.vstack([s.X_train, s.X_test])
        return X, np.arange(len(X))

    return build


def test_legacy_synthetic_track_leaks_contemporaneous_latent():
    s = datasets.load_tabular("synthetic_track_c", seed=11, fold=0, n_steps=160)
    X = np.vstack([s.X_train, s.X_test])
    y = np.concatenate([s.y_train, s.y_test])
    with pytest.raises(LeakyFeatureError):
        assert_no_feature_leak(X[:, 0], y)


def test_legacy_etf_track_expanding_std_includes_label(monkeypatch):
    with pytest.raises(LeakyFeatureError):
        assert_features_causal(_legacy_etf_build(monkeypatch), _panel())


def test_legacy_folds_are_identical_at_lean_full_size():
    tests = [
        datasets.load_tabular("synthetic_track_c", seed=11, fold=f, n_steps=160).X_test
        for f in range(4)
    ]
    for t in tests[1:]:
        np.testing.assert_array_equal(t, tests[0])


def test_repaired_features_are_causal():
    def build(rets):
        p = causal_return_features(rets)
        return p.X, p.row_t

    assert_features_causal(build, _panel())


def test_repaired_synthetic_has_no_contemporaneous_leak():
    p = causal_synthetic_panel(seed=11, n_steps=400)
    for j in range(p.X.shape[1]):
        assert_no_feature_leak(p.X[:, j], p.y)


def test_walk_forward_test_windows_are_disjoint():
    folds = list(iter_walk_forward(1000, n_folds=6, min_train=300, test_size=100))
    assert len(folds) == 6
    seen: set[int] = set()
    for f in folds:
        idx = set(f.test_idx.tolist())
        assert not idx & seen
        assert f.train_idx.max() < f.test_idx.min()
        seen |= idx
