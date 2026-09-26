"""Build train/test tensors for orchestration runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wharton_lab.data.etf_track import load_etf_track
from wharton_lab.data.synthetic import SyntheticWorldConfig, WorldKind, generate_world


@dataclass
class TabularSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    extras: dict


def _train_test(features: np.ndarray, labels: np.ndarray, fold: int, *, train_frac: float = 0.7):
    n = len(labels)
    test_size = max(8, int(n * (1 - train_frac)))
    split = n - test_size - fold * 2
    split = max(int(n * train_frac), split)
    split = min(split, n - test_size)
    return features[:split], labels[:split], features[split:], labels[split:]


# Process-level ETF cache — avoid repeated disk/Yahoo hits within a run.
_ETF_MEMO: dict[tuple, object] = {}


def load_tabular(dataset: str, seed: int, fold: int, *, n_steps: int = 80) -> TabularSplit:
    ds = dataset.lower()
    if ds in ("synthetic_track_c", "synthetic_track_c_signal", "track_c"):
        world = generate_world(
            SyntheticWorldConfig(kind=WorldKind.SIGNAL, n_steps=n_steps, seed=seed, n_assets=5)
        )
        X, y = world.features, world.labels
        X_tr, y_tr, X_te, y_te = _train_test(X, y, fold)
        return TabularSplit(
            X_tr,
            y_tr,
            X_te,
            y_te,
            {
                "past_returns_train": world.returns[: len(y_tr), 0],
                "past_returns_test": world.returns[len(y_tr) : len(y_tr) + len(y_te), 0],
                "returns_panel": world.returns,
                "source": "synthetic_track_c",
            },
        )
    if ds in ("etf_track_a", "track_a", "etf"):
        key = (seed, n_steps)
        etf = _ETF_MEMO.get(key)
        if etf is None:
            etf = load_etf_track(seed=seed)
            _ETF_MEMO[key] = etf
        rets = etf.returns
        n = min(n_steps, rets.shape[0])
        rets = rets[:n]
        r0 = rets[:, 0]
        # Vectorized rolling std (expanding) — O(n) vs Python loop
        csum = np.cumsum(r0)
        csum2 = np.cumsum(r0**2)
        idx = np.arange(1, n + 1, dtype=float)
        var = np.maximum(csum2 / idx - (csum / idx) ** 2, 0.0)
        feat = np.column_stack([np.concatenate([[0.0], r0[:-1]]), np.sqrt(var)])
        labels = r0
        X_tr, y_tr, X_te, y_te = _train_test(feat, labels, fold)
        return TabularSplit(
            X_tr,
            y_tr,
            X_te,
            y_te,
            {
                "past_returns_train": labels[: len(y_tr)],
                "past_returns_test": labels[len(y_tr) : len(y_tr) + len(y_te)],
                "returns_panel": rets,
                "source": etf.source,
                "synthetic_proxy": etf.synthetic_proxy,
            },
        )
    raise ValueError(f"Unknown dataset {dataset}")


def panel_rank_features(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-sectional rows: each (date, asset) is one row."""
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError(f"returns_panel must be 2D, got {returns.shape}")
    # Drop non-finite columns/rows that would break cross-section concat later.
    col_ok = np.all(np.isfinite(returns), axis=0)
    returns = returns[:, col_ok]
    if returns.shape[1] < 2:
        raise ValueError("Need at least 2 finite assets for ranking panel")
    row_ok = np.all(np.isfinite(returns), axis=1)
    returns = returns[row_ok]
    t, k = returns.shape
    rows_x = []
    rows_y = []
    date_ids = []
    for i in range(1, t):
        for j in range(k):
            rows_x.append([returns[i - 1, j], returns[i - 1].mean(), returns[i - 1].std()])
            rows_y.append(returns[i, j])
            date_ids.append(i)
    return (
        np.asarray(rows_x, dtype=float),
        np.asarray(rows_y, dtype=float),
        np.asarray(date_ids, dtype=int),
    )


def sequence_matrix(features: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    n, d = features.shape
    if n <= seq_len:
        seq_len = max(2, n // 2)
    rows = []
    targets = []
    for i in range(seq_len, n):
        rows.append(features[i - seq_len : i].reshape(-1))
        targets.append(features[i, 0])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def covariance_windows(returns: np.ndarray, window: int, n_assets: int) -> tuple[np.ndarray, np.ndarray]:
    t, k = returns.shape
    k = min(k, n_assets)
    rows = []
    targets = []
    for i in range(window, t):
        w = returns[i - window : i, :k]
        rows.append(w.reshape(-1))
        targets.append(np.linalg.eigh(np.cov(w.T))[0][-1])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)
