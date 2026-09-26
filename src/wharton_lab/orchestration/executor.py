"""Execute a single manifest row."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from wharton_lab.baselines.ewma_cov import ewma_covariance
from wharton_lab.baselines.persistence import persistence_forecast
from wharton_lab.baselines.ridge import ridge_forecast
from wharton_lab.evaluation.metrics import spearman_ic
from wharton_lab.models.m01.model import average_pinball_loss
from wharton_lab.orchestration.datasets import (
    TabularSplit,
    covariance_windows,
    load_tabular,
    panel_rank_features,
    sequence_matrix,
)
from wharton_lab.orchestration.manifest import ManifestRow
from wharton_lab.orchestration.model_factory import build_model
from wharton_lab.orchestration.profiles import ExperimentProfile


def _config_hash(model_id: str, variant: str, seed: int) -> str:
    payload = json.dumps({"model_id": model_id, "variant": variant, "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _mse(y: np.ndarray, pred: np.ndarray) -> float:
    err = np.asarray(y, dtype=float).ravel() - np.asarray(pred, dtype=float).ravel()
    return float(np.mean(err**2))


def _research_liabilities(horizon: int) -> np.ndarray:
    return np.linspace(0.02, 0.05, horizon)


def run_baseline(model_id: str, split: TabularSplit) -> dict[str, Any]:
    mid = model_id.upper()
    if mid == "B_RIDGE":
        pred = ridge_forecast(split.X_train, split.y_train, split.X_test)
        metric = _mse(split.y_test, pred)
        return {"primary_value": metric, "metric_name": "mse", "predictions": pred.tolist()[:5]}
    if mid == "B_PERSISTENCE":
        pred = persistence_forecast(split.y_train, len(split.y_test))
        metric = _mse(split.y_test, pred)
        return {"primary_value": metric, "metric_name": "mse", "predictions": pred.tolist()[:5]}
    if mid == "B_EWMA_COV":
        rets = split.extras.get("returns_panel")
        if rets is None:
            raise ValueError("EWMA baseline requires returns_panel")
        w = min(20, rets.shape[0] // 2)
        emp = np.cov(rets[-w:].T)
        pred_cov = ewma_covariance(rets, lam=0.94)
        err = float(np.linalg.norm(pred_cov - emp, ord="fro"))
        return {"primary_value": err, "metric_name": "frobenius_cov_error"}
    raise ValueError(f"Unknown baseline {model_id}")


def run_model(
    model_id: str,
    variant: str,
    split: TabularSplit,
    *,
    smoke: bool,
    horizon: int,
) -> dict[str, Any]:
    mid = model_id.upper()
    model = build_model(mid, variant, smoke=smoke)

    if mid == "M01":
        past_tr = split.extras["past_returns_train"]
        past_te = split.extras["past_returns_test"]
        model.fit(split.X_train, split.y_train, past_returns=past_tr)
        q = model.predict_quantiles(split.X_test, past_returns=past_te)
        med = q[:, q.shape[1] // 2]
        metric = average_pinball_loss(split.y_test, q, model.quantiles_)
        ridge = ridge_forecast(split.X_train, split.y_train, split.X_test)
        ridge_mse = _mse(split.y_test, ridge)
        return {
            "primary_value": metric,
            "metric_name": "pinball_loss",
            "baseline_ridge_mse": ridge_mse,
            "baseline_comparison": "lower_pinball_vs_ridge_mse",
        }

    if mid == "M02":
        rets = split.extras["returns_panel"]
        X, y, date_ids = panel_rank_features(rets)
        n = len(y)
        cut = int(n * 0.75)
        # Prefer factory config; smoke stays light
        if smoke:
            model.config.rank_epochs = min(model.config.rank_epochs, 4)
        model.fit(X[:cut], y[:cut], date_ids=date_ids[:cut])
        scores = model.predict(X[cut:])
        ic = spearman_ic(y[cut:], scores)
        return {"primary_value": float(ic), "metric_name": "spearman_ic"}

    if mid == "M03":
        rets = split.extras["returns_panel"]
        model.fit(split.X_train, split.y_train, returns_panel=rets[: len(split.y_train) + 1])
        pred = model.predict(split.X_test)
        return {"primary_value": _mse(split.y_test, pred), "metric_name": "mse"}

    if mid in ("M04", "M05", "M09"):
        model.fit(split.X_train, split.y_train)
        pred = model.predict(split.X_test)
        return {"primary_value": _mse(split.y_test, pred), "metric_name": "mse"}

    if mid == "M06":
        feat = np.vstack([split.X_train, split.X_test])
        seq_len = model.config.seq_len
        Xs, ys = sequence_matrix(feat, seq_len)
        cut = max(8, int(len(ys) * 0.7))
        model.fit(Xs[:cut], ys[:cut])
        pred = model.predict(Xs[cut:])
        return {"primary_value": _mse(ys[cut:], pred), "metric_name": "latent_mse"}

    if mid == "M07":
        rets = split.extras["returns_panel"]
        Xw, yw = covariance_windows(rets, window=20, n_assets=model.config.n_assets)
        cut = max(5, int(len(yw) * 0.7))
        model.fit(Xw[:cut], yw[:cut])
        preds = []
        for i in range(cut, len(yw)):
            p = model.predict(Xw[i : i + 1])
            preds.append(float(np.asarray(p).ravel()[-1]))
        return {"primary_value": _mse(yw[cut:], np.asarray(preds)), "metric_name": "cov_forecast_mse"}

    if mid == "M08":
        rets = split.extras["returns_panel"]
        t, k = rets.shape
        n_nodes = model.config.n_nodes
        sl = model.config.seq_len
        fd = model.config.node_dim
        active = min(k, n_nodes)
        rows = []
        ys = []
        masks = []
        for i in range(sl, t):
            window = rets[i - sl : i, :active]
            if window.shape[1] < n_nodes:
                pad = np.zeros((sl, n_nodes - window.shape[1]))
                window = np.concatenate([window, pad], axis=1)
            tensor = np.zeros((n_nodes, sl, fd))
            tensor[:, :, 0] = window.T
            rows.append(tensor.reshape(-1))
            y_row = np.zeros(n_nodes, dtype=float)
            y_row[:active] = rets[i, :active]
            ys.append(y_row)
            mask = np.zeros(n_nodes, dtype=float)
            mask[:active] = 1.0
            masks.append(mask)
        Xg = np.asarray(rows)
        yg = np.asarray(ys)
        mg = np.asarray(masks)
        cut = max(5, int(len(yg) * 0.7))
        model.fit(Xg[:cut], yg[:cut], node_mask=mg[:cut])
        pred = model.predict(Xg[cut:], node_mask=mg[cut:])
        # Score only active (non-padded) nodes
        w = mg[cut:]
        err = (yg[cut:] - pred) ** 2
        denom = float(w.sum()) if float(w.sum()) > 0 else 1.0
        return {
            "primary_value": float((err * w).sum() / denom),
            "metric_name": "graph_latent_mse",
        }

    if mid == "M10":
        model.fit(split.X_train, split.y_train)
        pred = model.predict(split.X_test)
        return {"primary_value": _mse(split.y_test, pred), "metric_name": "energy_score_proxy_mse"}

    if mid == "M11":
        rng = np.random.default_rng(split.y_train.shape[0])
        H, k = (3, 3) if smoke else (4, 3)
        n_scen = 4 if smoke else 6
        scen = rng.normal(scale=0.01, size=(n_scen, H, k))
        liab = _research_liabilities(H)
        model.fit(scen, np.zeros(n_scen), liability_schedule=liab, cash=1.0)
        w = model.predict(scen)
        shortfall = float(np.maximum(0.0, liab - w.sum()).mean())
        return {
            "primary_value": shortfall,
            "metric_name": "funding_shortfall",
            "research_proxy_liabilities": True,
        }

    if mid == "M12":
        rets = split.extras["returns_panel"]
        n = min(len(rets), 40)
        X = rets[:n].reshape(n, -1)
        y = rets[:n, 0]
        model.fit(X, y)
        pred = model.predict_market(X[-5:])
        H, k = 5, model.config.n_assets
        scen = rng_normal_scenarios(8, H, k, seed=0)
        liab = _research_liabilities(H)
        weights = model.plan(scen, liab, cash=1.0)
        return {
            "primary_value": _mse(y[-5:], pred.ravel()[:5]),
            "metric_name": "market_mse",
            "allocation_sum": float(np.sum(weights)),
        }

    raise ValueError(f"No runner for {model_id}")


def rng_normal_scenarios(n: int, H: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.01, size=(n, H, k))


def execute_row(
    row: ManifestRow,
    *,
    repo_root: Path,
    profile: ExperimentProfile,
    n_steps: int,
) -> tuple[dict[str, Any], Path]:
    smoke = profile.name.value == "SMOKE"
    t0 = time.perf_counter()
    split = load_tabular(row.dataset, row.seed, row.fold, n_steps=n_steps)
    if row.model_id.upper().startswith("B_"):
        metrics = run_baseline(row.model_id, split)
    else:
        metrics = run_model(
            row.model_id,
            row.variant,
            split,
            smoke=smoke,
            horizon=row.horizon,
        )
    elapsed = time.perf_counter() - t0
    receipt = {
        "run_id": row.run_id,
        "model_id": row.model_id,
        "variant": row.variant,
        "dataset": row.dataset,
        "fold": row.fold,
        "seed": row.seed,
        "horizon": row.horizon,
        "profile": row.profile,
        "config_hash": row.config_hash or _config_hash(row.model_id, row.variant, row.seed),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 3),
        "metrics": metrics,
        "data_provenance": {
            "source": split.extras.get("source"),
            "synthetic_proxy": split.extras.get("synthetic_proxy"),
        },
    }
    out_dir = repo_root / "runs" / row.profile.lower() / row.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt, receipt_path
