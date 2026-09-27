"""Cost sensitivity + forecast×controller factorial (research proxies)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wharton_lab.baselines.ewma_cov import ewma_covariance
from wharton_lab.baselines.persistence import persistence_forecast
from wharton_lab.baselines.ridge import ridge_forecast
from wharton_lab.data.synthetic import SyntheticWorldConfig, WorldKind, generate_world
from wharton_lab.models.m11 import M11Config, M11Model
from wharton_lab.orchestration.lockbox import LOCKBOX_SEED


BPS = (0, 5, 10, 25, 50)
FORECASTS = ("persistence", "ridge")
CONTROLLERS = ("myopic_equal", "mpc_cvar")


def _turnover_cost(w_prev: np.ndarray, w_new: np.ndarray, bps: float) -> float:
    """One-way turnover × bps."""
    return float(np.sum(np.abs(w_new - w_prev))) * (bps * 1e-4)


def run_factorial(repo_root: Path | str, *, seed: int = LOCKBOX_SEED) -> dict:
    root = Path(repo_root)
    out_dir = root / "runs" / "factorial"
    out_dir.mkdir(parents=True, exist_ok=True)

    world = generate_world(
        SyntheticWorldConfig(kind=WorldKind.SIGNAL, n_steps=160, seed=seed, n_assets=5)
    )
    rets = world.returns
    X, y = world.features, world.labels
    cut = int(0.7 * len(y))
    X_tr, y_tr, X_te, y_te = X[:cut], y[:cut], X[cut:], y[cut:]
    path = rets[cut : cut + 40, :3]  # evaluation path (T, k)
    T, k = path.shape
    liab = np.linspace(0.01, 0.03, 4)

    cells = []
    for fc_name in FORECASTS:
        if fc_name == "ridge":
            pred = ridge_forecast(X_tr, y_tr, X_te)
            fc_mse = float(np.mean((y_te - pred) ** 2))
        else:
            pred = persistence_forecast(y_tr, len(y_te))
            fc_mse = float(np.mean((y_te - pred) ** 2))

        for ctrl in CONTROLLERS:
            for bps in BPS:
                wealth = 1.0
                w = np.full(k, 1.0 / k)
                shortfalls = []
                for t in range(T):
                    if ctrl == "myopic_equal":
                        w_new = np.full(k, 1.0 / k)
                    else:
                        # MPC on rolling mini-scenarios around path
                        rng = np.random.default_rng(seed + t)
                        scen = rng.normal(scale=0.01, size=(6, 4, k)) + path[t]
                        m = M11Model(M11Config(horizon=4))
                        m.fit(scen, np.zeros(6), liability_schedule=liab, cash=wealth)
                        w_new = np.asarray(m.predict(scen), dtype=float).ravel()[:k]
                        if w_new.size < k:
                            w_new = np.full(k, 1.0 / k)
                        w_new = np.clip(w_new, 0, None)
                        s = w_new.sum()
                        w_new = w_new / s if s > 1e-12 else np.full(k, 1.0 / k)

                    cost = _turnover_cost(w, w_new, float(bps))
                    wealth = wealth * (1.0 + float(w_new @ path[t])) - cost
                    # operating obligation each step (proxy)
                    ob = float(liab[min(t, len(liab) - 1)]) * 0.25
                    shortfalls.append(max(0.0, ob - max(wealth, 0.0) * 0.02))
                    w = w_new

                cell = {
                    "forecast": fc_name,
                    "controller": ctrl,
                    "cost_bps_one_way": bps,
                    "forecast_mse": fc_mse,
                    "terminal_wealth": wealth,
                    "shortfall_mean": float(np.mean(shortfalls)),
                    "shortfall_cvar_80": float(np.mean(sorted(shortfalls)[int(0.8 * len(shortfalls)) :])),
                }
                cells.append(cell)

    # EWMA sanity on same panel
    cov = ewma_covariance(rets[:cut], lam=0.94)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "turnover_definition": "one_way",
        "cost_scenarios_bps": list(BPS),
        "base_cost_bps": 10,
        "note": "Research proxies only — not client funding guarantees",
        "status": "INVALID_AS_EVIDENCE (D-054)",
        "defects": [
            "forecast predictions are computed but never passed to either controller",
            "mpc_cvar scenarios at step t are centred on path[t], the return it then earns (look-ahead)",
            "evaluated on the spent lockbox seed after the lockbox was opened",
            "single 40-step path, no uncertainty; shortfall never binds",
        ],
        "ewma_frobenius_vs_sample": float(
            np.linalg.norm(cov - np.cov(rets[:cut].T), ord="fro")
        ),
        "cells": cells,
    }
    (out_dir / "FACTORIAL_RESULTS.json").write_text(json.dumps(payload, indent=2))

    # Markdown table at base cost
    lines = [
        "# Forecast × controller factorial (synthetic lockbox seed)",
        "",
        "**Status: INVALID AS EVIDENCE (protocol/DECISIONS.md D-054).** Forecasts are never "
        "passed to the controllers (both forecast rows are identical); `mpc_cvar` scenarios "
        "at step t are centred on the return it then earns (look-ahead); single 40-step path "
        "on the spent lockbox seed. Retained for provenance only.",
        "",
        "Costs are **one-way** turnover × research bps. Base = **10 bps**.",
        "",
        "| forecast | controller | bps | terminal_wealth | shortfall_mean |",
        "|---|---|---:|---:|---:|",
    ]
    for c in cells:
        if c["cost_bps_one_way"] == 10:
            lines.append(
                f"| {c['forecast']} | {c['controller']} | {c['cost_bps_one_way']} | "
                f"{c['terminal_wealth']:.4f} | {c['shortfall_mean']:.6f} |"
            )
    lines += ["", "## Cost sensitivity (ridge × mpc_cvar)", ""]
    lines += ["| bps | terminal_wealth | shortfall_mean |", "|---:|---:|---:|"]
    for c in cells:
        if c["forecast"] == "ridge" and c["controller"] == "mpc_cvar":
            lines.append(
                f"| {c['cost_bps_one_way']} | {c['terminal_wealth']:.4f} | {c['shortfall_mean']:.6f} |"
            )
    (out_dir / "FACTORIAL_SUMMARY.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    run_factorial(root)
    print(f"Wrote {root / 'runs' / 'factorial'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
