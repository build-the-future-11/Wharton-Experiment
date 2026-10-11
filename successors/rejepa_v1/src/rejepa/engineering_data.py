"""Deterministic synthetic data for end-to-end engineering verification only."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from .data import write_deterministic_snapshot

ENGINEERING_LABEL = "SYNTHETIC_ENGINEERING_FIXTURE_NOT_MARKET_EVIDENCE"


@dataclass(frozen=True)
class EngineeringDataset:
    panel_csv: str
    panel_manifest: str
    training_npz: str
    metadata_json: str
    samples: int
    assets: int
    decision_dates: int
    input_dim: int
    context_length: int
    horizon: int
    label: str = ENGINEERING_LABEL


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def generate_engineering_dataset(
    destination: str | Path,
    *,
    seed: int = 11,
    asset_count: int = 6,
    decision_dates: int = 96,
    context_length: int = 16,
    horizon: int = 5,
) -> EngineeringDataset:
    """Create a causal regime-shift fixture; never an empirical market substitute."""
    if asset_count < 3 or decision_dates < 40 or context_length < 4 or horizon < 1:
        raise ValueError("Engineering data requires >=3 assets, >=40 dates, context >=4 and horizon >=1")
    if context_length + 2 * horizon + 8 >= decision_dates:
        raise ValueError("Not enough dates for purged chronological training and validation")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total_dates = decision_dates + horizon
    dates = pd.bdate_range("2024-01-02", periods=total_dates, tz="UTC") + timedelta(hours=21)

    regime = np.zeros(total_dates, dtype=int)
    factor = np.zeros(total_dates)
    for index in range(1, total_dates):
        switch_probability = 0.04 if regime[index - 1] == 0 else 0.09
        regime[index] = 1 - regime[index - 1] if rng.random() < switch_probability else regime[index - 1]
        innovation_scale = 0.006 if regime[index] == 0 else 0.018
        factor[index] = 0.18 * factor[index - 1] + rng.normal(0.0, innovation_scale)
    loadings = np.linspace(0.65, 1.25, asset_count)
    phase = np.linspace(0.0, np.pi, asset_count)
    idiosyncratic = rng.normal(0.0, 0.004, size=(total_dates, asset_count))
    seasonal = 0.0015 * np.sin(np.arange(total_dates)[:, None] / 6.0 + phase[None, :])
    log_returns = factor[:, None] * loadings[None, :] + seasonal + idiosyncratic
    log_returns = np.clip(log_returns, -0.18, 0.18)
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    volume = np.exp(
        12.0
        + 0.25 * regime[:, None]
        + np.linspace(-0.2, 0.2, asset_count)[None, :]
        + rng.normal(0.0, 0.12, size=(total_dates, asset_count))
    )
    currencies = np.asarray((["USD", "EUR", "JPY"] * ((asset_count + 2) // 3))[:asset_count])
    markets = np.asarray((["US", "EU", "JP"] * ((asset_count + 2) // 3))[:asset_count])
    fx = np.ones((total_dates, asset_count))
    for asset in range(asset_count):
        if currencies[asset] != "USD":
            base = 1.08 if currencies[asset] == "EUR" else 0.0068
            fx[:, asset] = base * np.exp(np.cumsum(rng.normal(0.0, 0.0015, total_dates)))

    rows: list[dict[str, object]] = []
    for date_index in range(decision_dates):
        event_time = dates[date_index]
        available = event_time + timedelta(minutes=5)
        outcome_available = dates[date_index + horizon] + timedelta(minutes=5)
        for asset in range(asset_count):
            local_log_return = log_returns[date_index + 1 : date_index + horizon + 1, asset].sum()
            fx_log_return = np.log(fx[date_index + horizon, asset] / fx[date_index, asset])
            rows.append(
                {
                    "asset_id": f"SYNTH-{asset:02d}",
                    "event_time": event_time.isoformat(),
                    "availability_time": available.isoformat(),
                    "outcome_available_time": outcome_available.isoformat(),
                    "market": markets[asset],
                    "currency": currencies[asset],
                    "eligible": True,
                    "point_in_time_membership": True,
                    "close": float(close[date_index, asset]),
                    "total_return": float(np.expm1(local_log_return + fx_log_return)),
                    "volume": float(volume[date_index, asset]),
                    "source_id": "rejepa-causal-regime-generator-v0.3",
                    "fx_to_usd": float(fx[date_index, asset]),
                    "delisted": False,
                    "corporate_action_id": "synthetic-none",
                }
            )
    panel = pd.DataFrame(rows)
    panel_path = target / "panel.csv"
    panel_manifest = write_deterministic_snapshot(
        panel,
        panel_path,
        snapshot_id=f"rejepa-engineering-seed-{seed}",
        source_name="RE-JEPA deterministic causal regime generator v0.3",
        license_or_right="Locally generated engineering fixture; scientific and market claims prohibited",
        retrieved_at="2026-10-11T00:00:00Z",
    )

    market_return = log_returns.mean(axis=1)
    rolling_vol = np.zeros_like(log_returns)
    for index in range(total_dates):
        start = max(0, index - context_length + 1)
        rolling_vol[index] = log_returns[start : index + 1].std(axis=0, ddof=0)
    features = np.stack(
        [
            log_returns,
            np.broadcast_to(market_return[:, None], log_returns.shape),
            rolling_vol,
            np.log1p(volume) / 20.0,
        ],
        axis=-1,
    )
    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    realized: list[float] = []
    decision_time_ns: list[int] = []
    outcome_time_ns: list[int] = []
    horizons: list[int] = []
    for date_index in range(context_length - 1, decision_dates - horizon):
        for asset in range(asset_count):
            contexts.append(features[date_index - context_length + 1 : date_index + 1, asset])
            futures.append(features[date_index + 1 : date_index + horizon + 1, asset])
            realized.append(float(log_returns[date_index + 1 : date_index + horizon + 1, asset].sum()))
            decision_time_ns.append(int(dates[date_index].value))
            outcome_time_ns.append(int((dates[date_index + horizon] + timedelta(minutes=5)).value))
            horizons.append(horizon)
    context_array = np.asarray(contexts, dtype=np.float32)
    future_array = np.asarray(futures, dtype=np.float32)
    realized_array = np.asarray(realized, dtype=np.float32)
    training_path = target / "training.npz"
    with NamedTemporaryFile("wb", dir=target, delete=False) as handle:
        np.savez_compressed(
            handle,
            context=context_array,
            future_target=future_array,
            realized_return=realized_array,
            decision_time_ns=np.asarray(decision_time_ns, dtype=np.int64),
            outcome_available_ns=np.asarray(outcome_time_ns, dtype=np.int64),
            horizon=np.asarray(horizons, dtype=np.int64),
            features=context_array.reshape(len(context_array), -1),
            target=realized_array,
        )
        temporary = Path(handle.name)
    os.replace(temporary, training_path)
    metadata_path = target / "ENGINEERING_DATASET.json"
    dataset = EngineeringDataset(
        panel_csv=str(panel_path),
        panel_manifest=str(panel_path.with_suffix(".csv.manifest.json")),
        training_npz=str(training_path),
        metadata_json=str(metadata_path),
        samples=len(context_array),
        assets=asset_count,
        decision_dates=decision_dates,
        input_dim=context_array.shape[-1],
        context_length=context_length,
        horizon=horizon,
    )
    _atomic_json(
        metadata_path,
        {
            **asdict(dataset),
            "seed": seed,
            "panel_sha256": panel_manifest.sha256,
            "training_sha256": _sha256(training_path),
            "scientific_use": "PROHIBITED",
            "market_data": False,
            "protected_data": False,
            "generator_contract": (
                "Two-state stochastic volatility factor, heterogeneous loadings, deterministic seasonality, "
                "idiosyncratic noise and synthetic FX; all training windows are chronological."
            ),
        },
    )
    return dataset
