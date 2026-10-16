"""Strict custom window adapter for an explicitly admitted S6 development source.

This adapter does not acquire data or verify a provider's provenance assertions.
The required sidecar is an admission assertion plus a digest, checked before the
payload is opened. Numerical outcomes/features are converted only after every
partition's timestamp metadata passes the protected-boundary and causal checks.
No protected-tail evaluation is supported by this development-only adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

PROTECTED_LIMIT = 1735689600.0  # 2025-01-01T00:00:00Z
PARTITIONS = ("train", "calibration", "evaluation")


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite numeric UTC-seconds value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _required(mapping: dict[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"missing required field: {name}")
    return mapping[name]


def _numeric(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("booleans, strings and complex arrays are not numerical observations")
        result = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _strings(value: Any, name: str, *, unique: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _header(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != "s6-windows-v1":
        raise ValueError("payload schema_version must be s6-windows-v1")
    if payload.get("return_units") != "log_return":
        raise ValueError("return_units must be log_return")
    currency = _required(payload, "currency")
    if not isinstance(currency, str) or re.fullmatch(r"[A-Z]{3}", currency) is None:
        raise ValueError("currency must be an uppercase three-letter base-currency code")
    horizon = _number(_required(payload, "horizon_seconds"), "horizon_seconds")
    boundary = _number(_required(payload, "protected_start"), "protected_start")
    cutoff = _number(_required(payload, "fit_cutoff"), "fit_cutoff")
    if horizon <= 0:
        raise ValueError("horizon_seconds must be positive")
    if boundary > PROTECTED_LIMIT:
        raise ValueError("protected_start cannot move beyond 2025-01-01")
    if cutoff >= boundary:
        raise ValueError("fit_cutoff must precede protected_start")
    return {"schema_version": "s6-windows-v1", "return_units": "log_return",
            "currency": currency, "horizon_seconds": horizon, "protected_start": boundary,
            "fit_cutoff": cutoff,
            "asset_universe": _strings(_required(payload, "asset_universe"), "asset_universe", unique=True)}


def _partition_metadata(part: Any, name: str, header: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(part, dict):
        raise ValueError(f"{name} must be a partition object")
    assets = _strings(_required(part, "assets"), f"{name}.assets", unique=False)
    n = len(assets)
    vectors = {field: _numeric(_required(part, field), f"{name}.{field}", (n,))
               for field in ("origin_times", "maturity_times", "available_times")}
    origins, maturity, available = (vectors[field] for field in
                                     ("origin_times", "maturity_times", "available_times"))
    stamps = _numeric(_required(part, "window_times"), f"{name}.window_times")
    if stamps.ndim != 2 or stamps.shape[0] != n or stamps.shape[1] < 2:
        raise ValueError(f"{name}.window_times requires [row,time>=2]")
    releases = _numeric(_required(part, "window_available"), f"{name}.window_available", stamps.shape)
    timing = {**vectors, "window_times": stamps, "window_available": releases}
    if name == "train":
        timing["future_times"] = _numeric(_required(part, "future_times"), "train.future_times", stamps.shape)
        timing["future_available"] = _numeric(_required(part, "future_available"),
                                              "train.future_available", stamps.shape)
    # Evaluate every date guard before touching windows, targets or returns.
    if any(np.any(values >= header["protected_start"]) for values in timing.values()):
        raise ValueError(f"{name} timestamp reaches the protected boundary")
    if np.any(np.diff(stamps, axis=1) <= 0):
        raise ValueError(f"{name} window times must be strictly increasing")
    if np.any(releases < stamps) or np.any(stamps >= origins[:, None]) or np.any(releases >= origins[:, None]):
        raise ValueError(f"{name} windows must be observed and available strictly before origin")
    if not np.allclose(maturity - origins, header["horizon_seconds"], rtol=0, atol=1e-6):
        raise ValueError(f"{name} maturity must match the fixed horizon_seconds")
    if np.any(maturity <= origins) or np.any(available < maturity):
        raise ValueError(f"{name} requires origin < maturity <= availability")
    identities = list(zip(origins.tolist(), assets, strict=True))
    if len(set(identities)) != n:
        raise ValueError(f"{name} contains duplicate date/asset records")
    universe = set(header["asset_universe"])
    for date in np.unique(origins):
        if {asset for row, asset in enumerate(assets) if origins[row] == date} != universe:
            raise ValueError(f"{name} requires the complete declared asset universe at every date")
    if name == "train":
        future_times, future_releases = timing["future_times"], timing["future_available"]
        if (np.any(np.diff(future_times, axis=1) <= 0)
                or np.any(future_times <= origins[:, None])
                or np.any(future_releases < future_times)):
            raise ValueError("future windows require increasing post-origin times and valid availability")
        if (np.any(available >= header["fit_cutoff"])
                or np.any(future_releases >= header["fit_cutoff"])):
            raise ValueError("all training labels and future targets must be available before fit_cutoff")
    elif np.any(origins <= header["fit_cutoff"]):
        raise ValueError(f"{name} origins must follow fit_cutoff")
    return {**timing, "assets": assets}


def validate_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize all custom-window partitions, without model fitting.

    Explicit asset_universe must be identical and complete on every date. Data
    are already in one base currency; this function performs no FX conversion.
    Log-return outcomes cannot encode total loss. Such observations must use a
    separate default-aware data/model contract rather than inventing finite logs.
    """
    header = _header(payload)
    raw = _required(payload, "partitions")
    if not isinstance(raw, dict) or set(raw) != set(PARTITIONS):
        raise ValueError("partitions must contain exactly train, calibration and evaluation")
    # All partitions' metadata pass before any outcomes/features are converted.
    metadata = {name: _partition_metadata(raw[name], name, header) for name in PARTITIONS}
    cutoff = float(np.max(metadata["calibration"]["available_times"]))
    evaluation = metadata["evaluation"]
    if np.any(evaluation["origin_times"] <= cutoff):
        raise ValueError("evaluation must follow complete calibration availability")
    dates = np.unique(evaluation["origin_times"])
    if len(dates) > 1 and np.any(dates[:-1] + header["horizon_seconds"] > dates[1:] + 1e-6):
        raise ValueError("evaluation intervals must be synchronized and nonoverlapping")
    normalized = {}
    feature_shape: tuple[int, int] | None = None
    for name in PARTITIONS:
        part, meta = raw[name], metadata[name]
        n, times = meta["window_times"].shape
        windows = _numeric(_required(part, "windows"), f"{name}.windows")
        if windows.ndim != 3 or windows.shape[:2] != (n, times) or windows.shape[2] == 0:
            raise ValueError(f"{name}.windows requires [row,time,feature]")
        if feature_shape is None:
            feature_shape = windows.shape[1:]
        elif windows.shape[1:] != feature_shape:
            raise ValueError("all partitions must share time and feature dimensions")
        targets = _numeric(_required(part, "y"), f"{name}.y", (n,))
        returns = _numeric(_required(part, "simple_returns"), f"{name}.simple_returns", (n,))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            expected = np.expm1(targets)
        if (not np.isfinite(expected).all() or np.any(expected <= -1)
                or np.any(returns <= -1)
                or not np.allclose(returns, expected, rtol=1e-10, atol=1e-12)):
            raise ValueError(f"{name} simple_returns must equal expm1(y) and preserve positive gross value")
        result = {**meta, "windows": windows, "y": targets, "simple_returns": returns}
        if name == "train":
            result["future_windows"] = _numeric(_required(part, "future_windows"),
                                                "train.future_windows", windows.shape)
        normalized[name] = result
    header["last_outcome_time"] = max(float(np.max(part["maturity_times"])) for part in normalized.values())
    header["last_available_time"] = max(float(np.max(part["available_times"])) for part in normalized.values())
    header["calibration_cutoff"] = cutoff
    header["input_dim"] = feature_shape[1]
    header["window_length"] = feature_shape[0]
    header["admission_scope"] = "validated declared window contract; provenance assertions require external verification"
    return {"metadata": header, "partitions": normalized}


def _json(text: str) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"nonstandard JSON number: {value}")

    result = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError("JSON root must be an object")
    return result


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Read `<path>.manifest.json`, gate protected scope, hash then parse payload.

    Sidecar schema_version is s6-manifest-v1 and requires kind, sha256,
    protected_start, last_outcome_time, source, rights and point_in_time. A false
    provenance assertion is not made true by a passing structural validator.
    """
    path = Path(path)
    sidecar = Path(str(path) + ".manifest.json")
    manifest = _json(sidecar.read_text())
    if manifest.get("schema_version") != "s6-manifest-v1" or manifest.get("kind") != "admitted_development":
        raise ValueError("sidecar must be an s6-manifest-v1 admitted_development record")
    boundary = _number(_required(manifest, "protected_start"), "manifest.protected_start")
    last = _number(_required(manifest, "last_outcome_time"), "manifest.last_outcome_time")
    if boundary > PROTECTED_LIMIT or last >= boundary:
        raise ValueError("sidecar declares protected outcomes; payload will not be opened")
    for key in ("source", "rights", "point_in_time"):
        value = _required(manifest, key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"manifest.{key} must document provenance")
    digest = _required(manifest, "sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("manifest.sha256 must be 64 lowercase hexadecimal characters")
    raw = path.read_bytes()  # First payload access; every sidecar gate is above.
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("payload digest does not match admitted sidecar")
    result = validate_dataset(_json(raw.decode("utf-8")))
    meta = result["metadata"]
    if meta["protected_start"] != boundary or meta["last_outcome_time"] != last:
        raise ValueError("payload protected scope or last outcome disagrees with sidecar")
    meta["manifest"] = manifest
    meta["source_path"] = str(path.resolve())
    meta["sha256"] = digest
    return result
