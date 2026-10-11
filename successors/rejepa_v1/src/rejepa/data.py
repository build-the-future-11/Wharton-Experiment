"""Point-in-time financial data validation and deterministic snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "asset_id",
    "event_time",
    "availability_time",
    "outcome_available_time",
    "market",
    "currency",
    "eligible",
    "point_in_time_membership",
    "close",
    "total_return",
    "volume",
    "source_id",
)

REQUIRED_FEATURE_COLUMNS = (
    "entity_id",
    "feature_name",
    "observation_time",
    "availability_time",
    "value",
    "source_id",
    "vintage_id",
    "frequency",
)


@dataclass(frozen=True)
class DataManifest:
    snapshot_id: str
    source_name: str
    license_or_right: str
    retrieved_at: str
    row_count: int
    columns: tuple[str, ...]
    minimum_event_time: str
    maximum_event_time: str
    sha256: str
    base_currency: str = "USD"


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    row_count: int
    asset_count: int


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame.copy()
    for column in ("event_time", "availability_time", "outcome_available_time"):
        canonical[column] = pd.to_datetime(canonical[column], utc=True, errors="coerce")
    return canonical.sort_values(["event_time", "asset_id"], kind="mergesort").reset_index(drop=True)


def validate_point_in_time_frame(frame: pd.DataFrame) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        return ValidationReport(False, (f"missing columns: {', '.join(missing)}",), (), len(frame), 0)
    canonical = _canonicalize(frame)
    times = canonical[["event_time", "availability_time", "outcome_available_time"]]
    if times.isna().any().any():
        errors.append("timestamps must parse as timezone-aware instants")
    if (canonical["availability_time"] < canonical["event_time"]).any():
        errors.append("availability_time precedes event_time")
    if (canonical["outcome_available_time"] <= canonical["event_time"]).any():
        errors.append("outcome_available_time must follow event_time")
    if canonical.duplicated(["asset_id", "event_time"]).any():
        errors.append("duplicate asset_id/event_time rows")
    numeric = canonical[["close", "total_return", "volume"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        errors.append("price, return and volume fields must be finite")
    if (numeric["close"] <= 0).any() or (numeric["volume"] < 0).any():
        errors.append("close must be positive and volume nonnegative")
    if (numeric["total_return"] <= -1.0).any():
        errors.append("simple total_return cannot be at or below -100 percent")
    for column in ("eligible", "point_in_time_membership"):
        values = canonical[column]
        if values.isna().any() or not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            errors.append(f"{column} must contain explicit booleans")
    for column in ("asset_id", "market", "currency", "source_id"):
        if canonical[column].isna().any() or (canonical[column].astype(str).str.len() == 0).any():
            errors.append(f"{column} cannot be blank")
    if (
        canonical["point_in_time_membership"]
        .map(lambda value: isinstance(value, (bool, np.bool_)) and not bool(value))
        .any()
    ):
        warnings.append("snapshot contains rows outside point-in-time membership")
    if "delisted" not in canonical.columns:
        warnings.append("delisted indicator absent; survivorship coverage is unverified")
    if "corporate_action_id" not in canonical.columns:
        warnings.append("corporate-action lineage absent")
    non_base = canonical["currency"] != "USD"
    if non_base.any():
        if "fx_to_usd" not in canonical.columns:
            errors.append("non-USD rows require fx_to_usd")
        else:
            fx = pd.to_numeric(canonical.loc[non_base, "fx_to_usd"], errors="coerce")
            if not np.isfinite(fx.to_numpy(dtype=float)).all() or (fx <= 0).any():
                errors.append("non-USD fx_to_usd must be finite and positive")
    return ValidationReport(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        row_count=len(canonical),
        asset_count=int(canonical["asset_id"].nunique()),
    )


def validate_point_in_time_features(frame: pd.DataFrame) -> ValidationReport:
    """Validate fundamentals, curves, inflation, macro and classification vintages."""
    missing = sorted(set(REQUIRED_FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        return ValidationReport(False, (f"missing columns: {', '.join(missing)}",), (), len(frame), 0)
    canonical = frame.copy()
    for column in ("observation_time", "availability_time"):
        canonical[column] = pd.to_datetime(canonical[column], utc=True, errors="coerce")
    errors: list[str] = []
    warnings: list[str] = []
    if canonical[["observation_time", "availability_time"]].isna().any().any():
        errors.append("feature timestamps must parse as timezone-aware instants")
    if (canonical["availability_time"] < canonical["observation_time"]).any():
        errors.append("feature availability precedes observation")
    if canonical.duplicated(["entity_id", "feature_name", "observation_time", "vintage_id"]).any():
        errors.append("duplicate feature vintage")
    values = pd.to_numeric(canonical["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        errors.append("feature values must be finite")
    allowed = {"daily", "weekly", "monthly", "quarterly", "annual", "event"}
    unknown = sorted(set(canonical["frequency"].astype(str)) - allowed)
    if unknown:
        errors.append("unsupported feature frequencies: " + ", ".join(unknown))
    if "revision_of_vintage_id" not in canonical.columns:
        warnings.append("revision lineage absent; vintage audit remains manual")
    return ValidationReport(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        row_count=len(canonical),
        asset_count=int(canonical["entity_id"].nunique()),
    )


def features_available_as_of(frame: pd.DataFrame, decision_time: str | pd.Timestamp) -> pd.DataFrame:
    report = validate_point_in_time_features(frame)
    if not report.valid:
        raise ValueError("Invalid feature frame: " + "; ".join(report.errors))
    decision = pd.Timestamp(decision_time)
    decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
    canonical = frame.copy()
    canonical["observation_time"] = pd.to_datetime(canonical["observation_time"], utc=True)
    canonical["availability_time"] = pd.to_datetime(canonical["availability_time"], utc=True)
    eligible = canonical.loc[canonical["availability_time"] <= decision].copy()
    return eligible.sort_values(
        ["availability_time", "entity_id", "feature_name", "vintage_id"], kind="mergesort"
    ).reset_index(drop=True)


def information_available_as_of(frame: pd.DataFrame, decision_time: str | pd.Timestamp) -> pd.DataFrame:
    report = validate_point_in_time_frame(frame)
    if not report.valid:
        raise ValueError("Invalid point-in-time frame: " + "; ".join(report.errors))
    decision = pd.Timestamp(decision_time)
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    else:
        decision = decision.tz_convert("UTC")
    canonical = _canonicalize(frame)
    return canonical.loc[canonical["availability_time"] <= decision].copy()


def select_phase_rows(
    frame: pd.DataFrame,
    *,
    phase: str,
    protected_start: str | pd.Timestamp,
    protected_access_authorized: bool = False,
) -> pd.DataFrame:
    """Fail closed around the protected boundary."""
    report = validate_point_in_time_frame(frame)
    if not report.valid:
        raise ValueError("Invalid point-in-time frame: " + "; ".join(report.errors))
    boundary = pd.Timestamp(protected_start)
    if boundary.tzinfo is None:
        boundary = boundary.tz_localize("UTC")
    else:
        boundary = boundary.tz_convert("UTC")
    canonical = _canonicalize(frame)
    if phase in {"development", "validation"}:
        return canonical.loc[canonical["event_time"] < boundary].copy()
    if phase in {"confirmatory", "protected"}:
        if not protected_access_authorized:
            raise PermissionError("Protected rows require explicit execution authorization")
        return canonical.loc[canonical["event_time"] >= boundary].copy()
    raise ValueError("phase must be development, validation, confirmatory or protected")


def write_deterministic_snapshot(
    frame: pd.DataFrame,
    destination: str | Path,
    *,
    snapshot_id: str,
    source_name: str,
    license_or_right: str,
    retrieved_at: str,
    base_currency: str = "USD",
) -> DataManifest:
    report = validate_point_in_time_frame(frame)
    if not report.valid:
        raise ValueError("Snapshot rejected: " + "; ".join(report.errors))
    if not source_name or not license_or_right:
        raise ValueError("Source and license/usage right are required")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = _canonicalize(frame)
    serial = canonical.copy()
    for column in ("event_time", "availability_time", "outcome_available_time"):
        serial[column] = serial[column].map(lambda value: value.isoformat())
    payload = serial.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    manifest = DataManifest(
        snapshot_id=snapshot_id,
        source_name=source_name,
        license_or_right=license_or_right,
        retrieved_at=retrieved_at,
        row_count=len(serial),
        columns=tuple(serial.columns),
        minimum_event_time=str(serial["event_time"].min()),
        maximum_event_time=str(serial["event_time"].max()),
        sha256=digest,
        base_currency=base_currency,
    )
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_snapshot(path: str | Path, manifest: DataManifest) -> bool:
    payload = Path(path).read_bytes()
    return hashlib.sha256(payload).hexdigest() == manifest.sha256


def load_data_manifest(path: str | Path) -> DataManifest:
    """Load a strict snapshot manifest without accepting unknown or missing fields."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Data manifest must be a JSON object")
    payload["columns"] = tuple(payload.get("columns", ()))
    try:
        return DataManifest(**payload)
    except TypeError as error:
        raise ValueError(f"Invalid data manifest schema: {error}") from error
