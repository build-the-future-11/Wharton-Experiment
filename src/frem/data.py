"""Strict point-in-time panel adapters and causal windows; no network side effects.

The price timestamp is a market event time. Availability is a separate, required
timestamp. Labels enter at the next executable event strictly after a decision.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


class DataValidationError(ValueError):
    """A panel cannot support the claimed information or accounting boundary."""


def _utc(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise DataValidationError(f"{name} has missing timestamps")
    for value in values:
        if pd.Timestamp(value).tzinfo is None:
            raise DataValidationError(
                f"{name} requires timezone-qualified event times, not dates or naive timestamps"
            )
    return pd.to_datetime(values, utc=True, errors="raise").astype("datetime64[ns, UTC]")


def _bool(values: pd.Series, name: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "1": True,
        "0": False,
    }
    result = values.map(mapping)
    if result.isna().any():
        raise DataValidationError(f"{name} must contain explicit booleans")
    return result.astype(bool)


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Hash canonical values plus provenance, independent of caller row order."""
    order = [
        c for c in ("asset", "timestamp", "available_at", "series", "observation_time") if c in frame.columns
    ]
    canonical = frame.sort_values(order) if order else frame
    canonical = canonical.reindex(sorted(canonical.columns), axis=1)
    payload = canonical.to_csv(index=False, float_format="%.17g", na_rep="<MISSING>")
    provenance = {k: v for k, v in frame.attrs.items() if k not in {"fingerprint", "file_sha256"}}
    encoded = json.dumps(provenance, sort_keys=True, default=str, separators=(",", ":"))
    return sha256((payload + "\n" + encoded).encode("utf-8")).hexdigest()


def load_panel(path: str | Path, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Load CSV/parquet. ``column_map`` maps source names to canonical names.

    An optional ``<path>.manifest.json`` is loaded as provenance. No adjustment,
    timezone guessing, backward filling, or remote fetching is performed.
    """
    source = Path(path)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    elif source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        raise DataValidationError("Supported panel formats are .csv and .parquet/.pq")
    if column_map:
        unknown = set(column_map) - set(frame.columns)
        if unknown:
            raise DataValidationError(f"Mapped source columns do not exist: {sorted(unknown)}")
        frame = frame.rename(columns=column_map)
    if frame.columns.duplicated().any():
        raise DataValidationError("Column mapping created duplicate canonical names")
    for name in frame.columns:
        if name.endswith("_at") or name in {"timestamp", "eligible_from", "eligible_to"}:
            if name == "eligible_to" and frame[name].isna().any():
                known = frame[name].notna()
                if known.any():
                    _utc(frame.loc[known, name], name)
                frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
            else:
                frame[name] = _utc(frame[name], name)
    for name in ("eligible", "executable"):
        if name in frame:
            frame[name] = _bool(frame[name], name)
    numeric = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "total_return_index",
        "base_fx",
        "split_ratio",
        "cash_distribution",
        "dirty_price",
    }
    numeric.update(c for c in frame.columns if c.startswith("feature_"))
    numeric.discard("feature_available_at")
    for name in sorted(numeric.intersection(frame.columns)):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    manifest_path = Path(str(source) + ".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise DataValidationError("Panel manifest must be a JSON object")
        frame.attrs.update(manifest)
    raw_hash = sha256(source.read_bytes()).hexdigest()
    expected = frame.attrs.get("expected_file_sha256")
    if expected and expected != raw_hash:
        raise DataValidationError("Panel file hash disagrees with its manifest")
    frame.attrs["file_sha256"] = raw_hash
    frame.attrs["source_file"] = source.name
    if {"asset", "timestamp"}.issubset(frame.columns):
        frame = frame.sort_values(["asset", "timestamp"], kind="stable").reset_index(drop=True)
    frame.attrs["fingerprint"] = data_fingerprint(frame)
    return frame


def _effective_availability(frame: pd.DataFrame) -> pd.Series:
    names = [
        c
        for c in ("available_at", "fx_available_at", "corporate_actions_available_at", "feature_available_at")
        if c in frame
    ]
    columns = [_utc(frame[c], c).astype("int64").to_numpy() for c in names]
    return pd.Series(pd.to_datetime(np.maximum.reduce(columns), utc=True), index=frame.index)


def validate_panel(frame: pd.DataFrame, strict: bool = True) -> dict[str, Any]:
    """Validate explicit provenance and accounting. Strict mode fails closed.

    The universe attestation is evidence metadata, not proof that a vendor is
    complete. Present-day constituent exports are rejected for confirmatory work.
    """
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "timestamp",
        "available_at",
        "asset",
        "close",
        "currency",
        "asset_type",
        "price_kind",
        "eligible",
        "eligibility_known_at",
        "universe_source",
        "executable",
        "source",
    }
    absent = sorted(required - set(frame.columns))
    if absent:
        errors.append(f"Missing canonical columns: {absent}")
    if len(frame) == 0:
        errors.append("Panel is empty")
    if not absent and len(frame):
        work = frame.copy()
        for name in ("timestamp", "available_at", "eligibility_known_at"):
            try:
                work[name] = _utc(work[name], name)
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
        for name in ("eligible", "executable"):
            try:
                work[name] = _bool(work[name], name)
            except DataValidationError as exc:
                errors.append(str(exc))
        for name in ("eligible_from", "eligible_to", "classification_known_at"):
            if name in work:
                known = work[name].notna()
                try:
                    if known.any():
                        _utc(work.loc[known, name], name)
                except (ValueError, TypeError) as exc:
                    errors.append(str(exc))
        if work.duplicated(["asset", "timestamp"]).any():
            errors.append("Duplicate (asset,timestamp) rows; revisions need a separate vintage adapter")
        for name in ("asset", "currency", "asset_type", "price_kind", "source", "universe_source"):
            if work[name].isna().any() or work[name].astype(str).str.strip().eq("").any():
                errors.append(f"{name} contains missing/empty identifiers")
        prices = pd.to_numeric(work["close"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(prices) & (prices > 0)):
            errors.append("close must be a finite positive traded/valuation price, never a yield")
        allowed_types = {"equity", "etf", "bond", "bond_etf", "fund", "commodity_etf"}
        bad_types = set(work["asset_type"].astype(str)) - allowed_types
        if bad_types:
            errors.append(f"Unsupported asset_type: {sorted(bad_types)}")
        allowed_prices = {"total_return_index", "raw_with_actions", "bond_dirty_with_coupons"}
        bad_prices = set(work["price_kind"].astype(str)) - allowed_prices
        if bad_prices:
            errors.append(
                f"Unsupported price_kind: {sorted(bad_prices)}; yields/adjusted-close guesses rejected"
            )
        for asset, group in work.groupby("asset", sort=False):
            group = group.sort_values("timestamp")
            for name in ("currency", "asset_type", "price_kind"):
                if group[name].nunique(dropna=False) != 1:
                    errors.append(f"{asset}: {name} changes; split the instrument into explicit identifiers")
            kind = str(group["price_kind"].iloc[0])
            typ = str(group["asset_type"].iloc[0])
            if kind == "total_return_index":
                if "total_return_index" not in group:
                    errors.append(f"{asset}: total_return_index values missing")
                else:
                    values = pd.to_numeric(group["total_return_index"], errors="coerce").to_numpy(float)
                    if not np.all(np.isfinite(values) & (values > 0)):
                        errors.append(f"{asset}: total-return index must be finite and positive")
                if "corporate_action_source" not in group or group["corporate_action_source"].isna().any():
                    errors.append(f"{asset}: total-return methodology/corporate_action_source missing")
            if kind in {"raw_with_actions", "bond_dirty_with_coupons"}:
                action_cols = {
                    "split_ratio",
                    "cash_distribution",
                    "corporate_actions_available_at",
                    "corporate_action_source",
                }
                if action_cols - set(group):
                    errors.append(
                        f"{asset}: explicit split, distribution and availability provenance required"
                    )
                else:
                    splits = pd.to_numeric(group["split_ratio"], errors="coerce").to_numpy(float)
                    dividends = pd.to_numeric(group["cash_distribution"], errors="coerce").to_numpy(float)
                    if not np.all(np.isfinite(splits) & (splits > 0)) or not np.all(np.isfinite(dividends)):
                        errors.append(f"{asset}: invalid split_ratio or cash_distribution")
            if typ == "bond" and kind not in {"total_return_index", "bond_dirty_with_coupons"}:
                errors.append(f"{asset}: actual bonds need total return or dirty price plus coupons")
            if kind == "bond_dirty_with_coupons" and typ != "bond":
                errors.append(f"{asset}: bond accounting requires asset_type=bond")
            if (
                typ == "bond"
                and kind == "bond_dirty_with_coupons"
                and not bool(group.get("split_ratio", pd.Series([1])).eq(1).all())
            ):
                errors.append(f"{asset}: bond split_ratio must be one; encode redemption separately")
        if {"high", "low"}.issubset(work):
            bad = (
                (work["high"] < work["low"]) | (work["high"] < work["close"]) | (work["low"] > work["close"])
            )
            if bad.any():
                errors.append("OHLC range does not contain close")
        if "volume" in work and (pd.to_numeric(work["volume"], errors="coerce").dropna() < 0).any():
            errors.append("volume must be nonnegative when observed")
        if not errors:
            if (work["available_at"] < work["timestamp"]).any():
                errors.append("Prices are marked available before their market event")
            if (work["eligibility_known_at"] > work["available_at"]).any():
                errors.append("Eligibility was not known by the price publication time")
            try:
                effective = _effective_availability(work)
                for asset, index in work.groupby("asset").groups.items():
                    rows = work.loc[index].sort_values("timestamp").index
                    values = effective.loc[rows].astype("int64").to_numpy()
                    if np.any(np.diff(values) <= 0):
                        errors.append(
                            f"{asset}: feature availability must increase strictly with market events"
                        )
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
        base_currency = str(frame.attrs.get("base_currency", "USD"))
        foreign = work["currency"].astype(str) != base_currency
        if foreign.any():
            if not {"base_fx", "fx_available_at", "fx_source"}.issubset(work):
                errors.append("Foreign-currency assets require base_fx, fx_available_at and fx_source")
            else:
                fx = pd.to_numeric(work.loc[foreign, "base_fx"], errors="coerce").to_numpy(float)
                if not np.all(np.isfinite(fx) & (fx > 0)):
                    errors.append("base_fx must be positive base-currency units per local currency unit")
                if work.loc[foreign, "fx_source"].isna().any():
                    errors.append("Foreign-currency FX provenance missing")
        feature_cols = [c for c in work if c.startswith("feature_") and c != "feature_available_at"]
        if feature_cols and "feature_available_at" not in work:
            errors.append("External feature_* columns require feature_available_at")
        if feature_cols and not frame.attrs.get("feature_provenance"):
            errors.append("External feature_* columns require manifest feature_provenance")
    status = frame.attrs.get("universe_status")
    if status not in {"point_in_time_complete", "fixed_ex_ante"}:
        errors.append("Manifest universe_status must be point_in_time_complete or fixed_ex_ante")
    if status == "fixed_ex_ante" and not frame.attrs.get("universe_frozen_at"):
        errors.append("fixed_ex_ante requires the original universe_frozen_at timestamp")
    if frame.attrs.get("delistings_included") is not True and status == "point_in_time_complete":
        errors.append("Point-in-time complete claim requires delistings_included=true and a source audit")
    if not frame.attrs.get("availability_policy"):
        errors.append("Manifest availability_policy is required; a daily date is not a market close time")
    if not frame.attrs.get("total_return_methodology"):
        errors.append("Manifest total_return_methodology must describe distributions and delistings")
    warnings.append(
        "Provenance attestations require source review; schema checks cannot establish vendor completeness."
    )
    report = {
        "valid": not errors,
        "rows": len(frame),
        "assets": int(frame["asset"].nunique()) if "asset" in frame else 0,
        "errors": errors,
        "warnings": warnings,
        "fingerprint": data_fingerprint(frame),
    }
    if strict and errors:
        raise DataValidationError("Panel validation failed:\n- " + "\n- ".join(errors))
    return report


def _local_wealth(group: pd.DataFrame) -> np.ndarray:
    kind = str(group["price_kind"].iloc[0])
    if kind == "total_return_index":
        return group["total_return_index"].to_numpy(dtype=float)
    close = group["close"].to_numpy(dtype=float)
    split = group["split_ratio"].to_numpy(dtype=float)
    distribution = group["cash_distribution"].to_numpy(dtype=float)
    gross = (close[1:] * split[1:] + distribution[1:]) / close[:-1]
    if not np.all(np.isfinite(gross) & (gross > 0)):
        raise DataValidationError("Corporate actions create nonpositive total wealth")
    return np.r_[1.0, np.cumprod(gross)]


def _feature_frame(group: pd.DataFrame, base_currency: str) -> pd.DataFrame:
    local = _local_wealth(group)
    fx = (
        np.ones(len(group))
        if str(group["currency"].iloc[0]) == base_currency
        else group["base_fx"].to_numpy(dtype=float)
    )
    base = local * fx
    ret = pd.Series(np.r_[np.nan, np.diff(np.log(base))], index=group.index)
    output = pd.DataFrame(index=group.index)
    output["log_return"] = ret
    output["realized_vol_20"] = ret.rolling(20, min_periods=2).std(ddof=1)
    output["momentum_5"] = ret.rolling(5, min_periods=1).sum()
    output["momentum_20"] = ret.rolling(20, min_periods=1).sum()
    output["elapsed_days"] = group["timestamp"].diff().dt.total_seconds() / 86400.0
    output["availability_lag_days"] = (
        _effective_availability(group) - group["timestamp"]
    ).dt.total_seconds() / 86400.0
    if "volume" in group:
        output["log_volume"] = np.log1p(group["volume"].to_numpy(dtype=float))
    if {"high", "low"}.issubset(group):
        output["high_low_fraction"] = (group["high"] - group["low"]) / group["close"]
    for name in group:
        if name.startswith("feature_") and name != "feature_available_at":
            output[name] = pd.to_numeric(group[name], errors="raise")
    output.attrs["base_wealth"] = base
    return output


def point_in_time_macro(
    macro: pd.DataFrame, decision_times: Sequence[Any], max_age_days: float | None = None
) -> pd.DataFrame:
    """Join released vintages, choosing the newest observation period known now.

    A late revision to an old observation does not replace a newer observation.
    Dates with no published value remain missing. Never backward fill.
    """
    required = {"series", "observation_time", "available_at", "value"}
    if required - set(macro):
        raise DataValidationError(f"Macro table missing {sorted(required - set(macro))}")
    table = macro.copy()
    if max_age_days is not None and max_age_days < 0:
        raise ValueError("max_age_days cannot be negative")
    table["available_at"] = _utc(table["available_at"], "macro.available_at")
    table["observation_time"] = _utc(table["observation_time"], "macro.observation_time")
    if (table["observation_time"] > table["available_at"]).any():
        raise DataValidationError("Macro actuals cannot precede the period they describe")
    if table.duplicated(["series", "observation_time", "available_at"]).any():
        raise DataValidationError("Duplicate macro observation/vintage keys")
    table["value"] = pd.to_numeric(table["value"].replace(".", np.nan), errors="raise")
    if table["series"].isna().any() or table["series"].astype(str).str.strip().eq("").any():
        raise DataValidationError("Macro series identifiers cannot be missing")
    if np.isinf(table["value"].to_numpy(float)).any():
        raise DataValidationError("Macro observations cannot be infinite")
    table["series"] = table["series"].astype(str)
    series_names = list(table.attrs.get("series_universe", sorted(table["series"].unique())))
    if len(series_names) != len(set(series_names)) or set(table["series"]) - set(series_names):
        raise DataValidationError("Macro series_universe must uniquely include every observed series")
    times = pd.DatetimeIndex(pd.to_datetime(list(decision_times), utc=True)).as_unit("ns")
    if times.hasnans:
        raise DataValidationError("Decision times cannot be missing")
    order = np.argsort(times.asi8, kind="stable")
    output = pd.DataFrame(index=np.arange(len(times)))
    for series in sorted(series_names):
        rows = table.loc[table["series"] == series]
        rows = rows.sort_values(["available_at", "observation_time"])
        records = list(rows.itertuples(index=False))
        state: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
        cursor = 0
        values = np.full(len(times), np.nan)
        ages = np.full(len(times), np.nan)
        for index in order:
            decision = times[index]
            while cursor < len(records) and records[cursor].available_at <= decision:
                row = records[cursor]
                state[row.observation_time] = (float(row.value), row.available_at)
                cursor += 1
            if state:
                period = max(state)
                value, release = state[period]
                age = (decision - period).total_seconds() / 86400.0
                if max_age_days is None or age <= max_age_days:
                    values[index] = value
                    ages[index] = (decision - release).total_seconds() / 86400.0
        output[f"macro_{series}"] = values
        output[f"macro_{series}_release_age_days"] = ages
    return output


def add_causal_peer_features(frame: pd.DataFrame, max_age: str = "3D", min_peers: int = 2) -> pd.DataFrame:
    """Add known market/sector peer-return snapshots with bounded staleness.

    Aggregates refer to each peer's most recent observed interval, which may not
    be the same wall-clock interval in another market. They are covariates, never
    synchronous portfolio returns or estimated graph edges.
    """
    validate_panel(frame)
    required = {"market", "sector", "classification_known_at"}
    if required - set(frame):
        raise DataValidationError(f"Peer features need {sorted(required - set(frame))}")
    if min_peers < 1:
        raise ValueError("min_peers must be positive")
    try:
        age_limit_ns = int(to_offset(max_age).nanos)
    except (TypeError, ValueError) as error:
        raise ValueError("max_age must be a fixed positive duration") from error
    if age_limit_ns <= 0:
        raise ValueError("max_age must be positive")
    age_limit = pd.Timedelta(age_limit_ns, unit="ns")
    result = frame.copy().reset_index(drop=True)
    for name in ("timestamp", "available_at", "eligibility_known_at"):
        result[name] = _utc(result[name], name)
    result["eligible"] = _bool(result["eligible"], "eligible")
    result["classification_known_at"] = _utc(result["classification_known_at"], "classification_known_at")
    effective = _effective_availability(result)
    if (result["classification_known_at"] > effective).any():
        raise DataValidationError("Market/sector classification uses future knowledge")
    returns = pd.Series(np.nan, index=result.index, dtype=float)
    base_currency = str(frame.attrs.get("base_currency", "USD"))
    for _, group in result.groupby("asset", sort=False):
        group = group.sort_values("timestamp")
        returns.loc[group.index] = _feature_frame(group, base_currency)["log_return"]
    names = [
        "feature_market_peer_return",
        "feature_sector_peer_return",
        "feature_market_peer_dispersion",
        "feature_market_peer_count",
        "feature_sector_peer_count",
    ]
    for name in names:
        result[name] = np.nan
    state: dict[str, dict[str, Any]] = {}
    for decision, indices in effective.groupby(effective, sort=True).groups.items():
        for index in indices:
            row = result.loc[index]
            state[str(row["asset"])] = {
                "time": decision,
                "return": float(returns.loc[index]),
                "market": row["market"],
                "sector": row["sector"],
                "eligible": bool(row["eligible"]),
                "eligible_from": row.get("eligible_from"),
                "eligible_to": row.get("eligible_to"),
            }
        for index in indices:
            row = result.loc[index]
            peers = [
                entry
                for asset, entry in state.items()
                if asset != str(row["asset"])
                and entry["eligible"]
                and decision - entry["time"] <= age_limit
                and np.isfinite(entry["return"])
                and (pd.isna(entry["eligible_from"]) or pd.Timestamp(entry["eligible_from"]) <= decision)
                and (pd.isna(entry["eligible_to"]) or pd.Timestamp(entry["eligible_to"]) > decision)
            ]
            market = [x["return"] for x in peers if x["market"] == row["market"]]
            sector = [
                x["return"] for x in peers if x["sector"] == row["sector"] and x["market"] == row["market"]
            ]
            result.loc[index, "feature_market_peer_count"] = len(market)
            result.loc[index, "feature_sector_peer_count"] = len(sector)
            if len(market) >= min_peers:
                result.loc[index, "feature_market_peer_return"] = float(np.mean(market))
                result.loc[index, "feature_market_peer_dispersion"] = float(np.std(market))
            if len(sector) >= min_peers:
                result.loc[index, "feature_sector_peer_return"] = float(np.mean(sector))
    result["feature_available_at"] = effective
    provenance = dict(result.attrs.get("feature_provenance", {}))
    provenance["causal_peer_snapshots"] = {
        "max_age": str(age_limit),
        "min_peers": min_peers,
        "method": "latest available returns, self excluded",
    }
    result.attrs["feature_provenance"] = provenance
    result.attrs["fingerprint"] = data_fingerprint(result)
    return result


@dataclass
class WindowData:
    x: np.ndarray
    future_x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    label_time: np.ndarray
    asset_id: np.ndarray
    feature_names: list[str]
    asset_names: list[str]
    entry_time: np.ndarray | None = None
    exit_time: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x)
        self.future_x = np.asarray(self.future_x)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.time = np.asarray(self.time, dtype=np.int64)
        self.label_time = np.asarray(self.label_time, dtype=np.int64)
        self.asset_id = np.asarray(self.asset_id, dtype=np.int64)
        n = len(self.y)
        if self.x.ndim != 3 or self.future_x.ndim != 3:
            raise DataValidationError("x and future_x must have shapes [N,L,F] and [N,H,F]")
        if self.x.shape[0] != n or self.future_x.shape[0] != n:
            raise DataValidationError("Window arrays do not share their sample count")
        if self.x.shape[2] != len(self.feature_names) or self.future_x.shape[2] != len(self.feature_names):
            raise DataValidationError("Feature names do not match feature dimensions")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise DataValidationError("Feature names must be unique")
        for name in ("time", "label_time", "asset_id"):
            if getattr(self, name).shape != (n,):
                raise DataValidationError(f"{name} must have shape [N]")
        if n and (
            np.any(self.label_time <= self.time)
            or np.any(self.asset_id < 0)
            or np.any(self.asset_id >= len(self.asset_names))
        ):
            raise DataValidationError("Invalid label maturity or asset IDs")
        order = np.lexsort((self.asset_id, self.time))
        if not np.array_equal(order, np.arange(n)):
            raise DataValidationError("Rows must be sorted by (time,asset_id)")
        if n and pd.DataFrame({"t": self.time, "a": self.asset_id}).duplicated().any():
            raise DataValidationError("Duplicate decision/asset rows")
        for name in ("entry_time", "exit_time"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=np.int64)
                if value.shape != (n,):
                    raise DataValidationError(f"{name} must have shape [N]")
                setattr(self, name, value)
        if (self.entry_time is None) != (self.exit_time is None):
            raise DataValidationError("entry_time and exit_time must either both be supplied or both be omitted")
        if self.entry_time is not None and np.any(self.entry_time <= self.time):
            raise DataValidationError("Entry must occur strictly after the decision")
        if (
            self.exit_time is not None
            and self.entry_time is not None
            and (np.any(self.exit_time <= self.entry_time) or np.any(self.label_time < self.exit_time))
        ):
            raise DataValidationError("Invalid entry, exit or publication ordering")

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, indices: Sequence[int] | np.ndarray) -> WindowData:
        index = np.asarray(indices)
        if index.dtype == bool:
            index = np.flatnonzero(index)
        index = index.astype(np.int64)
        # Chronological order is part of the public contract, even after selection.
        index = index[np.lexsort((self.asset_id[index], self.time[index]))]
        return WindowData(
            self.x[index],
            self.future_x[index],
            self.y[index],
            self.time[index],
            self.label_time[index],
            self.asset_id[index],
            list(self.feature_names),
            list(self.asset_names),
            None if self.entry_time is None else self.entry_time[index],
            None if self.exit_time is None else self.exit_time[index],
            dict(self.metadata),
        )


def build_windows(
    frame: pd.DataFrame,
    context: int,
    horizon: int,
    feature_columns: Sequence[str] | None = None,
    macro: pd.DataFrame | None = None,
    base_currency: str = "USD",
    include_missingness: bool = True,
    macro_max_age_days: float | None = None,
) -> WindowData:
    """Build causal contexts and executable total-return labels.

    The next tradable close is the execution assumption, not a promise of a fill.
    Price impact/spread belong in the explicit execution ledger. A one-bar delay
    means the first target interval begins at the *next* observed executable price.
    """
    if context < 2 or horizon < 1:
        raise ValueError("context >= 2 and horizon >= 1 are required")
    work = frame.copy()
    work.attrs["base_currency"] = base_currency
    report = validate_panel(work)
    for name in ("timestamp", "available_at", "eligibility_known_at"):
        work[name] = _utc(work[name], name)
    assets = sorted(work["asset"].astype(str).unique())
    asset_lookup = {name: index for index, name in enumerate(assets)}
    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    ys: list[float] = []
    times: list[int] = []
    labels: list[int] = []
    ids: list[int] = []
    entries: list[int] = []
    exits: list[int] = []
    names: list[str] | None = None
    dropped = {"ineligible": 0, "unmatured_target": 0, "invalid_target": 0}
    for asset, group in work.groupby("asset", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        effective = _effective_availability(group)
        event_ns = group["timestamp"].astype("int64").to_numpy()
        available_ns = effective.astype("int64").to_numpy()
        features = _feature_frame(group, base_currency)
        wealth = features.attrs["base_wealth"]
        if macro is not None:
            joined = point_in_time_macro(macro, effective, max_age_days=macro_max_age_days)
            for name in joined:
                features[name] = joined[name].to_numpy()
        chosen = list(feature_columns) if feature_columns is not None else list(features.columns)
        missing = set(chosen) - set(features)
        if missing:
            raise DataValidationError(f"Unknown requested feature columns: {sorted(missing)}")
        values = features[chosen].to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise DataValidationError("Infinite features must be corrected at source, not imputed")
        current_names = chosen
        if include_missingness:
            values = np.concatenate([values, (~np.isfinite(values)).astype(float)], axis=1)
            current_names = chosen + [f"{name}__missing" for name in chosen]
        if names is not None and names != current_names:
            raise DataValidationError("Assets have inconsistent feature schemas")
        names = current_names
        executable = np.flatnonzero(_bool(group["executable"], "executable").to_numpy())
        eligible = _bool(group["eligible"], "eligible").to_numpy()
        eligibility_ns = group["eligibility_known_at"].astype("int64").to_numpy()
        for end in range(context - 1, len(group)):
            decision = available_ns[end]
            if not eligible[end] or eligibility_ns[end] > decision:
                dropped["ineligible"] += 1
                continue
            if "eligible_from" in group and pd.Timestamp(group.loc[end, "eligible_from"]).value > decision:
                dropped["ineligible"] += 1
                continue
            if (
                "eligible_to" in group
                and pd.notna(group.loc[end, "eligible_to"])
                and pd.Timestamp(group.loc[end, "eligible_to"]).value <= decision
            ):
                dropped["ineligible"] += 1
                continue
            if work.attrs.get("universe_status") == "fixed_ex_ante":
                frozen = pd.Timestamp(work.attrs["universe_frozen_at"])
                if frozen.tzinfo is None:
                    raise DataValidationError("universe_frozen_at must include a timezone")
                if frozen.value > decision:
                    dropped["ineligible"] += 1
                    continue
            position = int(np.searchsorted(event_ns[executable], decision, side="right"))
            if position + horizon >= len(executable):
                dropped["unmatured_target"] += 1
                continue
            entry = int(executable[position])
            future_rows = executable[position + 1 : position + horizon + 1]
            exit_row = int(future_rows[-1])
            target = float(np.log(wealth[exit_row] / wealth[entry]))
            maturity = int(max(available_ns[entry], available_ns[future_rows].max()))
            if not np.isfinite(target):
                dropped["invalid_target"] += 1
                continue
            contexts.append(values[end - context + 1 : end + 1].astype(np.float32))
            futures.append(values[future_rows].astype(np.float32))
            ys.append(target)
            times.append(int(decision))
            labels.append(maturity)
            ids.append(asset_lookup[str(asset)])
            entries.append(int(event_ns[entry]))
            exits.append(int(event_ns[exit_row]))
    if not contexts or names is None:
        raise DataValidationError(
            "No usable windows: check length, context, eligibility and future executable prices"
        )
    order = np.lexsort((np.asarray(ids), np.asarray(times)))
    return WindowData(
        np.stack(contexts)[order],
        np.stack(futures)[order],
        np.asarray(ys)[order],
        np.asarray(times, dtype=np.int64)[order],
        np.asarray(labels, dtype=np.int64)[order],
        np.asarray(ids, dtype=np.int64)[order],
        names,
        assets,
        np.asarray(entries, dtype=np.int64)[order],
        np.asarray(exits, dtype=np.int64)[order],
        {
            "panel_fingerprint": report["fingerprint"],
            "context": context,
            "horizon": horizon,
            "base_currency": base_currency,
            "execution": "next executable close strictly after decision",
            "label": "cumulative base-currency log total return over H executable intervals",
            "dropped_rows": dropped,
            "missingness_indicators": include_missingness,
            "macro_fingerprint": data_fingerprint(macro) if macro is not None else None,
        },
    )


def build_inference_windows(
    frame: pd.DataFrame,
    context: int,
    horizon: int,
    feature_columns: Sequence[str] | None = None,
    macro: pd.DataFrame | None = None,
    base_currency: str = "USD",
    include_missingness: bool = True,
    macro_max_age_days: float | None = None,
    as_of: str | int | pd.Timestamp | None = None,
    max_staleness: str = "7D",
) -> WindowData:
    """Latest eligible contexts without requiring, inventing or loading outcomes.

    ``future_x`` is an unused all-NaN shape carrier; y is NaN and label_time is
    int64 max (unknown). Frozen prediction must consume x only. There are no
    known entry/exit times until a future execution observation actually arrives.
    """
    if context < 2 or horizon < 1:
        raise ValueError("context >= 2 and horizon >= 1 are required")
    work = frame.copy()
    work.attrs["base_currency"] = base_currency
    validate_panel(work)
    for name in ("timestamp", "available_at", "eligibility_known_at"):
        work[name] = _utc(work[name], name)
    effective = _effective_availability(work)
    if as_of is None:
        decision = int(effective.astype("int64").max())
    elif isinstance(as_of, (int, np.integer)):
        decision = int(as_of)
    else:
        value = pd.Timestamp(as_of)
        if value.tzinfo is None:
            raise DataValidationError("as_of requires an explicit timezone")
        decision = int(value.value)
    try:
        age_limit = int(to_offset(max_staleness).nanos)
    except (TypeError, ValueError) as error:
        raise ValueError("max_staleness must be a fixed positive duration") from error
    if age_limit <= 0:
        raise ValueError("max_staleness must be positive")
    work = work.loc[
        (effective.astype("int64") <= decision) & (work["timestamp"].astype("int64") <= decision)
    ].copy()
    if not len(work):
        raise DataValidationError("No information was available by as_of")
    assets = sorted(frame["asset"].astype(str).unique())
    lookup = {asset: i for i, asset in enumerate(assets)}
    arrays: list[np.ndarray] = []
    asset_ids: list[int] = []
    names: list[str] | None = None
    ages: dict[str, float] = {}
    skipped: dict[str, str] = {}
    frozen_at = work.attrs.get("universe_frozen_at")
    if work.attrs.get("universe_status") == "fixed_ex_ante" and pd.Timestamp(frozen_at).value > decision:
        raise DataValidationError("The fixed universe had not yet been declared at as_of")
    for asset, group in work.groupby("asset", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        availability = _effective_availability(group)
        if len(group) < context:
            skipped[str(asset)] = "insufficient context"
            continue
        last = group.iloc[-1]
        age = decision - int(availability.iloc[-1].value)
        if age > age_limit:
            skipped[str(asset)] = "stale last observation"
            continue
        if not bool(_bool(group["eligible"], "eligible").iloc[-1]):
            skipped[str(asset)] = "not eligible at last known snapshot"
            continue
        if int(last["eligibility_known_at"].value) > decision:
            skipped[str(asset)] = "eligibility not known"
            continue
        if "eligible_from" in group and pd.Timestamp(last["eligible_from"]).value > decision:
            skipped[str(asset)] = "eligibility has not begun"
            continue
        if (
            "eligible_to" in group
            and pd.notna(last["eligible_to"])
            and pd.Timestamp(last["eligible_to"]).value <= decision
        ):
            skipped[str(asset)] = "eligibility has expired"
            continue
        features = _feature_frame(group, base_currency)
        if macro is not None:
            joined = point_in_time_macro(macro, availability, max_age_days=macro_max_age_days)
            for name in joined:
                features[name] = joined[name].to_numpy()
        selected = list(feature_columns) if feature_columns is not None else list(features.columns)
        if set(selected) - set(features):
            raise DataValidationError(f"Unknown inference features {sorted(set(selected) - set(features))}")
        values = features[selected].to_numpy(float)[-context:]
        if np.isinf(values).any():
            raise DataValidationError("Infinite inference features")
        current_names = selected
        if include_missingness:
            values = np.concatenate([values, (~np.isfinite(values)).astype(float)], axis=1)
            current_names = selected + [f"{name}__missing" for name in selected]
        if names is not None and names != current_names:
            raise DataValidationError("Inference feature schema varies between assets")
        names = current_names
        arrays.append(values.astype(np.float32))
        asset_ids.append(lookup[str(asset)])
        ages[str(asset)] = age / 86400e9
    if not arrays or names is None:
        raise DataValidationError("No eligible, sufficiently recent asset has a complete context")
    n = len(arrays)
    return WindowData(
        np.stack(arrays),
        np.full((n, horizon, len(names)), np.nan, dtype=np.float32),
        np.full(n, np.nan),
        np.full(n, decision, dtype=np.int64),
        np.full(n, np.iinfo(np.int64).max, dtype=np.int64),
        np.asarray(asset_ids),
        names,
        assets,
        metadata={
            "mode": "unlabeled_inference",
            "context": context,
            "horizon": horizon,
            "base_currency": base_currency,
            "as_of_ns": decision,
            "max_staleness": max_staleness,
            "observation_age_days": ages,
            "skipped_assets": skipped,
            "panel_fingerprint": data_fingerprint(work),
            "macro_fingerprint": data_fingerprint(macro) if macro is not None else None,
            "execution": "future executable observation not yet known",
        },
    )


def align_asset_universe(data: WindowData, reference_names: Sequence[str]) -> WindowData:
    """Reuse the trained asset-ID mapping for a narrower inference panel.

    This does not silently discard unknown assets: they require a predeclared
    cold-start model or a separately trained model and fail closed here.
    """
    reference = list(reference_names)
    if len(reference) != len(set(reference)):
        raise ValueError("Reference asset names contain duplicates")
    lookup = {name: index for index, name in enumerate(reference)}
    observed_names = [data.asset_names[int(i)] for i in data.asset_id]
    unknown = set(observed_names) - set(reference)
    if unknown:
        raise DataValidationError(f"Unseen assets require a cold-start protocol: {sorted(unknown)}")
    mapped = np.asarray([lookup[name] for name in observed_names], dtype=np.int64)
    order = np.lexsort((mapped, data.time))
    return WindowData(
        data.x[order],
        data.future_x[order],
        data.y[order],
        data.time[order],
        data.label_time[order],
        mapped[order],
        list(data.feature_names),
        reference,
        None if data.entry_time is None else data.entry_time[order],
        None if data.exit_time is None else data.exit_time[order],
        dict(data.metadata),
    )


@dataclass
class TrainOnlyRobustScaler:
    clip: float | None = 12.0
    eps: float = 1e-8
    median_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    feature_names_: list[str] = field(default_factory=list)
    mask_: np.ndarray | None = None
    fit_metadata_: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be positive and finite")
        if self.clip is not None and (not np.isfinite(self.clip) or self.clip <= 0):
            raise ValueError("clip must be positive and finite or None")

    def fit(self, train: WindowData) -> TrainOnlyRobustScaler:
        if self.median_ is not None:
            raise ValueError("Scaler already fitted; create a fresh instance for each training split")
        if len(train) == 0:
            raise ValueError("Cannot fit scaler to empty training data")
        values = np.asarray(train.x, dtype=np.float64).reshape(-1, train.x.shape[-1])
        if np.isinf(values).any():
            raise DataValidationError("Infinite training features are invalid")
        all_missing = np.all(np.isnan(values), axis=0)
        safe = values.copy()
        safe[:, all_missing] = 0.0
        median = np.nanmedian(safe, axis=0)
        q25, q75 = np.nanquantile(safe, [0.25, 0.75], axis=0)
        scale = np.maximum(q75 - q25, self.eps)
        constant = (q75 - q25) <= self.eps
        scale[constant] = 1.0
        mask = np.array([name.endswith("__missing") for name in train.feature_names], dtype=bool)
        median[mask], scale[mask] = 0.0, 1.0
        self.median_, self.scale_, self.mask_ = median, scale, mask
        self.feature_names_ = list(train.feature_names)
        training_hash = sha256()
        for array in (train.x, train.time, train.asset_id):
            training_hash.update(np.ascontiguousarray(array).tobytes())
        self.fit_metadata_ = {
            "fit_time_max": int(train.time.max()),
            "fit_label_time_max": int(train.label_time.max()),
            "fit_rows": len(train),
            "fit_source": "training context x only",
            "overlapping_context_observations_repeated": True,
            "all_missing_features": [train.feature_names[i] for i in np.flatnonzero(all_missing)],
            "panel_fingerprint": train.metadata.get("panel_fingerprint"),
            "training_context_fingerprint": training_hash.hexdigest(),
        }
        return self

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.scale_ is None or self.mask_ is None:
            raise ValueError("Scaler must be fitted before transforming")
        values = np.asarray(x, dtype=np.float64)
        if values.shape[-1] != len(self.median_):
            raise ValueError("Feature dimension differs from training")
        if np.isinf(values).any():
            raise DataValidationError("Infinite features are invalid")
        result = (np.where(np.isnan(values), self.median_, values) - self.median_) / self.scale_
        if self.clip is not None:
            if self.clip <= 0:
                raise ValueError("clip must be positive or None")
            result[..., ~self.mask_] = np.clip(result[..., ~self.mask_], -self.clip, self.clip)
        return result.astype(np.float32)

    def transform(self, data: WindowData) -> WindowData:
        if data.feature_names != self.feature_names_:
            raise ValueError("Feature order/name differs from the training scaler")
        return replace(
            data,
            x=self.transform_x(data.x),
            future_x=self.transform_x(data.future_x),
            metadata={**data.metadata, "scaler_fit": dict(self.fit_metadata_)},
        )

    def fit_transform(self, train: WindowData) -> WindowData:
        return self.fit(train).transform(train)

    def to_dict(self) -> dict[str, Any]:
        if self.median_ is None or self.scale_ is None or self.mask_ is None:
            raise ValueError("Cannot serialize an unfitted scaler")
        return {
            "clip": self.clip,
            "eps": self.eps,
            "median": self.median_.tolist(),
            "scale": self.scale_.tolist(),
            "mask": self.mask_.tolist(),
            "feature_names": self.feature_names_,
            "fit_metadata": self.fit_metadata_,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> TrainOnlyRobustScaler:
        obj = cls(clip=state["clip"], eps=float(state["eps"]))
        obj.median_ = np.asarray(state["median"], dtype=np.float64)
        obj.scale_ = np.asarray(state["scale"], dtype=np.float64)
        obj.mask_ = np.asarray(state["mask"], dtype=bool)
        obj.feature_names_ = list(state["feature_names"])
        obj.fit_metadata_ = dict(state["fit_metadata"])
        if (
            obj.median_.shape != obj.scale_.shape
            or obj.mask_.shape != obj.scale_.shape
            or len(obj.feature_names_) != len(obj.scale_)
            or np.any(obj.scale_ <= 0)
            or not np.all(np.isfinite(obj.median_))
            or not np.all(np.isfinite(obj.scale_))
        ):
            raise DataValidationError("Invalid serialized scaler state")
        return obj
