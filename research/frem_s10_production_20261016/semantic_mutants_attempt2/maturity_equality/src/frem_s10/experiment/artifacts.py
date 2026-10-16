"""Immutable streamed campaign artifacts with exact inventory verification."""

from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np

from frem_s10.arrays import FloatArray, frozen_f64
from frem_s10.domain import ForecastDistribution
from frem_s10.errors import ArtifactIOError, ContractError, IntegrityError
from frem_s10.identities import JSONObject, JSONValue, checked_json, object_json, scientific_digest
from frem_s10.metadata import strict_int
from frem_s10.runtime.artifacts import publish_json, read_verified_json

from .forecasts import ForecastFrame, ForecastTape
from .matrix import MODEL_VARIANTS, campaign_matrix
from .replay import CellFailure, CellResult, ReplayInputs, campaign_steps

_ARRAY_UNITS = {
    "SessionPanel": {"open": "USD_per_share", "close": "USD_per_share", "dividends": "USD_per_share"},
    "OriginRecord": {
        "x": "raw_context_dimension_specific",
        "g": "price_gross",
        "d": "cash_dividend_per_start_dollar",
        "h1_g": "price_gross",
        "h1_d": "cash_dividend_per_start_dollar",
    },
    "Transform": {
        "feature_mean": "raw_context_dimension_specific",
        "feature_scale": "raw_context_dimension_specific",
        "score_scale": "total_simple_return",
    },
    "ForecastDistribution": {
        "g": "price_gross",
        "d": "cash_dividend_per_start_dollar",
        "probabilities": "probability",
    },
    "LedgerState": {"shares": "shares", "marks": "USD_per_share", "price_pnl": "USD", "dividends": "USD"},
    "LedgerEvent": {
        "price_pnl": "USD",
        "dividends": "USD",
        "shares_before": "shares",
        "shares_after": "shares",
    },
    "Action": {"weights": "weight"},
    "FeeSolution": {"risky_dollars": "USD", "target_weights": "weight"},
    "ForecastScore": {
        "marginal_crps": "standardized_total_return",
        "loss_probabilities": "probability",
        "loss_outcomes": "binary_loss_event",
        "raw_marginal_crps": "total_simple_return",
        "raw_mean_squared_errors": "squared_total_simple_return",
        "interval_lower_raw": "total_simple_return",
        "interval_upper_raw": "total_simple_return",
        "interval_coverage": "binary_inclusive_coverage",
        "interval_outcome_boundary_ties": "binary_outcome_equals_interval_boundary",
    },
}


def _array(value: object, owner: str, name: str) -> JSONObject:
    if not isinstance(value, np.ndarray) or value.dtype not in (np.dtype("float64"), np.dtype("int64")):
        raise ContractError("artifact arrays require explicit binary64 or int64 storage")
    unit = _ARRAY_UNITS.get(owner, {}).get(name)
    if unit is None:
        raise ContractError(f"artifact array has no declared unit: {owner}.{name}")
    return {
        "array_schema": "s10-typed-array-v1",
        "dtype": value.dtype.str,
        "shape": [int(length) for length in value.shape],
        "unit": unit,
        "context": f"{owner}.{name}",
        "values": checked_json(value.tolist()),
    }


def _encode(value: object, *, owner: str = "", name: str = "") -> JSONValue:
    if isinstance(value, np.ndarray):
        return _array(value, owner, name)
    if is_dataclass(value) and not isinstance(value, type):
        record_type = type(value).__name__
        content = {
            field.name: _encode(cast(object, getattr(value, field.name)), owner=record_type, name=field.name)
            for field in fields(value)
        }
        return {"record_type": record_type, "fields": content}
    if type(value) is date:
        return {"record_type": "SessionDate", "iso_date": value.isoformat()}
    if isinstance(value, tuple):
        return [_encode(item, owner=owner, name=name) for item in value]
    if isinstance(value, frozenset) and all(isinstance(item, str) for item in value):
        return sorted(value)
    return checked_json(value)


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    """A verified inventory identity; completion is distinct from scientific support."""

    manifest_sha256: str
    completed_cells: int
    complete: bool
    evidence_scope: str


def _header(inputs: ReplayInputs, tape: ForecastTape, experiment_id: str) -> JSONObject:
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ContractError("explicit experiment attempt identity required")
    if inputs.evidence_scope != tape.evidence_scope or inputs.transform.record_id != tape.transform_id:
        raise IntegrityError("publication input and forecast scope or transform mismatch")
    return {
        "schema": "s10-campaign-header-v1",
        "experiment_id": experiment_id,
        "source_identity": scientific_digest(_encode(inputs)),
        "source_ids": list(inputs.panel.source_ids),
        "tape_identity": tape.record_id,
        "evidence_scope": inputs.evidence_scope,
        "expected_cells": [cell.cell_id for cell in campaign_matrix()],
        "forecast_frame_identities": [frame.record_id for frame in tape.frames],
        "interval_axis": [0.50, 0.80, 0.95],
        "asset_axis": ["SPY", "TLT", "GLD"],
        "loss_axis": ["SPY", "TLT", "GLD", "equal_weight_basket"],
        "scientific_support_claimed": False,
    }


def _cell_payload(step: CellResult | CellFailure, header: JSONObject) -> JSONObject:
    if isinstance(step, CellResult) and step.final_state.phase != "FINAL":
        raise IntegrityError("a completed cell publication requires actual final liquidation")
    ledger = step if isinstance(step, CellResult) else step.partial
    if ledger is not None and ledger.final_state.cell_id != ledger.spec.cell_id:
        raise IntegrityError("published ledger state belongs to a different cell")
    if ledger is not None and ledger.events and ledger.events[-1].after_id != ledger.final_state.record_id:
        raise IntegrityError("published last event does not identify its retained ledger state")
    return {
        "schema": "s10-campaign-cell-v1",
        "experiment_id": header["experiment_id"],
        "source_identity": header["source_identity"],
        "tape_identity": header["tape_identity"],
        "evidence_scope": header["evidence_scope"],
        "scientific_support_claimed": False,
        "cell_id": step.spec.cell_id if isinstance(step, CellResult) else step.cell_id,
        "status": "COMPLETE" if isinstance(step, CellResult) else "FAILED",
        "last_state_id": None if ledger is None else ledger.final_state.record_id,
        "payload": _encode(step),
    }


def _publish_steps(
    directory: Path, steps: Iterable[CellResult | CellFailure], header: JSONObject, inventory: list[JSONValue]
) -> tuple[int, str | None]:
    expected = tuple(cell.cell_id for cell in campaign_matrix())
    completed = 0
    failure: str | None = None
    for ordinal, step in enumerate(steps):
        cell_id = step.spec.cell_id if isinstance(step, CellResult) else step.cell_id
        if ordinal >= len(expected) or cell_id != expected[ordinal]:
            raise IntegrityError("published cell is unexpected, duplicated or out of matrix order")
        name = f"cell-{ordinal:03}.json"
        digest = publish_json(directory, name, _cell_payload(step, header))
        inventory.append({"name": name, "sha256": digest, "cell_id": cell_id})
        if isinstance(step, CellFailure):
            failure = cell_id
            break
        completed += 1
    if failure is None and completed != len(expected):
        failure = expected[completed]
        step = CellFailure(failure, "INCOMPLETE_MATRIX", "cell stream ended before all declared cells")
        name = f"cell-{completed:03}.json"
        digest = publish_json(directory, name, _cell_payload(step, header))
        inventory.append({"name": name, "sha256": digest, "cell_id": failure})
    return completed, failure


def _publish_tape(directory: Path, tape: ForecastTape) -> tuple[JSONObject, tuple[JSONObject, ...]]:
    rows: list[JSONObject] = []
    for ordinal, frame in enumerate(tape.frames):
        name = f"forecast-frame-{ordinal:04}.json"
        payload: JSONObject = {
            "schema": "s10-forecast-frame-artifact-v1",
            "frame_identity": frame.record_id,
            "evidence_scope": tape.evidence_scope,
            "payload": _encode(frame),
        }
        digest = publish_json(directory, name, payload)
        rows.append(
            {
                "name": name,
                "sha256": digest,
                "frame_identity": frame.record_id,
                "anchor_index": frame.anchor_index,
                "variant": frame.variant,
            }
        )
    index: JSONObject = {
        "schema": "s10-forecast-index-v1",
        "tape_identity": tape.record_id,
        "evidence_scope": tape.evidence_scope,
        "transform_id": tape.transform_id,
        "weekly_origins": tape.weekly_origins,
        "final_state_ids": _encode(tape.final_state_ids),
        "frames": list(rows),
    }
    digest = publish_json(directory, "forecasts.json", index)
    return {"name": "forecasts.json", "sha256": digest}, tuple(rows)


def publish_campaign(
    directory: Path,
    inputs: ReplayInputs,
    tape: ForecastTape,
    *,
    experiment_id: str,
    steps: Iterable[CellResult | CellFailure] | None = None,
) -> PublicationReceipt:
    """Publish every result before advancing the next cell, preserving failed prefixes.

    The caller supplies an empty owned artifact directory and has already
    passed execution admission. This worker performs no source reads or
    admission bypass. Optional provided steps support generated integration
    and composition; exact order/completion validation still applies.
    """
    if not directory.is_dir() or any(directory.iterdir()):
        raise ArtifactIOError("campaign publication requires an existing empty directory")
    header = _header(inputs, tape, experiment_id)
    header_digest = publish_json(directory, "header.json", header)
    tape_row, frame_rows = _publish_tape(directory, tape)
    inventory: list[JSONValue] = [{"name": "header.json", "sha256": header_digest}, tape_row, *frame_rows]
    completed, failure = _publish_steps(
        directory, campaign_steps(inputs, tape) if steps is None else steps, header, inventory
    )
    manifest: JSONObject = {
        "schema": "s10-campaign-inventory-v1",
        "evidence_scope": inputs.evidence_scope,
        "experiment_id": experiment_id,
        "complete": failure is None and completed == 192,
        "completed_cells": completed,
        "forecast_frames": len(frame_rows),
        "failure_cell_id": failure,
        "scientific_support_claimed": False,
        "files": inventory,
    }
    digest = publish_json(directory, "manifest.json", manifest)
    verify_publication(directory, digest)
    return PublicationReceipt(digest, completed, bool(manifest["complete"]), inputs.evidence_scope)


def _inventory(manifest: JSONObject) -> tuple[JSONObject, ...]:
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) < 3:
        raise IntegrityError("campaign inventory requires header, forecasts and cell evidence")
    rows = tuple(object_json(row) for row in files)
    if any(not isinstance(row.get("name"), str) or not isinstance(row.get("sha256"), str) for row in rows):
        raise IntegrityError("inventory filenames and hashes must be explicit strings")
    names = tuple(row["name"] for row in rows)
    if len(set(cast(tuple[str, ...], names))) != len(names) or names[:2] != ("header.json", "forecasts.json"):
        raise IntegrityError("inventory header order or file uniqueness failed")
    return rows


def _verify_cell(payload: JSONObject, header: JSONObject, expected_id: str, complete: bool) -> None:
    if payload.get("cell_id") != expected_id or payload.get("status") != (
        "COMPLETE" if complete else "FAILED"
    ):
        raise IntegrityError("published cell identity or completion status is inconsistent")
    for field in ("experiment_id", "source_identity", "tape_identity", "evidence_scope"):
        if payload.get(field) != header.get(field):
            raise IntegrityError("published cell provenance differs from campaign header")
    if payload.get("scientific_support_claimed") is not False:
        raise IntegrityError("artifact completion cannot assert scientific support")
    record = object_json(payload.get("payload"))
    expected_type = "CellResult" if complete else "CellFailure"
    if record.get("record_type") != expected_type:
        raise IntegrityError("artifact completion differs from its retained record type")
    if complete:
        values = object_json(record.get("fields"))
        state = object_json(object_json(values.get("final_state")).get("fields"))
        if state.get("phase") != "FINAL" or state.get("cell_id") != expected_id:
            raise IntegrityError("completed artifact lacks its actual terminal cell state")


def _verify_header(header: JSONObject, manifest: JSONObject) -> None:
    if (
        header.get("schema") != "s10-campaign-header-v1"
        or manifest.get("schema") != "s10-campaign-inventory-v1"
    ):
        raise IntegrityError("unsupported campaign artifact schema")
    if header.get("scientific_support_claimed") is not False or header.get("evidence_scope") not in (
        "KERNEL_REPLAY",
        "GENERATED_ENGINEERING",
    ):
        raise IntegrityError("campaign evidence scope or support claim is invalid")
    if _text(header.get("experiment_id")) != manifest.get("experiment_id"):
        raise IntegrityError("manifest experiment identity differs from campaign header")
    axes: JSONObject = {
        "interval_axis": [0.50, 0.80, 0.95],
        "asset_axis": ["SPY", "TLT", "GLD"],
        "loss_axis": ["SPY", "TLT", "GLD", "equal_weight_basket"],
    }
    if any(header.get(name) != value for name, value in axes.items()):
        raise IntegrityError("campaign header axes differ from the frozen scientific definitions")


def _read_row(directory: Path, row: JSONObject) -> JSONObject:
    return object_json(
        read_verified_json(
            directory / cast(str, row["name"]), cast(str, row["sha256"]), maximum_bytes=134217728
        )
    )


def _text(value: JSONValue) -> str:
    if not isinstance(value, str) or not value:
        raise IntegrityError("nonempty artifact text required")
    return value


def _law_array(value: JSONValue, name: str) -> FloatArray:
    encoded = object_json(value)
    if set(encoded) != {"array_schema", "dtype", "shape", "unit", "context", "values"} or (
        encoded.get("array_schema"),
        encoded.get("dtype"),
        encoded.get("unit"),
        encoded.get("context"),
    ) != (
        "s10-typed-array-v1",
        np.dtype("float64").str,
        _ARRAY_UNITS["ForecastDistribution"][name],
        f"ForecastDistribution.{name}",
    ):
        raise IntegrityError("forecast array dtype, unit or context mismatch")
    _numeric_array_value(encoded.get("values"))
    values = np.asarray(encoded.get("values"), dtype=np.float64)
    if encoded.get("shape") != list(values.shape):
        raise IntegrityError("forecast array shape differs from its declared shape")
    return frozen_f64(values, shape=(None,) if name == "probabilities" else (None, 3), name=name)


def _numeric_array_value(value: JSONValue) -> None:
    if isinstance(value, list):
        for item in value:
            _numeric_array_value(item)
    elif type(value) not in (int, float):
        raise IntegrityError("forecast array contains nonnumeric or boolean values")


def _record_fields(
    record: JSONObject, record_type: type[ForecastFrame] | type[ForecastDistribution]
) -> JSONObject:
    content = object_json(record.get("fields"))
    if (
        set(record) != {"record_type", "fields"}
        or record.get("record_type") != record_type.__name__
        or set(content) != {field.name for field in fields(record_type)}
    ):
        raise IntegrityError("forecast record has missing or unexpected scientific fields")
    return content


def _decode_frame(payload: JSONObject) -> ForecastFrame:
    record = object_json(payload.get("payload"))
    if record.get("record_type") != "ForecastFrame":
        raise IntegrityError("forecast artifact is not a frame")
    values = _record_fields(record, ForecastFrame)
    raw_laws = values.get("laws")
    if not isinstance(raw_laws, list):
        raise IntegrityError("frame laws must be an ordered list")
    laws: list[tuple[str, ForecastDistribution]] = []
    for item in raw_laws:
        if not isinstance(item, list) or len(item) != 2:
            raise IntegrityError("frame law entry must contain arm and finite distribution")
        encoded = object_json(item[1])
        if encoded.get("record_type") != "ForecastDistribution":
            raise IntegrityError("frame law has an unsupported record type")
        law = _record_fields(encoded, ForecastDistribution)
        laws.append(
            (
                _text(item[0]),
                ForecastDistribution(
                    _text(law.get("record_id")),
                    _law_array(law.get("g"), "g"),
                    _law_array(law.get("d"), "d"),
                    _law_array(law.get("probabilities"), "probabilities"),
                ),
            )
        )
    return ForecastFrame(
        strict_int(values.get("anchor_index"), name="frame anchor"),
        _text(values.get("origin_id")),
        strict_int(values.get("decision_us"), name="frame decision"),
        _text(values.get("variant")),
        tuple(laws),
        _text(values.get("state_before_id")),
        _text(values.get("state_after_id")),
    )


def _verify_frames(
    directory: Path, rows: tuple[JSONObject, ...], manifest: JSONObject, header: JSONObject
) -> int:
    count = manifest.get("forecast_frames")
    if type(count) is not int or not 1 <= count <= 6000 or len(rows) < count + 3:
        raise IntegrityError("invalid forecast-frame inventory count")
    index = _read_row(directory, rows[1])
    if index.get("schema") != "s10-forecast-index-v1" or index.get("frames") != list(rows[2 : 2 + count]):
        raise IntegrityError("forecast frame index differs from inventory")
    if header.get("forecast_frame_identities") != [row.get("frame_identity") for row in rows[2 : 2 + count]]:
        raise IntegrityError("forecast frames differ from the complete declared tape")
    for name in ("tape_identity", "evidence_scope"):
        if index.get(name) != header.get(name):
            raise IntegrityError("forecast index provenance differs from header")
    frame_axes: list[tuple[int, str, str, int]] = []
    for ordinal, row in enumerate(rows[2 : 2 + count]):
        if row.get("name") != f"forecast-frame-{ordinal:04}.json":
            raise IntegrityError("forecast frames are missing or out of order")
        frame_axes.append(_verify_frame(_read_row(directory, row), row, header))
    _verify_frame_axes(tuple(frame_axes))
    _verify_tape_identity(index, rows[2 : 2 + count])
    return count


def _verify_frame(frame: JSONObject, row: JSONObject, header: JSONObject) -> tuple[int, str, str, int]:
    if (
        frame.get("schema") != "s10-forecast-frame-artifact-v1"
        or frame.get("frame_identity") != row.get("frame_identity")
        or frame.get("evidence_scope") != header.get("evidence_scope")
    ):
        raise IntegrityError("forecast frame identity or scope differs from its index")
    decoded = _decode_frame(frame)
    if (decoded.anchor_index, decoded.variant) != (row.get("anchor_index"), row.get("variant")):
        raise IntegrityError("forecast frame axes differ from index")
    if decoded.record_id != row.get("frame_identity"):
        raise IntegrityError("forecast numerical content differs from its binary component identity")
    return decoded.anchor_index, decoded.variant, decoded.origin_id, decoded.decision_us


def _verify_frame_axes(frame_axes: tuple[tuple[int, str, str, int], ...]) -> None:
    keys = tuple((anchor, MODEL_VARIANTS.index(variant)) for anchor, variant, _, _ in frame_axes)
    if keys != tuple(sorted(set(keys))):
        raise IntegrityError("saved forecast tape keys are duplicate or out of order")
    for anchor in dict.fromkeys(item[0] for item in frame_axes):
        selected = tuple(item for item in frame_axes if item[0] == anchor)
        if tuple(item[1] for item in selected) != MODEL_VARIANTS:
            raise IntegrityError("saved forecast tape origin lacks its exact six variants")
        if len({(item[2], item[3]) for item in selected}) != 1:
            raise IntegrityError("saved forecast variants disagree about their common origin")


def _verify_tape_identity(index: JSONObject, rows: tuple[JSONObject, ...]) -> None:
    """Rebuild the small tape hash envelope without retaining every numerical frame."""
    weekly = strict_int(index.get("weekly_origins"), name="saved weekly origins", minimum=1)
    final = index.get("final_state_ids")
    if weekly > 1000 or not isinstance(final, list) or len(final) != len(MODEL_VARIANTS):
        raise IntegrityError("saved forecast tape budget or final state registry is invalid")
    for item, variant in zip(final, MODEL_VARIANTS, strict=True):
        if not isinstance(item, list) or len(item) != 2 or item[0] != variant:
            raise IntegrityError("saved forecast tape final state order is invalid")
        _text(item[1])
    identity = scientific_digest(
        {
            "schema": "s10-forecast-tape-v2",
            "transform": _text(index.get("transform_id")),
            "scope": _text(index.get("evidence_scope")),
            "weekly_origins": weekly,
            "final_states": final,
            "frames": [_text(row.get("frame_identity")) for row in rows],
        }
    )
    if identity != index.get("tape_identity"):
        raise IntegrityError("saved forecast tape metadata differs from its semantic identity")


def verify_publication(directory: Path, manifest_sha256: str) -> JSONObject:
    """Re-read every hash and reject missing, extra, misordered or mislabeled cells."""
    manifest = object_json(read_verified_json(directory / "manifest.json", manifest_sha256))
    rows = _inventory(manifest)
    names = {cast(str, row["name"]) for row in rows} | {"manifest.json"}
    if {path.name for path in directory.iterdir()} != names:
        raise IntegrityError("artifact directory has missing or unexpected files")
    header = _read_row(directory, rows[0])
    _verify_header(header, manifest)
    frame_count = _verify_frames(directory, rows, manifest, header)
    expected = tuple(cell.cell_id for cell in campaign_matrix())
    if header.get("expected_cells") != list(expected) or manifest.get("evidence_scope") != header.get(
        "evidence_scope"
    ):
        raise IntegrityError("inventory does not identify the exact matrix and evidence scope")
    completed = manifest.get("completed_cells")
    if type(completed) is not int or not 0 <= completed <= 192:
        raise IntegrityError("invalid completed cell count")
    complete = manifest.get("complete")
    if type(complete) is not bool or complete != (completed == 192):
        raise IntegrityError("completion does not match the full192 cell count")
    if len(rows) != 2 + frame_count + completed + (0 if complete else 1):
        raise IntegrityError("inventory cell count does not match success and failure records")
    for ordinal, row in enumerate(rows[2 + frame_count :]):
        if row.get("name") != f"cell-{ordinal:03}.json" or row.get("cell_id") != expected[ordinal]:
            raise IntegrityError("inventory cell order differs from the exact matrix")
        _verify_cell(_read_row(directory, row), header, expected[ordinal], ordinal < completed)
    failure = None if complete else expected[completed]
    if manifest.get("failure_cell_id") != failure or manifest.get("scientific_support_claimed") is not False:
        raise IntegrityError("inventory failure or evidence claim is inconsistent")
    return manifest
