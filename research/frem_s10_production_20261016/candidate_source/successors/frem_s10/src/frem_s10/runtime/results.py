"""Immutable numeric results and saved-output regeneration, without model fitting."""

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import cast

import numpy as np

from frem_s10.analysis.bootstrap import assess_precision, pilot_indices
from frem_s10.analysis.report import render_statistics
from frem_s10.arrays import FloatArray, IntArray
from frem_s10.data.precision import PrecisionInputs
from frem_s10.errors import IntegrityError
from frem_s10.experiment.analysis import analyze_campaign
from frem_s10.experiment.precision import PrecisionScores
from frem_s10.experiment.readback import load_campaign
from frem_s10.identities import JSONObject, JSONValue, checked_json, object_json
from frem_s10.metadata import strict_int
from frem_s10.runtime.artifacts import publish_json, read_verified_json
from frem_s10.runtime.claims import validate_report_claims


def array_document(array: FloatArray | IntArray, *, unit: str, axis: str) -> JSONObject:
    """Encode finite numeric values with their exact binary dtype and dimensions."""
    return {
        "schema": "s10-result-array-v1",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "unit": unit,
        "axis": axis,
        "values": checked_json(array.tolist()),
    }


def _array(
    value: JSONValue, *, unit: str, axis: str, dtype: str
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    document = object_json(value)
    if set(document) != {"schema", "dtype", "shape", "unit", "axis", "values"} or (
        document["schema"],
        document["dtype"],
        document["unit"],
        document["axis"],
    ) != ("s10-result-array-v1", dtype, unit, axis):
        raise IntegrityError("saved numerical array schema or units differ")
    _numeric_values(document["values"], integer=dtype == "<i8")
    array = np.asarray(document["values"], dtype=np.float64)
    if document["shape"] != list(array.shape) or not np.all(np.isfinite(array)):
        raise IntegrityError("saved numerical array shape or finiteness differs")
    return array


def _numeric_values(value: JSONValue, *, integer: bool) -> None:
    if isinstance(value, list):
        for item in value:
            _numeric_values(item, integer=integer)
    elif type(value) not in ((int,) if integer else (int, float)):
        raise IntegrityError("saved numeric values cannot be coerced from strings or booleans")


def publish_precision(directory: Path, scores: PrecisionScores, *, projected_a: int, projected_b: int) -> str:
    """Retain every pilot score and both independent fixed bootstrap streams."""
    streams: list[JSONValue] = []
    for block, indices in scores.bootstrap:
        name = f"bootstrap-{block}.json"
        digest = publish_json(
            directory, name, array_document(indices, unit="observation_index", axis="draw,date")
        )
        streams.append({"block": block, "path": name, "sha256": digest})
    value: JSONObject = {
        "schema": "s10-precision-evidence-v1",
        "inputs_id": scores.inputs_id,
        "evidence_scope": scores.evidence_scope,
        "origin_ids": list(scores.origin_ids),
        "forecast_ids": [list(row) for row in scores.forecast_ids],
        "final_state_id": scores.final_state_id,
        "weekly_issue_count": scores.weekly_issue_count,
        "projected_a": projected_a,
        "projected_b": projected_b,
        "cold_start_blocks": list(scores.cold_start_blocks),
        "local_fallback_reasons": list(scores.local_fallback_reasons),
        "candidate": array_document(scores.candidate, unit="standardized_total_return", axis="pilot_date"),
        "equal": array_document(scores.equal, unit="standardized_total_return", axis="pilot_date"),
        "global_stack": array_document(
            scores.global_stack, unit="standardized_total_return", axis="pilot_date"
        ),
        "result": _statistics(scores.result),
        "bootstrap": streams,
        "authorizes_campaign": False,
    }
    return publish_json(directory, "precision.json", value)


def audit_precision(directory: Path, identity: str) -> JSONObject:
    """Recompute the fixed gate from saved values and verify both saved RNG arrays."""
    value = object_json(read_verified_json(directory / "precision.json", identity, maximum_bytes=8_388_608))
    if value.get("schema") != "s10-precision-evidence-v1" or value.get("authorizes_campaign") is not False:
        raise IntegrityError("unrecognized or self-authorizing precision evidence")
    arrays = tuple(
        _array(value[name], unit="standardized_total_return", axis="pilot_date", dtype="<f8")
        for name in ("candidate", "equal", "global_stack")
    )
    a, b = (strict_int(value[name], name=name, minimum=24) for name in ("projected_a", "projected_b"))
    result = assess_precision(arrays[0], arrays[1], arrays[2], a, b)
    if _statistics(result) != value["result"]:
        raise IntegrityError("saved precision result differs from independent statistical regeneration")
    _audit_pilot_streams(directory, value["bootstrap"], len(arrays[0]))
    origins = value.get("origin_ids")
    if (
        not isinstance(origins, list)
        or len(origins) != len(arrays[0])
        or len(set(cast(list[str], origins))) != len(origins)
    ):
        raise IntegrityError("pilot dates are missing or duplicate")
    return {
        "schema": "s10-precision-audit-v1",
        "evidence_sha256": identity,
        "status": result.status,
        "evidence_scope": value["evidence_scope"],
        "pilot_blocks": result.pilot_blocks,
        "projected_blocks": result.projected_blocks,
        "result": value["result"],
    }


def _audit_pilot_streams(directory: Path, value: JSONValue, count: int) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise IntegrityError("both pilot bootstrap artifacts required")
    expected = pilot_indices(count)
    for raw, block in zip(value, (2, 4), strict=True):
        reference = object_json(raw)
        if reference.get("block") != block or reference.get("path") != f"bootstrap-{block}.json":
            raise IntegrityError("pilot bootstrap ordering differs")
        saved = read_verified_json(
            directory / f"bootstrap-{block}.json", str(reference.get("sha256")), maximum_bytes=67_108_864
        )
        observed = _array(saved, unit="observation_index", axis="draw,date", dtype="<i8")
        if not np.array_equal(observed, expected[block]):
            raise IntegrityError("pilot bootstrap differs from the frozen RNG schedule")


def audit_precision_archive(
    directory: Path, identity: str, archive: Path, archive_identity: str, inputs: PrecisionInputs
) -> JSONObject:
    """Recompute pilot ES from verified issued laws and the admitted pre-2010 targets."""
    from frem_s10.experiment.model_archive import decode_transition, verify_model_archive

    audit = audit_precision(directory, identity)
    saved = object_json(read_verified_json(directory / "precision.json", identity, maximum_bytes=8_388_608))
    manifest = verify_model_archive(archive, archive_identity)
    rows = manifest["files"]
    if not isinstance(rows, list) or manifest.get("variants") != ["primary"]:
        raise IntegrityError("precision requires its complete primary-only weekly archive")
    selected = set(inputs.pilot.origin_indices)
    records = tuple(
        record
        for record in inputs.records
        if record.origin_us >= 1199145600000000 and min(selected) <= record.anchor_index <= max(selected)
    )
    if (
        len(rows) != len(records)
        or saved["weekly_issue_count"] != len(records)
        or saved["inputs_id"] != inputs.record_id
    ):
        raise IntegrityError("saved precision origin schedule or admitted input identity differs")
    paired: list[tuple[float, float, float]] = []
    origins: list[str] = []
    forecasts: list[JSONValue] = []
    cold: list[JSONValue] = []
    fallback: list[JSONValue] = []
    last_state = ""
    for raw, record in zip(rows, records, strict=True):
        row = object_json(raw)
        item = object_json(
            read_verified_json(archive / str(row["name"]), str(row["sha256"]), maximum_bytes=33_554_432)
        )
        step = decode_transition(object_json(item["transition"]))
        request = step.batch.get("contextual_stack").request
        if (
            (request.record_id, request.anchor_index, request.origin_us, request.endpoint_us)
            != (record.record_id, record.anchor_index, record.origin_us, record.endpoint_us)
            or not np.array_equal(request.x, record.x)
            or step.state.transform.record_id != inputs.transform.record_id
        ):
            raise IntegrityError("archived issue differs from admitted causal pilot input")
        last_state = step.state.record_id
        if record.anchor_index in selected:
            paired.append(_archived_scores(step, record, inputs))
            origins.append(record.record_id)
            forecasts.append(
                [step.batch.get(name).record_id for name in ("contextual_stack", "equal", "global_stack")]
            )
            cold.append(step.batch.mixtures.global_certificate is None)
            fallback.append(step.batch.mixtures.fallback_reason)
    expected: JSONObject = {
        "origin_ids": list(origins),
        "forecast_ids": forecasts,
        "cold_start_blocks": cold,
        "local_fallback_reasons": fallback,
        "final_state_id": last_state,
        "projected_a": len(inputs.projected_a.origin_indices),
        "projected_b": len(inputs.projected_b.origin_indices),
    }
    if any(saved[name] != value for name, value in expected.items()):
        raise IntegrityError(
            "pilot identities, diagnostics or terminal state differ from archived transitions"
        )
    for column, name in enumerate(("candidate", "equal", "global_stack")):
        actual = _array(saved[name], unit="standardized_total_return", axis="pilot_date", dtype="<f8")
        if not np.array_equal(actual, np.asarray(paired, dtype=np.float64)[:, column]):
            raise IntegrityError("pilot scores differ from archived laws and admitted outcomes")
    return audit | {
        "schema": "s10-precision-archive-audit-v1",
        "archive_manifest_sha256": archive_identity,
        "inputs_id": inputs.record_id,
        "weekly_issue_count": len(records),
        "final_state_id": last_state,
    }


def _archived_scores(step: object, record: object, inputs: PrecisionInputs) -> tuple[float, float, float]:
    from frem_s10.domain import OriginRecord
    from frem_s10.math.scores import energy_score
    from frem_s10.model.records import ForecastTransition

    if not isinstance(step, ForecastTransition) or not isinstance(record, OriginRecord):
        raise IntegrityError("validated archive transition and admitted origin required")
    scores = []
    for name in ("contextual_stack", "equal", "global_stack"):
        law = step.batch.get(name).distribution
        scores.append(
            energy_score(
                inputs.transform.scale_return(law.r),
                law.probabilities,
                inputs.transform.scale_return(record.r),
            )
        )
    return scores[0], scores[1], scores[2]


def _statistics(value: object) -> JSONValue:
    """Serialize statistical records while keeping shared bootstrap arrays separate."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _statistics(cast(object, getattr(value, field.name)))
            for field in fields(value)
            if field.name != "bootstrap"
        }
    if isinstance(value, tuple):
        return [_statistics(item) for item in value]
    return checked_json(value)


def audit_campaign(directory: Path, identity: str, output: Path) -> str:
    """Read saved event evidence, regenerate every endpoint, and publish a fresh audit."""
    campaign = load_campaign(directory, identity)
    if not campaign.complete:
        raise IntegrityError("a failed or incomplete campaign cannot produce complete scientific endpoints")
    analysis = analyze_campaign(campaign)
    streams: list[JSONValue] = []
    for fold, block, indices in analysis.bootstrap:
        name = f"bootstrap-{fold}-{block}.json"
        digest = publish_json(
            output, name, array_document(indices, unit="observation_index", axis="draw,date")
        )
        streams.append({"fold": fold, "block": block, "path": name, "sha256": digest})
    summary: JSONObject = {
        "schema": "s10-campaign-audit-v1",
        "evidence_sha256": identity,
        "evidence_scope": campaign.evidence_scope,
        "status": "COMPLETE",
        "cells": 192,
        "primary": _statistics(analysis.primary),
        "ablations": _statistics(analysis.ablations),
        "bootstrap": streams,
        "source_authority_approved": False,
        "numerical_disposition": analysis.primary.disposition,
    }
    report = render_statistics(
        analysis.primary,
        analysis.ablations,
        evidence_sha256=identity,
        scope="GENERATED_ENGINEERING"
        if campaign.evidence_scope == "GENERATED_ENGINEERING"
        else "SOURCE_AUTHORITY_UNVERIFIED",
    )
    report_document: JSONObject = {
        "schema": "s10-rendered-report-v1",
        "markdown": report,
        "evidence_sha256": identity,
        "evidence_scope": campaign.evidence_scope,
    }
    report_id = publish_json(output, "report.json", report_document)
    summary["report"] = {"path": "report.json", "sha256": report_id}
    validate_report_claims(summary, report_document)
    return publish_json(output, "audit.json", summary)
