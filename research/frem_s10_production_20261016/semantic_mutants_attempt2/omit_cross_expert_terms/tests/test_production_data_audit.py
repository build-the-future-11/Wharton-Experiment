"""Independent small source and clock admission tests; no empirical modeling."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from frem_s10.data.production import (
    CONVENTION,
    DESIGN_ID,
    LIMITATIONS,
    calendar_document,
    create_source_admission,
    validate_source_admission,
)
from frem_s10.errors import IntegrityError
from frem_s10.metadata import parse_utc_timestamp


def _put(root: Path, name: str, obj: object) -> dict[str, str]:
    payload = (json.dumps(obj, sort_keys=True) + "\n").encode()
    (root / name).write_bytes(payload)
    return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}


def _request(tmp_path: Path) -> Path:
    csv = (
        b",open,close,break_start,break_end\n"
        b"2005-01-03T00:00:00Z,2005-01-03T14:30:00Z,2005-01-03T21:00:00Z,,\n"
        b"2016-11-25T00:00:00Z,2016-11-25T14:30:00Z,2016-11-25T18:00:00Z,,\n"
        b"2017-01-03T00:00:00Z,not-read,not-read,,\n"
    )
    (tmp_path / "clock.csv").write_bytes(csv)
    raw_ref = {"path": "clock.csv", "sha256": hashlib.sha256(csv).hexdigest()}
    calendar = _put(tmp_path, "calendar.json", calendar_document(csv, raw_ref))
    sources: list[dict[str, str]] = []
    acquisitions: list[dict[str, str]] = []
    for asset in ("SPY", "TLT", "GLD"):
        ref = _put(
            tmp_path,
            f"{asset}.json",
            {
                "symbol": asset,
                "rows": [
                    {
                        "timestamp": parse_utc_timestamp("2005-01-03T14:30:00Z") // 1000000,
                        "open": 10,
                        "close": 11,
                    },
                    {
                        "timestamp": parse_utc_timestamp("2016-11-25T14:30:00Z") // 1000000,
                        "open": 12,
                        "close": 11,
                    },
                    {
                        "timestamp": parse_utc_timestamp("2017-01-03T14:30:00Z") // 1000000,
                        "open": "protected sentinel",
                        "close": None,
                    },
                ],
                "dividends": {
                    "a": {"date": parse_utc_timestamp("2016-11-25T14:30:00Z") // 1000000, "amount": 0.2}
                },
                "split_events": {},
            },
        )
        sources.append({**ref, "symbol": asset, "retrieved_at": "2026-10-16T00:00:00Z"})
        acquisitions.append(
            {"symbol": asset, "retained_sha256": ref["sha256"], "retrieved_at": "2026-10-16T00:00:00Z"}
        )
    acquisition = _put(tmp_path, "acquisition.json", {"sources": acquisitions})
    calendar_acquisition = _put(tmp_path, "calendar_acquisition.json", {"fixture": True})
    terms = _put(tmp_path, "terms.json", {"fixture": True})
    provenance = _put(
        tmp_path,
        "provenance.json",
        {
            "schema": "s10-source-provenance-v1",
            "calendar_authority": "COMMUNITY_MAINTAINED_XNYS_REFERENCE_NOT_EXCHANGE_CERTIFICATION",
            "calendar_acquisition": calendar_acquisition,
            "terms_review": terms,
            "pit_certified": False,
            "redistribution_rights_established": False,
        },
    )
    decision = _put(
        tmp_path,
        "decision.json",
        {
            "schema": "s10-source-convention-decision-v1",
            "design_id": DESIGN_ID,
            "evidence_scope": "GENERATED_ENGINEERING",
            "status": "APPROVED",
            "calendar_identity": calendar["sha256"],
            "convention": CONVENTION,
        },
    )
    _put(
        tmp_path,
        "request.json",
        {
            "schema": "s10-source-request-v1",
            "design_id": DESIGN_ID,
            "evidence_scope": "GENERATED_ENGINEERING",
            "sources": sources,
            "calendar": calendar,
            "provenance": provenance,
            "acquisition": acquisition,
            "decision": decision,
            "limitations": list(LIMITATIONS),
        },
    )
    return tmp_path / "request.json"


def test_exact_close_clock_and_scope_filtered_admission(tmp_path: Path) -> None:
    request = _request(tmp_path)
    ref = create_source_admission(
        request, artifact_root=tmp_path, source_root=tmp_path, output_name="attempt"
    )
    admitted = validate_source_admission(dict(ref), artifact_root=tmp_path, source_root=tmp_path)
    assert len(admitted.sessions) == 2
    assert admitted.sessions[1].close_us == parse_utc_timestamp("2016-11-25T18:00:00Z")
    assert admitted.evidence_scope == "GENERATED_ENGINEERING"
    audit = json.loads((tmp_path / "attempt/integrity.json").read_text())
    assert audit["excluded_date_counts"] == [1, 1, 1]
    assert audit["dividend_event_counts"] == [1, 1, 1]
    assert audit["model_execution"] is False
    with pytest.raises(FileExistsError):
        create_source_admission(request, artifact_root=tmp_path, source_root=tmp_path, output_name="attempt")


def test_calendar_hash_and_clock_tampering_rejected(tmp_path: Path) -> None:
    request = _request(tmp_path)
    calendar = json.loads((tmp_path / "calendar.json").read_text())
    calendar["sessions"][1]["close_us"] += 3600000000
    ref = _put(tmp_path, "calendar.json", calendar)
    doc = json.loads(request.read_text())
    doc["calendar"] = ref
    _put(tmp_path, "request.json", doc)
    with pytest.raises(IntegrityError, match="exact retained CSV"):
        create_source_admission(request, artifact_root=tmp_path, source_root=tmp_path, output_name="bad")
    failure = json.loads((tmp_path / "bad/failure.json").read_text())
    assert failure["status"] == "FAILED"
    assert not (tmp_path / "bad/admission.json").exists()


def test_source_mutation_and_false_empirical_scope_rejected(tmp_path: Path) -> None:
    request = _request(tmp_path)
    ref = create_source_admission(
        request, artifact_root=tmp_path, source_root=tmp_path, output_name="attempt"
    )
    source = tmp_path / "SPY.json"
    original = source.read_bytes()
    source.write_bytes(original + b" ")
    with pytest.raises(IntegrityError, match="SHA256"):
        validate_source_admission(dict(ref), artifact_root=tmp_path, source_root=tmp_path)
    source.write_bytes(original)
    doc = json.loads(request.read_text())
    doc["evidence_scope"] = "RETROSPECTIVE_DEVELOPMENT"
    _put(tmp_path, "request.json", doc)
    with pytest.raises(IntegrityError, match="frozen S10"):
        create_source_admission(
            request, artifact_root=tmp_path, source_root=tmp_path, output_name="false_real"
        )


def test_metadata_validation_imports_no_numerical_or_model_modules(tmp_path: Path) -> None:
    request = _request(tmp_path)
    ref = create_source_admission(
        request, artifact_root=tmp_path, source_root=tmp_path, output_name="attempt"
    )
    script = """import json,sys
from pathlib import Path
from frem_s10.data.production import validate_source_admission
validate_source_admission(json.loads(sys.argv[1]), artifact_root=Path(sys.argv[2]), source_root=Path(sys.argv[2]))
assert 'numpy' not in sys.modules
assert not any(x.startswith('frem_s10.model') or x.startswith('frem_s10.experiment') for x in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(ref), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
