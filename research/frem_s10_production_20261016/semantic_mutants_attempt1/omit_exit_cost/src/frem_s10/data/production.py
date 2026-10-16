"""Hash-bound retrospective source admission, with import-light launch validation.

Admission audits source integrity only. Loaders are called inside a consumed,
resource-admitted worker; neither a receipt nor this module authorizes execution.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ..errors import ContractError, IntegrityError
from ..identities import checked_json
from ..metadata import ASSETS, from_utc_us, parse_utc_timestamp, scheduled_open_us, strict_int
from ..runtime.artifacts import publish_json
from .source_reader import SourceIdentity, require_fields, strict_json_loads

if TYPE_CHECKING:
    from ..experiment.replay import ReplayInputs
    from .precision import PrecisionInputs

DESIGN_ID = "FREM-S10-DESIGN-4-CL16"
CONVENTION = "SCHEDULED_EXCHANGE_REGULAR_SESSION_CLOSE_NOT_OBSERVED_EXECUTION"
REAL_HASHES = (
    "5079111b0a82fb75381ac1ef2585b20c7521ba9c5642929f6cf389dafb33adc8",
    "0a417fdb939bbdaed3f0faa8d75ea4a6909ab3dd977be032de1fe8d5bdb6c41d",
    "5befc3ccba30ed8d245198336c204596042683c9d662eb8d7c18ff1180ac9a7b",
)
LIMITATIONS = (
    "CURRENT_VINTAGE_RETROSPECTIVE_NOT_PIT_CERTIFIED",
    "SCHEDULED_CLOCK_NOT_OBSERVED_EXECUTION_OR_VENDOR_LATENCY",
    "DIVIDEND_EX_DATE_CASH_NOT_PAYMENT_DATE_SETTLEMENT",
    "REDISTRIBUTION_RIGHTS_NOT_ESTABLISHED",
    "SOURCE_ADMISSION_DOES_NOT_AUTHORIZE_COMPUTE_OR_CONFIRMATORY_CLAIMS",
)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ContractError("JSON object required")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("nonempty string required")
    return value


def _path(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or str(rel) != relative:
        raise IntegrityError("normalized relative artifact path required")
    current = root.resolve(strict=True)
    for part in rel.parts:
        current /= part
        if current.is_symlink():
            raise IntegrityError("artifact symlinks unsupported")
    if not current.resolve().is_relative_to(root.resolve(strict=True)):
        raise IntegrityError("artifact escapes root")
    return current


def _ref(value: object) -> dict[str, str]:
    obj = _object(value)
    require_fields(obj, {"path", "sha256"}, name="artifact reference")
    digest = _text(obj["sha256"])
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise IntegrityError("SHA256 reference required")
    return {"path": _text(obj["path"]), "sha256": digest}


def _read(root: Path, reference: object) -> bytes:
    ref = _ref(reference)
    path = _path(root, ref["path"])
    with path.open("rb") as stream:
        payload = stream.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != ref["sha256"]:
        raise IntegrityError("artifact size or SHA256 mismatch")
    return payload


def _write(root: Path, relative: str, document: dict[str, object]) -> dict[str, str]:
    path = _path(root, relative)
    digest = publish_json(path.parent, path.name, checked_json(document))
    return {"path": relative, "sha256": digest}


@dataclass(frozen=True, slots=True)
class SessionClock:
    """A retained calendar session with externally supplied opening and closing times."""

    day: date
    open_us: int
    close_us: int

    def __post_init__(self) -> None:
        """Reject impossible, out-of-scope or unsupported opening conventions."""
        strict_int(self.open_us, name="open_us")
        strict_int(self.close_us, name="close_us")
        if not date(2005, 1, 1) <= self.day < date(2017, 1, 1):
            raise ContractError("calendar escapes fixed pre-2017 source scope")
        if self.day.weekday() >= 5 or self.open_us != scheduled_open_us(self.day):
            raise IntegrityError("calendar opening differs from fixed regular-session convention")
        if self.close_us <= self.open_us + 2 or from_utc_us(self.close_us).date() != self.day:
            raise IntegrityError("calendar close must follow the same-session opening")


def calendar_document(csv_payload: bytes, source: dict[str, str]) -> dict[str, object]:
    """Extract exact 2005-2016 rows from the pinned maintained XNYS CSV, without inference."""
    if hashlib.sha256(csv_payload).hexdigest() != _ref(source)["sha256"]:
        raise IntegrityError("calendar CSV differs from provenance reference")
    reader = csv.DictReader(io.StringIO(csv_payload.decode("utf-8")))
    if reader.fieldnames != ["", "open", "close", "break_start", "break_end"]:
        raise ContractError("unsupported XNYS CSV schema")
    rows: list[dict[str, object]] = []
    for row in reader:
        day = from_utc_us(parse_utc_timestamp(row[""])).date()
        if not date(2005, 1, 1) <= day < date(2017, 1, 1):
            continue
        if row["break_start"] or row["break_end"]:
            raise ContractError("unexpected session break")
        clock = SessionClock(day, parse_utc_timestamp(row["open"]), parse_utc_timestamp(row["close"]))
        rows.append({"date": day.isoformat(), "open_us": clock.open_us, "close_us": clock.close_us})
    document: dict[str, object] = {
        "schema": "s10-session-clock-v1",
        "start": "2005-01-01",
        "end_exclusive": "2017-01-01",
        "convention": CONVENTION,
        "source": source,
        "sessions": rows,
    }
    _sessions(document)
    return document


def _sessions(document: dict[str, object]) -> tuple[SessionClock, ...]:
    require_fields(
        document, {"schema", "start", "end_exclusive", "convention", "source", "sessions"}, name="calendar"
    )
    if (document["schema"], document["start"], document["end_exclusive"], document["convention"]) != (
        "s10-session-clock-v1",
        "2005-01-01",
        "2017-01-01",
        CONVENTION,
    ):
        raise ContractError("unsupported calendar contract")
    rows = document["sessions"]
    if not isinstance(rows, list) or not rows:
        raise ContractError("nonempty session array required")
    clocks = []
    for item in rows:
        row = _object(item)
        require_fields(row, {"date", "open_us", "close_us"}, name="session clock")
        clocks.append(
            SessionClock(
                date.fromisoformat(_text(row["date"])),
                strict_int(row["open_us"], name="open_us"),
                strict_int(row["close_us"], name="close_us"),
            )
        )
    dates = tuple(clock.day for clock in clocks)
    if dates != tuple(sorted(set(dates))):
        raise IntegrityError("session sequence must be unique and ordered")
    return tuple(clocks)


@dataclass(frozen=True, slots=True)
class SourceAdmission:
    """Validated metadata; scope is an evidence limitation, never a run permission."""

    evidence_scope: str
    source_hashes: tuple[str, str, str]
    calendar_identity: str
    document_identity: str
    reference: dict[str, str]
    identities: tuple[SourceIdentity, ...]
    sessions: tuple[SessionClock, ...]


def _request(
    document: dict[str, object], root: Path
) -> tuple[str, tuple[SourceIdentity, ...], tuple[SessionClock, ...]]:
    require_fields(
        document,
        {
            "schema",
            "design_id",
            "evidence_scope",
            "sources",
            "calendar",
            "provenance",
            "acquisition",
            "decision",
            "limitations",
        },
        name="source request",
    )
    if document["schema"] != "s10-source-request-v1" or document["design_id"] != DESIGN_ID:
        raise ContractError("unsupported source admission request")
    scope = _text(document["evidence_scope"])
    if scope not in ("RETROSPECTIVE_DEVELOPMENT", "GENERATED_ENGINEERING"):
        raise ContractError("unsupported evidence scope")
    if document["limitations"] != list(LIMITATIONS):
        raise ContractError("all retrospective limitations must remain explicit")
    sources = document["sources"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise ContractError("exact three sources required")
    identities = tuple(_source_identity(item) for item in sources)
    if tuple(identity.symbol for identity in identities) != ASSETS:
        raise ContractError("ordered SPY/TLT/GLD source identities required")
    if scope == "RETROSPECTIVE_DEVELOPMENT" and tuple(i.sha256 for i in identities) != REAL_HASHES:
        raise IntegrityError("retrospective source hashes differ from frozen S10 sources")
    calendar = strict_json_loads(_read(root, document["calendar"]))
    clocks = _sessions(calendar)
    raw_csv = _read(root, calendar["source"])
    if calendar != calendar_document(raw_csv, _ref(calendar["source"])):
        raise IntegrityError("calendar differs from exact retained CSV extraction")
    _provenance(document, root, identities, scope)
    _decision(document, root, scope)
    return scope, identities, clocks


def _provenance(
    document: dict[str, object], root: Path, identities: tuple[SourceIdentity, ...], scope: str
) -> None:
    provenance = strict_json_loads(_read(root, document["provenance"]))
    require_fields(
        provenance,
        {
            "schema",
            "calendar_authority",
            "calendar_acquisition",
            "terms_review",
            "pit_certified",
            "redistribution_rights_established",
        },
        name="provenance",
    )
    if (
        provenance["schema"] != "s10-source-provenance-v1"
        or provenance["calendar_authority"]
        != "COMMUNITY_MAINTAINED_XNYS_REFERENCE_NOT_EXCHANGE_CERTIFICATION"
        or provenance["pit_certified"] is not False
        or provenance["redistribution_rights_established"] is not False
    ):
        raise IntegrityError("source provenance overstates its authority")
    for key in ("calendar_acquisition", "terms_review"):
        strict_json_loads(_read(root, provenance[key]))
    acquisition_ref = _ref(document["acquisition"])
    acquisition = strict_json_loads(_read(root, acquisition_ref))
    if scope == "RETROSPECTIVE_DEVELOPMENT" and acquisition_ref["sha256"] != (
        "12ce3b9a27d7c206cbd6a74a7ec3d12ba248649a26e74098d3750709ac37f562"
    ):
        raise IntegrityError("retrospective acquisition provenance differs from retained source receipt")
    sources = acquisition.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ContractError("acquisition must identify all three retained assets")
    for identity, item in zip(identities, sources, strict=True):
        source = _object(item)
        if (
            source.get("symbol") != identity.symbol
            or source.get("retained_sha256") != identity.sha256
            or parse_utc_timestamp(source.get("retrieved_at")) != identity.retrieved_at_us
        ):
            raise IntegrityError("source identity differs from acquisition provenance")


def _source_identity(value: object) -> SourceIdentity:
    obj = _object(value)
    require_fields(obj, {"path", "sha256", "symbol", "retrieved_at"}, name="source identity")
    return SourceIdentity(
        _text(obj["path"]),
        _text(obj["sha256"]),
        _text(obj["symbol"]),
        parse_utc_timestamp(obj["retrieved_at"]),
    )


def _decision(document: dict[str, object], root: Path, scope: str) -> None:
    decision = strict_json_loads(_read(root, document["decision"]))
    require_fields(
        decision,
        {"schema", "design_id", "evidence_scope", "status", "calendar_identity", "convention"},
        name="source convention decision",
    )
    expected = {
        "schema": "s10-source-convention-decision-v1",
        "design_id": DESIGN_ID,
        "evidence_scope": scope,
        "status": "APPROVED",
        "calendar_identity": _ref(document["calendar"])["sha256"],
        "convention": CONVENTION,
    }
    if decision != expected:
        raise IntegrityError("source convention decision is absent or mismatched")


def validate_source_admission(
    reference: dict[str, object], *, artifact_root: Path, source_root: Path
) -> SourceAdmission:
    """Validate bounded metadata and source hashes without decoding financial values."""
    ref = _ref(reference)
    doc = strict_json_loads(_read(artifact_root, ref))
    require_fields(doc, {"schema", "status", "request", "audit"}, name="source admission")
    if doc["schema"] != "s10-source-admission-v1" or doc["status"] != "PASS":
        raise IntegrityError("passing source integrity receipt required")
    request = strict_json_loads(_read(artifact_root, doc["request"]))
    scope, identities, clocks = _request(request, artifact_root)
    for identity in identities:
        _read(source_root, {"path": identity.logical_path, "sha256": identity.sha256})
    audit = strict_json_loads(_read(artifact_root, doc["audit"]))
    _validate_audit(audit, request, identities, clocks)
    hashes = (identities[0].sha256, identities[1].sha256, identities[2].sha256)
    return SourceAdmission(
        scope, hashes, _ref(request["calendar"])["sha256"], ref["sha256"], ref, identities, clocks
    )


def _validate_audit(
    audit: dict[str, object],
    request: dict[str, object],
    identities: tuple[SourceIdentity, ...],
    clocks: tuple[SessionClock, ...],
) -> None:
    require_fields(
        audit,
        {
            "schema",
            "status",
            "source_hashes",
            "calendar_identity",
            "session_count",
            "values_before",
            "excluded_date_counts",
            "dividend_event_counts",
            "model_execution",
            "evidence_scope",
        },
        name="source audit",
    )
    expected = {
        "schema": "s10-source-integrity-v1",
        "status": "PASS",
        "source_hashes": [i.sha256 for i in identities],
        "calendar_identity": _ref(request["calendar"])["sha256"],
        "session_count": len(clocks),
        "values_before": "2017-01-01",
        "model_execution": False,
        "evidence_scope": request["evidence_scope"],
    }
    if any(
        audit[key] != value or (key == "model_execution" and audit[key] is not False)
        for key, value in expected.items()
    ):
        raise IntegrityError("source integrity audit does not bind this request")
    for name in ("excluded_date_counts", "dividend_event_counts"):
        counts = audit[name]
        if not isinstance(counts, list) or len(counts) != 3:
            raise IntegrityError("three asset counts required")
        for count in counts:
            strict_int(count, name=name, minimum=0)


def create_source_admission(
    request_path: Path, *, artifact_root: Path, source_root: Path, output_name: str
) -> dict[str, str]:
    """Audit date-filtered prices/actions only, retaining an exclusive success or failure attempt."""
    destination = _path(artifact_root, output_name)
    destination.mkdir()  # Never replace any previous attempt or failed evidence.
    try:
        return _create(request_path, artifact_root, source_root, output_name)
    except Exception as exc:
        _write(
            artifact_root,
            f"{output_name}/failure.json",
            {
                "schema": "s10-source-failure-v1",
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "model_execution": False,
            },
        )
        raise


def _create(request_path: Path, artifact_root: Path, source_root: Path, output_name: str) -> dict[str, str]:
    from .admission import DataContract, admit_sources, calendar_identity
    from .source_reader import ScopedSourceReader

    try:
        relative = str(request_path.absolute().relative_to(artifact_root.resolve(strict=True)))
    except ValueError as exc:
        raise IntegrityError("source request must be beneath its artifact root") from exc
    with _path(artifact_root, relative).open("rb") as stream:
        payload = stream.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024:
        raise IntegrityError("source request exceeds byte ceiling")
    request = strict_json_loads(payload)
    scope, identities, clocks = _request(request, artifact_root)
    dates = tuple(clock.day for clock in clocks)
    contract = DataContract(date(2005, 1, 1), date(2017, 1, 1), dates, calendar_identity(dates))
    admitted = admit_sources(ScopedSourceReader(source_root, identities), identities, contract=contract)
    groups = admitted.panel.require_asset_event_ids()
    audit: dict[str, object] = {
        "schema": "s10-source-integrity-v1",
        "status": "PASS",
        "source_hashes": list(admitted.source_hashes),
        "calendar_identity": _ref(request["calendar"])["sha256"],
        "session_count": len(dates),
        "values_before": "2017-01-01",
        "excluded_date_counts": list(admitted.excluded_date_counts),
        "dividend_event_counts": [sum(row[a] is not None for row in groups) for a in range(3)],
        "model_execution": False,
        "evidence_scope": scope,
    }
    request_ref = _write(artifact_root, f"{output_name}/request.json", request)
    audit_ref = _write(artifact_root, f"{output_name}/integrity.json", audit)
    result = _write(
        artifact_root,
        f"{output_name}/admission.json",
        {"schema": "s10-source-admission-v1", "status": "PASS", "request": request_ref, "audit": audit_ref},
    )
    validate_source_admission(dict(result), artifact_root=artifact_root, source_root=source_root)
    return result


def load_precision_inputs(admission: SourceAdmission, *, source_root: Path) -> PrecisionInputs:
    """Load only pre-2010 numerical values inside an already consumed precision worker."""
    from .admission import DataContract, calendar_identity
    from .precision import prepare_precision_inputs
    from .source_reader import ScopedSourceReader

    dates = tuple(clock.day for clock in admission.sessions)
    pilot = tuple(day for day in dates if day < date(2010, 1, 1))
    contract = DataContract(date(2005, 1, 1), date(2010, 1, 1), pilot, calendar_identity(pilot))
    return prepare_precision_inputs(
        ScopedSourceReader(source_root, admission.identities),
        admission.identities,
        pilot_contract=contract,
        projected_sessions=dates,
        projected_calendar_id=calendar_identity(dates),
        evidence_scope=(
            "GENERATED_ENGINEERING"
            if admission.evidence_scope == "GENERATED_ENGINEERING"
            else "SOURCE_VIEW_WITHOUT_SCIENTIFIC_ADMISSION"
        ),
    )


def load_campaign_inputs(admission: SourceAdmission, *, source_root: Path) -> ReplayInputs:
    """Load pre-2017 values and exact supplied close clocks inside a consumed campaign worker."""
    from ..experiment.replay import SessionTiming, prepare_campaign
    from .admission import DataContract, admit_sources, calendar_identity
    from .source_reader import ScopedSourceReader

    dates = tuple(clock.day for clock in admission.sessions)
    contract = DataContract(date(2005, 1, 1), date(2017, 1, 1), dates, calendar_identity(dates))
    panel = admit_sources(
        ScopedSourceReader(source_root, admission.identities), admission.identities, contract=contract
    ).panel
    groups = panel.require_asset_event_ids()
    timing = tuple(
        SessionTiming(
            clock.open_us,
            clock.close_us,
            (
                (g[0],) if g[0] is not None else (),
                (g[1],) if g[1] is not None else (),
                (g[2],) if g[2] is not None else (),
            ),
        )
        for clock, g in zip(admission.sessions, groups, strict=True)
    )
    inputs = prepare_campaign(panel, timing)
    return (
        replace(inputs, evidence_scope="GENERATED_ENGINEERING")
        if (admission.evidence_scope == "GENERATED_ENGINEERING")
        else inputs
    )
