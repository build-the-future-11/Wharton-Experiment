"""Import-light audit of actual consumed attempts and sampled resource evidence."""

import hashlib
from pathlib import Path

from frem_s10.errors import IntegrityError
from frem_s10.identities import JSONObject, object_json, parse_json
from frem_s10.metadata import finite_float, parse_utc_timestamp, strict_int
from frem_s10.runtime.declarations import _read, _reference
from frem_s10.runtime.journal import inspect_journal


def audit_execution(result: JSONObject, *, artifact_root: Path) -> JSONObject:
    """Verify complete sampled supervision; this is not a hard kernel resource guarantee."""
    supervisor, _ = _read(result["supervisor"], artifact_root)
    marker, _ = _read(result["consumption"], artifact_root)
    worker, _ = _read(result["worker_evidence"], artifact_root)
    if result.get("operation") not in ("prepare-precision", "development-campaign"):
        raise IntegrityError("unknown supervised scientific operation")
    operation = "precision" if result.get("operation") == "prepare-precision" else "campaign"
    if (
        supervisor.get("state"),
        supervisor.get("operation"),
        supervisor.get("freeze_id"),
        supervisor.get("worker_exit_code"),
        supervisor.get("consumed"),
    ) != ("COMPLETE", operation, result.get("declaration_sha256"), 0, True):
        raise IntegrityError("supervised operation was not completed successfully")
    if (
        marker.get("schema_version"),
        marker.get("operation"),
        marker.get("freeze_id"),
        marker.get("state"),
        marker.get("resume_permitted"),
        marker.get("attempt_id"),
    ) != (1, operation, result.get("declaration_sha256"), "CONSUMED", False, supervisor.get("attempt_id")):
        raise IntegrityError("actual persistent consumed identity differs from supervision")
    worker_path, _ = _reference(result["worker_evidence"], artifact_root)
    if marker.get("output_path") != str(worker_path.parent):
        raise IntegrityError("consumption does not own the saved worker output")
    if (
        worker.get("declaration_sha256"),
        worker.get("operation"),
        worker.get("evidence_scope"),
        worker.get("scientific_disposition"),
    ) != (
        result.get("declaration_sha256"),
        result.get("operation"),
        result.get("evidence_scope"),
        result.get("scientific_disposition"),
    ):
        raise IntegrityError("worker result differs from the exact supervised operation")
    wall = 600 if operation == "precision" else 3600
    elapsed = finite_float(supervisor.get("elapsed_seconds"), name="elapsed", strictly_positive=True)
    rss = finite_float(supervisor.get("maximum_observed_owned_rss_mib"), name="RSS", strictly_positive=True)
    if (
        elapsed > wall
        or rss > 256
        or supervisor.get("wall_limit_seconds") != wall
        or supervisor.get("rss_limit_mib") != 256
    ):
        raise IntegrityError("successful supervision exceeds the fixed elapsed or RSS ceiling")
    _audit_resources(result, supervisor, artifact_root)
    return worker


def _audit_resources(result: JSONObject, supervisor: JSONObject, root: Path) -> None:
    host, _ = _read(result["preflight"], root)
    samples = host.get("samples")
    if host.get("status") != "PASS" or not isinstance(samples, list) or len(samples) != 3:
        raise IntegrityError("three passing preflight observations required")
    rows = tuple(object_json(sample) for sample in samples)
    pageouts = rows[0].get("pageouts")
    previous_clock: int | None = None
    for row in rows:
        clock = parse_utc_timestamp(row.get("timestamp"))
        if previous_clock is not None and clock - previous_clock < 1_000_000:
            raise IntegrityError("preflight resource samples are less than one second apart")
        previous_clock = clock
        available = (
            (
                strict_int(row.get("free_pages"), name="free pages", minimum=0)
                + strict_int(row.get("speculative_pages"), name="speculative pages", minimum=0)
            )
            * strict_int(row.get("page_size_bytes"), name="page size", minimum=1)
            / 2**20
        )
        if available != row.get("available_mib") or available < 512 or row.get("pageouts") != pageouts:
            raise IntegrityError("preflight memory floor or pageout condition failed")
    path, identity = _reference(result["resources"], root)
    count, _ = inspect_journal(path)
    if count != strict_int(supervisor.get("resource_observations"), name="resource observations", minimum=1):
        raise IntegrityError("resource observation count differs from controller completion")
    digest = hashlib.sha256()
    previous = -1.0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = object_json(object_json(parse_json(line))["event"])
            elapsed = finite_float(row.get("elapsed_seconds"), name="observed elapsed")
            if elapsed < previous or elapsed > finite_float(
                supervisor.get("elapsed_seconds"), name="elapsed"
            ):
                raise IntegrityError("resource elapsed observations regress or exceed completion")
            previous = elapsed
            _resource_row(row, supervisor, pageouts)
    if digest.hexdigest() != identity:
        raise IntegrityError("resource journal differs from its retained identity")
    timeline, expected = _reference(result["timeline"], root)
    inspect_journal(timeline)
    with timeline.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise IntegrityError("lifecycle journal differs from its retained identity")


def _resource_row(row: JSONObject, supervisor: JSONObject, pageouts: object) -> None:
    if (
        finite_float(row.get("free_plus_speculative_mib"), name="host MiB") < 512
        or row.get("pageouts") != pageouts
    ):
        raise IntegrityError("host resource condition failed during operation")
    rss = finite_float(row.get("owned_rss_mib"), name="observed RSS")
    if rss > 256 or rss > finite_float(supervisor.get("maximum_observed_owned_rss_mib"), name="reported RSS"):
        raise IntegrityError("resource RSS observation exceeds retained peak or ceiling")
    if strict_int(row.get("output_bytes"), name="observed output bytes", minimum=0) > strict_int(
        supervisor.get("output_limit_bytes"), name="output limit", minimum=1
    ):
        raise IntegrityError("observed output exceeded declared storage limit")
    processes = row.get("owned_processes")
    if not isinstance(processes, list) or not processes:
        raise IntegrityError("resource sample has no owned process identities")
