"""Explicit fail-closed FREM-S11 research commands."""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from frem_s11.evidence import (
    atomic_json,
    package_identity,
    sha256_file,
    verify_admission,
    verify_archive,
    verify_data_admission,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="s11-research")
    commands = parser.add_subparsers(dest="command", required=True)
    admit = commands.add_parser("admit-host")
    admit.add_argument("--output", required=True)
    admit.add_argument("--execute", action="store_true")
    fetch = commands.add_parser("acquire")
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--release-month", required=True, type=int)
    fetch.add_argument("--admission", required=True)
    fetch.add_argument("--execute", action="store_true")
    data = commands.add_parser("admit-data")
    data.add_argument("--archive", required=True)
    data.add_argument("--archive-receipt", required=True)
    data.add_argument("--admission", required=True)
    data.add_argument("--release-month", required=True, type=int)
    data.add_argument("--cutoff", required=True, type=int)
    data.add_argument("--output", required=True)
    data.add_argument("--execute", action="store_true")
    run = commands.add_parser("development")
    run.add_argument("--archive", required=True)
    run.add_argument("--archive-receipt", required=True)
    run.add_argument("--data-admission", required=True)
    run.add_argument("--admission", required=True)
    run.add_argument("--release-month", required=True, type=int)
    run.add_argument("--cutoff", required=True, type=int)
    run.add_argument("--output", required=True)
    run.add_argument("--consumption", required=True)
    run.add_argument("--execute", action="store_true")
    revision = commands.add_parser("revision-stress")
    revision.add_argument("--older", required=True)
    revision.add_argument("--older-receipt", required=True)
    revision.add_argument("--older-release", required=True, type=int)
    revision.add_argument("--newer", required=True)
    revision.add_argument("--newer-receipt", required=True)
    revision.add_argument("--newer-release", required=True, type=int)
    revision.add_argument("--admission", required=True)
    revision.add_argument("--through", required=True, type=int)
    revision.add_argument("--output", required=True)
    revision.add_argument("--execute", action="store_true")
    return parser


def _protocol_bytes() -> bytes:
    return importlib.resources.files("frem_s11").joinpath("protocol_v2.json").read_bytes()


def _configure_threads() -> None:
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"


def _fixed_development(args: argparse.Namespace) -> None:
    if (args.release_month, args.cutoff) != (201608, 201607):
        raise ValueError("development is fixed to release 201608 through 201607")


def _admit_data(args: argparse.Namespace) -> dict[str, object]:
    from frem_s11.admission import execution_pressure_stable, sample_memory
    from frem_s11.data import ASSETS, parse_french_archive

    _fixed_development(args)
    admission_path = Path(args.admission)
    admission = verify_admission(admission_path)
    archive_path = Path(args.archive)
    archive_receipt_path = Path(args.archive_receipt)
    archive_receipt = verify_archive(
        archive_path,
        archive_receipt_path,
        release_month=args.release_month,
    )
    panel = parse_french_archive(
        archive_path.read_bytes(),
        release_month=args.release_month,
        end_inclusive=args.cutoff,
    )
    required_months = {199001, 200001, 200712, 200801, 201607}
    calendar_complete = required_months <= set(panel.months.tolist())
    post_sample = sample_memory()
    resource_stable = execution_pressure_stable(admission, post_sample)
    status = "PASS" if calendar_complete and resource_stable else "BLOCKED"
    reasons = []
    if not calendar_complete:
        reasons.append("FIXED_FOLD_OR_TRAINING_HISTORY_UNAVAILABLE")
    if not resource_stable:
        reasons.append("PAGEOUT_COUNTER_CHANGED_DURING_DATA_ADMISSION")
    result = {
        "schema": "frem-s11-data-admission-v1",
        "observed_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "reasons": reasons,
        "archive_sha256": archive_receipt["sha256"],
        "archive_receipt_sha256": sha256_file(archive_receipt_path),
        "host_admission_sha256": sha256_file(admission_path),
        "source_identity": package_identity()["sha256"],
        "release_month": args.release_month,
        "cutoff": args.cutoff,
        "first_month": int(panel.months[0]),
        "last_month": int(panel.months[-1]),
        "rows": len(panel.months),
        "asset_order": ASSETS,
        "post_sample": {**asdict(post_sample), "available_mib": post_sample.available_mib},
        "scientific_result": False,
        "numeric_values_disclosed": False,
    }
    atomic_json(Path(args.output), result)
    return result


def _development(args: argparse.Namespace) -> dict[str, Any]:
    from frem_s11.admission import sample_memory
    from frem_s11.artifacts import write_development_bundle
    from frem_s11.backtest import run_backtest
    from frem_s11.data import parse_french_archive

    _fixed_development(args)
    admission_path = Path(args.admission)
    admission = verify_admission(admission_path)
    archive_path = Path(args.archive)
    archive_receipt_path = Path(args.archive_receipt)
    archive_receipt = verify_archive(
        archive_path,
        archive_receipt_path,
        release_month=args.release_month,
    )
    data_admission_path = Path(args.data_admission)
    verify_data_admission(
        data_admission_path,
        archive_sha256=str(archive_receipt["sha256"]),
        release_month=args.release_month,
        cutoff=args.cutoff,
    )
    identity = package_identity()
    consumption_path = Path(args.consumption)
    consumption = {
        "schema": "frem-s11-development-consumption-v1",
        "consumed_utc": datetime.now(UTC).isoformat(),
        "protocol_id": "FREM-S11-P1",
        "protocol_sha256": hashlib.sha256(_protocol_bytes()).hexdigest(),
        "source_identity": identity["sha256"],
        "archive_sha256": archive_receipt["sha256"],
        "data_admission_sha256": sha256_file(data_admission_path),
        "host_admission_sha256": sha256_file(admission_path),
        "status": "CONSUMED_BEFORE_DEVELOPMENT_PARSE",
        "protected_access": False,
    }
    output = Path(args.output)
    if output.exists() or output.is_symlink() or consumption_path.exists() or consumption_path.is_symlink():
        raise FileExistsError("development evidence output already exists")
    atomic_json(consumption_path, consumption)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / "ATTEMPT.json",
        {**consumption, "schema": "frem-s11-development-attempt-v1", "complete": False},
    )
    try:
        panel = parse_french_archive(
            archive_path.read_bytes(),
            release_month=args.release_month,
            end_inclusive=args.cutoff,
        )
        folds = ((200001, 200712), (200801, 201607))
        results = tuple(run_backtest(panel, start_month=start, end_month=end) for start, end in folds)
        post_sample = sample_memory()
        provenance = {
            "archive_receipt": {
                "path": str(archive_receipt_path),
                "sha256": sha256_file(archive_receipt_path),
            },
            "data_admission": {
                "path": str(data_admission_path),
                "sha256": sha256_file(data_admission_path),
            },
            "host_admission": {
                "path": str(admission_path),
                "sha256": sha256_file(admission_path),
            },
            "consumption": {
                "path": str(consumption_path),
                "sha256": sha256_file(consumption_path),
            },
            "python": sys.version.split()[0],
            "interpreter": str(Path(sys.executable).resolve()),
        }
        return write_development_bundle(
            output / "bundle",
            panel,
            ("A", "B"),
            results,
            protocol_bytes=_protocol_bytes(),
            admission=admission,
            post_sample=post_sample,
            provenance=provenance,
        )
    except BaseException as exc:
        atomic_json(
            output / "FAILURE.json",
            {
                "schema": "frem-s11-development-failure-v1",
                "failed_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "message": str(exc),
                "consumption_sha256": sha256_file(consumption_path),
                "protected_access": False,
            },
        )
        raise


def _revision(args: argparse.Namespace) -> dict[str, object]:
    from frem_s11.admission import execution_pressure_stable, sample_memory
    from frem_s11.data import parse_french_archive
    from frem_s11.revision import compare_vintages

    allowed = {
        (200808, 201608, 200807),
        (201608, 202408, 201607),
    }
    if (args.older_release, args.newer_release, args.through) not in allowed:
        raise ValueError("revision comparison is outside the frozen common-cutoff matrix")
    admission = verify_admission(Path(args.admission))
    older_path, newer_path = Path(args.older), Path(args.newer)
    verify_archive(older_path, Path(args.older_receipt), release_month=args.older_release)
    verify_archive(newer_path, Path(args.newer_receipt), release_month=args.newer_release)
    older = parse_french_archive(
        older_path.read_bytes(), release_month=args.older_release, end_inclusive=args.through
    )
    newer = parse_french_archive(
        newer_path.read_bytes(), release_month=args.newer_release, end_inclusive=args.through
    )
    audit = compare_vintages(older, newer, through=args.through)
    post_sample = sample_memory()
    resource_stable = execution_pressure_stable(admission, post_sample)
    result = {
        "schema": "frem-s11-revision-audit-v2",
        "through": args.through,
        **asdict(audit),
        "resource_stable_after_execution": resource_stable,
        "post_sample": {**asdict(post_sample), "available_mib": post_sample.available_mib},
        "status": "COMPLETE" if resource_stable else "INVALID_RESOURCE_PRESSURE",
        "scientific_result": False,
    }
    atomic_json(Path(args.output), result)
    return result


def main() -> int:
    """Run an explicit no-overwrite command."""
    args = _parser().parse_args()
    if not args.execute:
        raise SystemExit("--execute is required after reviewing the frozen protocol")
    if args.command == "admit-host":
        from frem_s11.admission import run_admission

        result = run_admission(Path(args.output))
    elif args.command == "acquire":
        from frem_s11.acquisition import acquire

        verify_admission(Path(args.admission))
        result = acquire(args.url, Path(args.output), release_month=args.release_month)
    elif args.command == "admit-data":
        _configure_threads()
        result = _admit_data(args)
    elif args.command == "development":
        _configure_threads()
        result = _development(args)
    else:
        _configure_threads()
        result = _revision(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    status = str(result.get("scientific_status", result.get("status", "")))
    return 3 if status == "BLOCKED" or "INVALID" in status else 0


if __name__ == "__main__":
    raise SystemExit(main())
