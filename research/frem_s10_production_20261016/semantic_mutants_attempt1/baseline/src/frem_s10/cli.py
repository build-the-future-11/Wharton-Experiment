"""Explicit local research operations with immutable evidence and fail-closed admission."""

import argparse
import json
from importlib.resources import files
from pathlib import Path

from frem_s10.errors import ExecutionStateError, IntegrityError, ResourceLimitError, S10Error
from frem_s10.identities import canonical_bytes, checked_json, digest_bytes, object_json, parse_json
from frem_s10.runtime.artifacts import publish_json
from frem_s10.runtime.resources import check_host


def plan(manifest: Path) -> dict[str, object]:
    """Validate exact design bytes semantically and enumerate all proposed cells.

    Reads only the explicit metadata document, never source financial values.
    A plan does not grant scientific execution or certify implementation.
    """
    payload = manifest.read_bytes()
    declaration = object_json(parse_json(payload))
    expected = object_json(parse_json(files("frem_s10").joinpath("science.json").read_bytes()))
    if canonical_bytes(declaration) != canonical_bytes(expected):
        raise S10Error("manifest must match the declared S10 design; changes need a recorded design revision")
    forecasts = declaration["forecast_arms"]
    policies = declaration["policy_arms"]
    benchmarks = declaration["benchmarks"]
    ablations = declaration["ablations"]
    assert isinstance(forecasts, list) and isinstance(policies, list)
    assert isinstance(benchmarks, list) and isinstance(ablations, list)
    cells: list[str] = []
    for fold in ("A", "B"):
        for cost in (0, 10, 20, 40):
            cells.extend(f"{fold}/strategy/{f}/{p}/{cost}" for f in forecasts for p in policies)
            cells.extend(f"{fold}/benchmark/{b}/{cost}" for b in benchmarks)
        cells.extend(f"{fold}/ablation/{a}/10" for a in ablations)
    return {
        "status": "PLAN_ONLY",
        "manifest_sha256": digest_bytes(payload),
        "cells": cells,
        "planned_ledgers": len(cells),
        "market_fits": 0,
        "market_ledgers": 0,
        "protected_access": False,
        "scientific_freeze": False,
        "scientific_status": "NOT_EVALUATED",
        "execution_available": False,
        "remaining_gates": [
            "S10 source/calendar admission",
            "complete engineering verification",
            "operation-specific admission, supervisor and consumption integration",
            "pre-2010 precision",
            "full-scale profile and saved-output scientific audit",
            "exact source/data/environment freeze",
            "untouched confirmation and independent reproduction",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Dispatch explicit manifests; real work always passes the fixed production gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planner = sub.add_parser("plan", help="validate exact design and enumerate the proposed matrix")
    planner.add_argument("--manifest", type=Path, required=True)
    for command in (planner, sub.add_parser("check-host", help="read host memory without model imports")):
        command.add_argument(
            "--output", type=Path, help="new receipt in an existing directory; never overwritten"
        )
    for name in ("admit", "freeze", "prepare-precision", "run", "qualify", "audit", "report"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name in ("prepare-precision", "run", "qualify"):
            command.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args.command, getattr(args, "manifest", None))
        if getattr(args, "output", None) is not None:
            publish_json(args.output.parent, args.output.name, checked_json(result))
        print(json.dumps(result, indent=2, allow_nan=False))
        return _result_exit(result)
    except ResourceLimitError as exc:
        print(json.dumps({"status": "RESOURCE_REJECTED", "error": str(exc)}))
        return 3
    except ExecutionStateError as exc:
        print(json.dumps({"status": "CONSUMPTION_REFUSED", "error": str(exc)}))
        return 5
    except (S10Error, OSError) as exc:
        print(json.dumps({"status": "REJECTED_UNCONSUMED", "error": str(exc)}))
        return 2


def _result_exit(result: dict[str, object]) -> int:
    state = result.get("state", result.get("status"))
    return {
        "BLOCKED": 3,
        "REJECTED_UNCONSUMED": 3,
        "FAILED": 6,
        "BUDGET_EXCEEDED": 7,
        "INTERRUPTED": 130,
    }.get(str(state), 0)


def _dispatch(command: str, manifest: Path | None) -> dict[str, object]:
    if command == "check-host":
        return check_host()
    if manifest is None:
        raise IntegrityError("explicit operation manifest required")
    if command == "plan":
        return plan(manifest)
    if command == "qualify":
        from frem_s10.runtime.qualification import run_qualification

        return dict(run_qualification(manifest))
    if command in ("prepare-precision", "run"):
        from frem_s10.runtime.pipeline import run_declared

        return dict(
            run_declared(
                manifest,
                expected_operation="prepare-precision"
                if command == "prepare-precision"
                else "development-campaign",
            )
        )
    if command in ("admit", "freeze"):
        return _prepare(command, manifest)
    return _saved(command, manifest)


def _prepare(command: str, manifest: Path) -> dict[str, object]:
    from frem_s10.runtime.declarations import _schema

    request = _schema(
        parse_json(manifest.read_bytes()),
        {"schema", "request", "artifact_root", "source_root", "consumption_root", "output_name"},
        "preparation manifest",
    )
    if request["schema"] != "s10-preparation-manifest-v1":
        raise IntegrityError("exact preparation manifest required")
    kwargs = {
        "artifact_root": Path(str(request["artifact_root"])),
        "source_root": Path(str(request["source_root"])),
    }
    reference: object
    if command == "admit":
        from frem_s10.data.production import create_source_admission

        reference = create_source_admission(
            Path(str(request["request"])), **kwargs, output_name=str(request["output_name"])
        )
    else:
        from frem_s10.runtime.pipeline import create_declaration

        reference = create_declaration(
            Path(str(request["request"])),
            **kwargs,
            consumption_root=Path(str(request["consumption_root"])),
            output_name=str(request["output_name"]),
        )
    return {"status": "COMPLETE", "operation": command, "reference": reference, "scientific_execution": False}


def _saved(command: str, manifest: Path) -> dict[str, object]:
    from frem_s10.runtime.artifacts import read_verified_json
    from frem_s10.runtime.declarations import _schema

    request = _schema(
        parse_json(manifest.read_bytes()),
        {"schema", "directory", "sha256", "output_directory"},
        "saved evidence operation",
    )
    if request["schema"] != "s10-saved-evidence-request-v1":
        raise IntegrityError("exact saved-evidence manifest required")
    directory, output = Path(str(request["directory"])), Path(str(request["output_directory"]))
    output.mkdir(exist_ok=False)
    if command == "audit":
        from frem_s10.runtime.results import audit_campaign

        identity = audit_campaign(directory, str(request["sha256"]), output)
        return {"status": "COMPLETE", "audit_sha256": identity, "scientific_execution": False}
    audit = object_json(read_verified_json(directory / "audit.json", str(request["sha256"])))
    reference = object_json(audit["report"])
    if reference.get("path") != "report.json":
        raise IntegrityError("canonical saved report reference required")
    report = read_verified_json(directory / "report.json", str(reference["sha256"]), maximum_bytes=8_388_608)
    identity = publish_json(output, "report.json", report)
    return {"status": "COMPLETE", "report_sha256": identity, "scientific_execution": False}
