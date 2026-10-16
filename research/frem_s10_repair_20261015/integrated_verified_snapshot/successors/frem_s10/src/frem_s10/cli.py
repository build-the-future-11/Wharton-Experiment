"""Working metadata-only CLI; scientific execution awaits the complete verified runner."""

import argparse
import json
from importlib.resources import files
from pathlib import Path

from frem_s10.errors import S10Error
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
    """Run only implemented metadata/host checks and preserve fresh receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planner = sub.add_parser("plan", help="validate exact design and enumerate the proposed matrix")
    planner.add_argument("--manifest", type=Path, required=True)
    for command in (planner, sub.add_parser("check-host", help="read host memory without model imports")):
        command.add_argument(
            "--output", type=Path, help="new receipt in an existing directory; never overwritten"
        )
    args = parser.parse_args(argv)
    try:
        result = plan(args.manifest) if args.command == "plan" else check_host()
        if args.output is not None:
            publish_json(args.output.parent, args.output.name, checked_json(result))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 3 if result["status"] == "BLOCKED" else 0
    except (S10Error, OSError) as exc:
        print(json.dumps({"status": "REJECTED_UNCONSUMED", "error": str(exc)}))
        return 2
