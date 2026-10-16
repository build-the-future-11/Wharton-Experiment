"""Strict finite manifests for portable generated S10 engineering exercises."""

import platform
import sys
from importlib.metadata import version
from pathlib import Path

from frem_s10.errors import ContractError, IntegrityError
from frem_s10.identities import JSONObject, digest_bytes, digest_json, object_json, parse_json
from frem_s10.metadata import finite_float, strict_int

VARIANTS = ("primary", "repeat_h1", "no_shrink", "window260", "neighbors16", "neighbors64")
KEYS = {
    "schema",
    "scope",
    "device",
    "precision",
    "seed",
    "steps",
    "variants",
    "budget_seconds",
    "dataset",
    "checkpoint_every_origins",
    "threads",
}


def read_manifest(path: Path) -> JSONObject:
    """Reject ignored flags, nonfinite settings, unbounded matrices and market data."""
    if path.is_symlink() or path.stat().st_size > 65536:
        raise ContractError("bounded plain smoke manifest required")
    value = object_json(parse_json(path.read_bytes()))
    if set(value) != KEYS or value["schema"] != "s10-generated-smoke-v1":
        raise ContractError("exact generated smoke manifest schema required")
    if value["scope"] not in ("smoke", "development-engineering"):
        raise ContractError("only generated engineering scope is supported")
    if (value["device"], value["precision"], value["dataset"]) != (
        "cpu",
        "float64",
        "s10-generated-smoke-data-v1",
    ):
        raise ContractError("S10 supports CPU float64 generated smoke only; CUDA has no implemented backend")
    if type(value["threads"]) is not int or value["threads"] != 1:
        raise ContractError("one thread is required")
    if type(value["checkpoint_every_origins"]) is not int or value["checkpoint_every_origins"] != 1:
        raise ContractError("every completed origin must be checkpointed")
    seed = strict_int(value["seed"], name="seed", minimum=0)
    steps = strict_int(value["steps"], name="steps", minimum=1)
    budget = finite_float(value["budget_seconds"], name="budget_seconds", strictly_positive=True)
    if seed >= 2**32 or steps > 80 or budget > 300:
        raise ContractError("generated engineering bounds: seed<2**32, steps<=80, wall<=300s")
    if value["scope"] == "smoke" and steps > 12:
        raise ContractError("tiny smoke is limited to twelve origins; use admitted development-engineering")
    variants = value["variants"]
    if (
        not isinstance(variants, list)
        or not variants
        or any(not isinstance(item, str) or item not in VARIANTS for item in variants)
        or len(variants) != len({str(item) for item in variants})
    ):
        raise ContractError("unique declared model variants required")
    return value


def smoke_plan(manifest: Path) -> JSONObject:
    """Enumerate every generated model cell without NumPy import or execution."""
    config = read_manifest(manifest)
    variants = config["variants"]
    assert isinstance(variants, list)
    return {
        "status": "PLAN_ONLY",
        "configuration": config,
        "configuration_sha256": digest_json(config),
        "cells": [{"variant": name, "seed": config["seed"], "origins": config["steps"]} for name in variants],
        "market_execution": False,
        "gpu_execution": False,
        "scientific_status": "NOT_EVALUATED",
        "resume_boundary": "all declared variants committed at one completed origin",
        "budget_semantics": "cumulative active wall time; checked between origins; no runtime promise",
    }


def runtime_identity() -> JSONObject:
    """Bind the installed source bytes and numeric environment, excluding file paths."""
    package = Path(__file__).resolve().parents[1]
    members = [*sorted(package.rglob("*.py")), package / "science.json"]
    if any(path.is_symlink() for path in members):
        raise IntegrityError("source symlinks cannot identify a generated replay")
    return {
        "source_sha256": digest_json(
            {str(p.relative_to(package)): digest_bytes(p.read_bytes()) for p in members}
        ),
        "python": sys.version,
        "numpy": version("numpy"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "precision": "float64",
        "device": "cpu",
    }
