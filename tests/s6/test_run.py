"""Read-only launcher admission/source checks; no scientific fitting is invoked."""

import hashlib
import json
from pathlib import Path

import pytest

from successors.frem_s6 import check, run


def launcher_inputs(tmp_path, monkeypatch):
    dataset = tmp_path / "windows.json"
    dataset.write_text("OUTCOMES MUST NOT BE READ BY RESOURCE PREFLIGHT")
    Path(str(dataset) + ".manifest.json").write_text("sidecar presence fixture")
    protocol = tmp_path / "protocol.json"
    declaration = {
        "status": "FROZEN_DEVELOPMENT", "cells": 24, "seeds": [11, 29, 47],
        "protected_start": 1735689600, "data_sha256": "a" * 64,
        "fold_rationale": "Fixture only; no model execution is requested.",
        "precision_analysis": "Fixture preflight does not certify scientific precision.",
        "spent_date_inventory": "Fixture dates are not a market evaluation.",
        "primary_endpoint": "date_nll", "economic_policy": "fixed signal",
        "limitations": "Numerical admission unit test only",
    }
    protocol.write_text(json.dumps(declaration))
    integrity = tmp_path / "integrity"
    integrity.mkdir()
    (integrity / "receipt.json").write_text(json.dumps({"returncode": 0, "sources_unchanged": True}))
    (integrity / "source_hashes.json").write_text(json.dumps({"source.py": "b" * 64}))
    monkeypatch.setattr(run, "source_identities", lambda: {"source.py": "b" * 64})
    monkeypatch.setattr(run.time, "sleep", lambda _: None)
    monkeypatch.setattr(run, "memory_snapshot", lambda: {"available_mib": 1024, "pageouts": 9})
    return dataset, protocol, integrity, declaration


def test_metadata_preflight_requires_matching_current_integrity(tmp_path, monkeypatch):
    dataset, protocol, integrity, _ = launcher_inputs(tmp_path, monkeypatch)
    result = run.preflight(dataset, protocol, integrity)
    assert result["status"] == "ADMITTED"
    assert result["market_fits_started"] == 0
    assert result["protected_access"] is False
    assert len(result["samples"]) == 3
    monkeypatch.setattr(run, "source_identities", lambda: {"source.py": "c" * 64})
    changed = run.preflight(dataset, protocol, integrity)
    assert changed["status"] == "BLOCKED_BEFORE_SCIENTIFIC_IMPORT"
    assert "SOURCE_CHANGED_SINCE_INTEGRITY_CHECK" in changed["reasons"]


@pytest.mark.parametrize("snapshots", [
    [{"available_mib": 900, "pageouts": 0}, {"available_mib": 511, "pageouts": 0},
     {"available_mib": 900, "pageouts": 0}],
    [{"available_mib": 900, "pageouts": 0}, {"available_mib": 900, "pageouts": 1},
     {"available_mib": 900, "pageouts": 1}],
])
def test_resource_failure_blocks_before_reading_outcome_payload(tmp_path, monkeypatch, snapshots):
    dataset, protocol, integrity, _ = launcher_inputs(tmp_path, monkeypatch)
    samples = iter(snapshots)
    monkeypatch.setattr(run, "memory_snapshot", lambda: next(samples))
    original_bytes, original_text = Path.read_bytes, Path.read_text

    def guarded_bytes(path):
        if path == dataset:
            raise AssertionError("resource failure must precede payload access")
        return original_bytes(path)

    def guarded_text(path, *args, **kwargs):
        if path == dataset:
            raise AssertionError("resource failure must precede payload access")
        return original_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_text)
    result = run.preflight(dataset, protocol, integrity)
    assert result["status"] == "BLOCKED_BEFORE_SCIENTIFIC_IMPORT"
    assert any(reason.startswith("RESOURCE_ADMISSION_FAILED") for reason in result["reasons"])


@pytest.mark.parametrize("field,value", [("cells", 23), ("seeds", [11, 29]),
                                        ("protected_start", 1767225600)])
def test_declared_matrix_and_protected_boundary_cannot_silently_change(tmp_path, monkeypatch, field, value):
    dataset, protocol, integrity, declaration = launcher_inputs(tmp_path, monkeypatch)
    declaration[field] = value
    protocol.write_text(json.dumps(declaration))
    result = run.preflight(dataset, protocol, integrity)
    assert "PROTOCOL_MATRIX_OR_PROTECTED_BOUNDARY_MISMATCH" in result["reasons"]


def test_missing_admission_material_and_failed_integrity_block(tmp_path, monkeypatch):
    dataset, protocol, integrity, _ = launcher_inputs(tmp_path, monkeypatch)
    Path(str(dataset) + ".manifest.json").unlink()
    (integrity / "receipt.json").write_text(json.dumps({"returncode": 1, "sources_unchanged": True}))
    result = run.preflight(dataset, protocol, integrity)
    assert "MISSING_ADMITTED_WINDOW_DATA_AND_MANIFEST" in result["reasons"]
    assert "INTEGRITY_NOT_PASSED" in result["reasons"]
    missing = run.preflight(None, None, None)
    assert "MISSING_FROZEN_DEVELOPMENT_PROTOCOL" in missing["reasons"]
    assert "MISSING_CURRENT_INTEGRITY_RECEIPT" in missing["reasons"]


def test_source_identity_includes_new_tests_and_external_runtime_dependencies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("successors/frem_s6/kernel.py", "tests/s6/test_kernel.py",
                 "successors/frem_portfolio_v1/engine.py", "scripts/public_equity_backtests.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original source\n")
    original = check.source_identities()
    assert set(original) == {"successors/frem_s6/kernel.py", "tests/s6/test_kernel.py",
                             "successors/frem_portfolio_v1/engine.py", "scripts/public_equity_backtests.py"}
    Path("successors/frem_portfolio_v1/engine.py").write_text("changed dependency\n")
    assert check.source_identities() != original
    Path("tests/s6/test_added.py").write_text("additional contract\n")
    assert "tests/s6/test_added.py" in check.source_identities()
    Path("successors/frem_s6/kernel.py").unlink()
    assert "successors/frem_s6/kernel.py" not in check.source_identities()


def test_receipt_write_refuses_overwrite(tmp_path):
    receipt = tmp_path / "receipt.json"
    check.write(receipt, {"status": "original failure"})
    with pytest.raises(FileExistsError):
        check.write(receipt, {"status": "repaired pass"})
    assert json.loads(receipt.read_text()) == {"status": "original failure"}


def test_dataset_digest_streams_without_read_bytes(tmp_path, monkeypatch):
    path = tmp_path / "opaque.dat"
    content = b"opaque identity bytes\n" * 100003
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()

    def forbidden_read_bytes(_):
        raise AssertionError("launcher hashing must not allocate the whole dataset")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    assert run.digest(path) == expected


def test_preflight_parse_failure_retained_without_starting_worker(tmp_path, monkeypatch):
    dataset, protocol, integrity, _ = launcher_inputs(tmp_path, monkeypatch)
    protocol.write_text("{malformed")
    output = tmp_path / "attempt"
    monkeypatch.setattr(run.sys, "argv", ["s6-run", "--execute", "--output", str(output),
                                          "--dataset", str(dataset), "--protocol", str(protocol),
                                          "--integrity", str(integrity)])

    def no_worker(*args, **kwargs):
        raise AssertionError("invalid preflight must not start a subprocess")

    monkeypatch.setattr(run.subprocess, "run", no_worker)
    with pytest.raises(json.JSONDecodeError):
        run.main()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["stage"] == "preflight"
    assert failure["market_fits_started"] == 0
