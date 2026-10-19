import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from frem_s11.acquisition import declared_sources
from frem_s11.admission import MemorySample, assess, parse_vm_stat
from frem_s11.errors import DataContractError
from frem_s11.evidence import package_identity, verify_admission, verify_archive


def test_vm_stat_parser_and_assessment_preserve_actual_counters() -> None:
    text = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 32768.
Pages speculative: 4096.
Pageouts: 77.
"""
    sample = parse_vm_stat(text, timestamp_utc="2026-10-18T00:00:00+00:00")
    assert sample.available_mib == 576
    receipt = assess((sample, sample, sample))
    assert receipt["status"] == "PASS"
    blocked = assess(
        (
            sample,
            MemorySample(sample.timestamp_utc, sample.page_size_bytes, 32768, 4096, 78),
            sample,
        )
    )
    assert blocked["status"] == "BLOCKED"
    assert "PAGEOUT_COUNTER_CHANGED" in blocked["reasons"]


def test_admission_rejects_stale_source_or_wrong_status(tmp_path: Path) -> None:
    receipt = {
        "schema": "frem-s11-host-admission-v1",
        "observed_utc": datetime.now(UTC).isoformat(),
        "status": "BLOCKED",
        "interpreter": str(Path(sys.executable).resolve()),
        "source_identity": package_identity()["sha256"],
    }
    path = tmp_path / "host.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(DataContractError, match="passing"):
        verify_admission(path)


def test_archive_receipt_binds_bytes_release_and_boundary(tmp_path: Path) -> None:
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"generated archive bytes")
    import hashlib

    receipt = {
        "schema": "frem-s11-acquisition-v2",
        "release_month": 201608,
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_url": declared_sources()[201608],
        "source_identity": package_identity()["sha256"],
        "parsed": False,
        "scientific_result": False,
    }
    sidecar = tmp_path / "release.zip.receipt.json"
    sidecar.write_text(json.dumps(receipt), encoding="utf-8")
    assert verify_archive(archive, sidecar, release_month=201608)["sha256"] == receipt["sha256"]
    archive.write_bytes(b"tampered")
    with pytest.raises(DataContractError, match="do not match"):
        verify_archive(archive, sidecar, release_month=201608)
