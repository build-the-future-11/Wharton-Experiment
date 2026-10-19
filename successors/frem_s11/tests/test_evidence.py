import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from frem_s11.acquisition import declared_sources
from frem_s11.evidence import atomic_json, atomic_write, package_identity, verify_admission, verify_archive


def test_atomic_write_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "evidence.bin"
    atomic_write(path, b"first")
    with pytest.raises(FileExistsError):
        atomic_write(path, b"second")
    assert path.read_bytes() == b"first"


def test_receipts_bind_interpreter_source_url_and_archive_bytes(tmp_path: Path) -> None:
    identity = package_identity()["sha256"]
    admission_path = tmp_path / "admission.json"
    admission = {
        "schema": "frem-s11-host-admission-v1",
        "status": "PASS",
        "interpreter": str(Path(sys.executable).resolve()),
        "source_identity": identity,
        "observed_utc": datetime.now(UTC).isoformat(),
    }
    atomic_json(admission_path, admission)
    assert verify_admission(admission_path)["status"] == "PASS"

    archive = tmp_path / "release.zip"
    archive.write_bytes(b"PK\x03\x04fixture")
    receipt_path = tmp_path / "release.zip.receipt.json"
    receipt = {
        "schema": "frem-s11-acquisition-v2",
        "release_month": 201608,
        "source_url": declared_sources()[201608],
        "source_identity": identity,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "bytes": archive.stat().st_size,
        "parsed": False,
        "scientific_result": False,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert verify_archive(archive, receipt_path, release_month=201608)["bytes"] == archive.stat().st_size
