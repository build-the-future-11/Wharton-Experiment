"""No-overwrite acquisition with exact release-vintage receipts."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

from frem_s11.errors import DataContractError
from frem_s11.evidence import atomic_json, atomic_write, package_identity


def declared_sources() -> dict[int, str]:
    """Return the exact embedded release-to-URL contract."""
    value = json.loads(files("frem_s11").joinpath("sources.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataContractError("embedded sources contract is malformed")
    try:
        result = {int(key): str(url) for key, url in value.items()}
    except (TypeError, ValueError) as exc:
        raise DataContractError("embedded sources contract is malformed") from exc
    if set(result) != {200808, 201608, 202408}:
        raise DataContractError("embedded source releases changed")
    return result


def acquire(url: str, output: Path, *, release_month: int, timeout_seconds: int = 30) -> dict[str, object]:
    """Download one declared HTTPS archive and retain an atomic sidecar receipt."""
    if declared_sources().get(release_month) != url:
        raise DataContractError("URL and release month must match the embedded official source contract")
    if output.exists() or output.with_suffix(output.suffix + ".receipt.json").exists():
        raise FileExistsError("release-vintage acquisition never overwrites existing evidence")
    request = urllib.request.Request(url, headers={"User-Agent": "FREM-S11-research/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read(8 * 1024 * 1024 + 1)
        final_url = response.geturl()
        content_type = response.headers.get_content_type()
        response_date = response.headers.get("Date")
        last_modified = response.headers.get("Last-Modified")
        etag = response.headers.get("ETag")
    if final_url != url or len(payload) > 8 * 1024 * 1024 or not payload:
        raise DataContractError("archive redirect, size, or empty-payload contract failed")
    if content_type not in {"application/zip", "application/x-zip-compressed", "application/octet-stream"}:
        raise DataContractError("archive content type is not accepted")
    if not payload.startswith(b"PK\x03\x04"):
        raise DataContractError("archive payload does not have a ZIP signature")
    receipt = {
        "schema": "frem-s11-acquisition-v2",
        "source_url": url,
        "final_url": final_url,
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "release_month": release_month,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "content_type": content_type,
        "response_date": response_date,
        "last_modified": last_modified,
        "etag": etag,
        "source_identity": package_identity()["sha256"],
        "parsed": False,
        "scientific_result": False,
    }
    sidecar = output.with_suffix(output.suffix + ".receipt.json")
    atomic_write(output, payload)
    try:
        atomic_json(sidecar, receipt)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return receipt
