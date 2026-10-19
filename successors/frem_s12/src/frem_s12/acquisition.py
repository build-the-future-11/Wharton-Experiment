"""Exact allowlisted no-overwrite public archive acquisition."""

from __future__ import annotations

import json
import urllib.request
from importlib.resources import files
from pathlib import Path

from frem_s12.errors import ContractError
from frem_s12.evidence import atomic_json, atomic_write, package_identity, sha256_bytes


def declared_sources() -> dict[tuple[str, int], str]:
    """Return the embedded family/release URL allowlist."""
    value = json.loads(files("frem_s12").joinpath("sources.json").read_text(encoding="utf-8"))
    result: dict[tuple[str, int], str] = {}
    for family, releases in value["families"].items():
        for release, url in releases.items():
            result[(str(family), int(release))] = str(url)
    return result


def acquire(*, family: str, release: int, output: Path, timeout_seconds: int = 120) -> dict[str, object]:
    """Acquire one exact archive and adjacent receipt without overwrite."""
    expected = declared_sources().get((family, release))
    if expected is None:
        raise ContractError("family/release is outside the frozen source contract")
    if output.exists() or output.is_symlink() or output.with_suffix(output.suffix + ".receipt.json").exists():
        raise FileExistsError("archive or receipt already exists")
    request = urllib.request.Request(expected, headers={"User-Agent": "FREM-S12-research/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        final_url = response.geturl()
        payload = response.read(10_000_001)
        metadata = dict(response.headers.items())
    if final_url != expected or not 1 <= len(payload) <= 10_000_000:
        raise ContractError("source redirected or payload violates size contract")
    atomic_write(output, payload)
    result = {
        "schema": "frem-s12-acquisition-v1",
        "family": family,
        "release": release,
        "url": expected,
        "final_url": final_url,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "headers": metadata,
        "source_identity": package_identity()["sha256"],
        "raw_redistribution_authorized": False,
    }
    atomic_json(output.with_suffix(output.suffix + ".receipt.json"), result)
    return result
