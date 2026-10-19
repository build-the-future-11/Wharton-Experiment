"""Pre-import macOS host admission for bounded S11 research."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from frem_s11.errors import DataContractError
from frem_s11.evidence import atomic_json, package_identity


@dataclass(frozen=True, slots=True)
class MemorySample:
    """One macOS free-plus-speculative and pageout observation."""

    timestamp_utc: str
    page_size_bytes: int
    free_pages: int
    speculative_pages: int
    pageouts: int

    @property
    def available_mib(self) -> float:
        """Return the frozen free-plus-speculative admission measure."""
        return (self.free_pages + self.speculative_pages) * self.page_size_bytes / 2**20


def parse_vm_stat(text: str, *, timestamp_utc: str) -> MemorySample:
    """Parse bounded `vm_stat` output without importing numerical libraries."""
    match = re.search(r"page size of ([0-9]+) bytes", text)
    if match is None:
        raise DataContractError("vm_stat page size is unavailable")
    counters: dict[str, int] = {}
    for line in text.splitlines():
        row = re.fullmatch(r"([^:]+):\s+([0-9]+)\.?", line.strip())
        if row:
            counters[row.group(1)] = int(row.group(2))
    try:
        return MemorySample(
            timestamp_utc,
            int(match.group(1)),
            counters["Pages free"],
            counters["Pages speculative"],
            counters["Pageouts"],
        )
    except KeyError as exc:
        raise DataContractError("vm_stat required counter is unavailable") from exc


def sample_memory() -> MemorySample:
    """Collect one read-only macOS kernel counter sample."""
    if sys.platform != "darwin" or sys.version_info[:2] != (3, 12):
        raise DataContractError("S11 requires a supported macOS Python 3.12 host")
    completed = subprocess.run(
        ["/usr/bin/vm_stat"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0 or len(completed.stdout) > 64 * 1024:
        raise DataContractError("vm_stat sampling failed")
    return parse_vm_stat(completed.stdout, timestamp_utc=datetime.now(UTC).isoformat())


def assess(samples: tuple[MemorySample, ...], *, floor_mib: float = 512.0) -> dict[str, object]:
    """Apply the exact three-sample memory and stable-pageout rule."""
    if len(samples) != 3:
        raise DataContractError("exactly three host samples are required")
    reasons: list[str] = []
    if any(sample.available_mib < floor_mib for sample in samples):
        reasons.append("AVAILABLE_MEMORY_BELOW_512_MIB")
    if any(left.pageouts != right.pageouts for left, right in pairwise(samples)):
        reasons.append("PAGEOUT_COUNTER_CHANGED")
    identity = package_identity()
    return {
        "schema": "frem-s11-host-admission-v1",
        "observed_utc": datetime.now(UTC).isoformat(),
        "status": "BLOCKED" if reasons else "PASS",
        "reasons": reasons,
        "floor_mib": floor_mib,
        "pressure_rule": "pageouts equal across all samples",
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "interpreter": str(Path(sys.executable).resolve()),
        "source_identity": identity["sha256"],
        "samples": [{**asdict(sample), "available_mib": sample.available_mib} for sample in samples],
        "scientific_execution": False,
    }


def run_admission(output: Path, *, interval_seconds: float = 1.0) -> dict[str, object]:
    """Sample three times and preserve a pass or blocked receipt."""
    rows: list[MemorySample] = []
    for index in range(3):
        rows.append(sample_memory())
        if index < 2:
            time.sleep(interval_seconds)
    receipt = assess(tuple(rows))
    atomic_json(output, receipt)
    return receipt


def execution_pressure_stable(admission: dict[str, object], post: MemorySample) -> bool:
    """Check that pageouts did not rise after the admitted scientific run."""
    samples = admission.get("samples")
    if not isinstance(samples, list) or len(samples) != 3 or not isinstance(samples[-1], dict):
        raise DataContractError("host admission samples are malformed")
    previous = samples[-1].get("pageouts")
    if not isinstance(previous, int):
        raise DataContractError("host admission pageout counter is malformed")
    return post.pageouts == previous
