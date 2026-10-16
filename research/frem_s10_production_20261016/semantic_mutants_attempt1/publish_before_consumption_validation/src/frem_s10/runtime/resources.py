"""Read-only macOS host admission; never import numerical libraries or waive floors."""

import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise

from frem_s10.errors import ResourceLimitError


@dataclass(frozen=True)
class MemorySample:
    """Raw kernel counters with their conversion and acquisition timestamp."""

    timestamp: str
    page_size_bytes: int
    free_pages: int
    speculative_pages: int
    pageouts: int

    @property
    def available_mib(self) -> float:
        """Return the declared free-plus-speculative measure."""
        return (self.free_pages + self.speculative_pages) * self.page_size_bytes / 2**20


def parse_vm_stat(output: str, timestamp: str) -> MemorySample:
    """Parse required kernel counters; absent or duplicate fields fail closed."""
    header = re.findall(r"page size of (\d+) bytes", output)
    if len(header) != 1 or int(header[0]) <= 0:
        raise ResourceLimitError("vm_stat page size unavailable")
    values = []
    for field in ("Pages free", "Pages speculative", "Pageouts"):
        matches = re.findall(rf"^{field}:\s+(\d+)\.?\s*$", output, re.MULTILINE)
        if len(matches) != 1:
            raise ResourceLimitError(f"vm_stat counter unavailable: {field}")
        values.append(int(matches[0]))
    return MemorySample(timestamp, int(header[0]), *values)


def sample_memory() -> MemorySample:
    """Collect one sample on the supported host; unavailable is not admission."""
    if sys.platform != "darwin":
        raise ResourceLimitError("equivalent host memory/pageout measurement unverified on this platform")
    try:
        result = subprocess.run(["/usr/bin/vm_stat"], check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceLimitError("host measurement failed") from exc
    return parse_vm_stat(result.stdout, datetime.now(UTC).isoformat())


def assess(samples: tuple[MemorySample, ...]) -> dict[str, object]:
    """Assess exactly three measurements using the unchanged 512 MiB rule."""
    if len(samples) != 3:
        raise ResourceLimitError("exactly three admission samples required")
    reasons = []
    if any(sample.available_mib < 512 for sample in samples):
        reasons.append("FREE_PLUS_SPECULATIVE_BELOW_512_MIB")
    if any(b.pageouts != a.pageouts for a, b in pairwise(samples)):
        reasons.append("PAGEOUT_COUNTER_CHANGED")
    return {
        "status": "BLOCKED" if reasons else "PASS",
        "reasons": reasons,
        "floor_mib": 512,
        "samples": [asdict(s) | {"available_mib": s.available_mib} for s in samples],
        "scientific_execution": False,
        "scope": "host admission only; no worker supervision certified",
    }


def check_host() -> dict[str, object]:
    """Take three readings at least one second apart without running a model."""
    samples = [sample_memory()]
    for _ in range(2):
        time.sleep(1)
        samples.append(sample_memory())
    return assess(tuple(samples))
