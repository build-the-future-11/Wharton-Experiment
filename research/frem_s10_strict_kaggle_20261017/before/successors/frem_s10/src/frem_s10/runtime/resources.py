"""Read-only platform-specific host admission without numerical imports.

Linux uses MemAvailable, pswpout and visible cgroup-v2 ancestor headroom.
These are distinct from the preserved macOS free-plus-speculative contract.
Sources: https://docs.kernel.org/filesystems/proc.html and
https://docs.kernel.org/admin-guide/cgroup-v2.html .
"""

import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath

from frem_s10.errors import ContractError, ResourceLimitError
from frem_s10.identities import JSONObject, JSONValue, checked_json, object_json
from frem_s10.metadata import parse_utc_timestamp

LINUX_CONTRACT = "s10-linux-memory-admission-v1"
_LINUX_MAX_BYTES = 65_536


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


def _nonnegative(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ResourceLimitError(f"invalid Linux resource counter: {name}")
    return value


def _absolute_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value.startswith("//")
        or str(PurePosixPath(value)) != value
        or ".." in PurePosixPath(value).parts
        or "\x00" in value
        or len(PurePosixPath(value).parts) > 128
    ):
        raise ResourceLimitError("bounded canonical absolute cgroup path required")
    return value


@dataclass(frozen=True)
class CgroupMemoryLimit:
    """One non-root ancestor's actual byte limit and subtree memory usage."""

    path: str
    memory_max_bytes: int | None
    memory_current_bytes: int

    def __post_init__(self) -> None:
        """Reject ambiguous paths, negative counters and unsupported integer ranges."""
        if _absolute_path(self.path) == "/":
            raise ResourceLimitError("the true cgroup root has no memory limit files")
        _nonnegative(self.memory_current_bytes, "memory.current")
        if self.memory_max_bytes is not None:
            _nonnegative(self.memory_max_bytes, "memory.max")

    @property
    def headroom_bytes(self) -> int | None:
        """Clamp over-limit usage to zero; max means no local hard limit."""
        if self.memory_max_bytes is None:
            return None
        return max(0, self.memory_max_bytes - self.memory_current_bytes)


@dataclass(frozen=True)
class LinuxSample:
    """A Linux observation with complete visible ancestry and truthful counter names."""

    timestamp: str
    mem_available_kib: int
    pswpout_pages: int
    cgroup_path: str
    cgroup_mount: str
    cgroup_limits: tuple[CgroupMemoryLimit, ...]

    def __post_init__(self) -> None:
        """Require all current-to-root constraints, excluding the unlimited true root."""
        try:
            parse_utc_timestamp(self.timestamp)
        except ContractError as exc:
            raise ResourceLimitError("Linux sample timestamp unavailable") from exc
        _nonnegative(self.mem_available_kib, "MemAvailable")
        _nonnegative(self.pswpout_pages, "pswpout")
        path = PurePosixPath(_absolute_path(self.cgroup_path))
        _absolute_path(self.cgroup_mount)
        expected = tuple(str(item) for item in (path, *path.parents) if str(item) != "/")
        if not isinstance(self.cgroup_limits, tuple) or any(
            not isinstance(item, CgroupMemoryLimit) for item in self.cgroup_limits
        ):
            raise ResourceLimitError("typed immutable cgroup ancestry required")
        if tuple(item.path for item in self.cgroup_limits) != expected:
            raise ResourceLimitError("complete ordered cgroup ancestor constraints required")

    @property
    def available_mib(self) -> float:
        """Use the smaller host estimate or unreclaimed hard-limit headroom."""
        limits = [self.mem_available_kib * 1024]
        limits.extend(value for item in self.cgroup_limits if (value := item.headroom_bytes) is not None)
        return min(limits) / 2**20


@dataclass(frozen=True)
class KaggleSample:
    """Visible Kaggle namespace counters under the explicit platform contract."""

    timestamp: str
    mem_available_kib: int
    pswpout_pages: int
    memory_max_bytes: int
    memory_high_bytes: int
    memory_current_bytes: int
    memory_events: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """Reject malformed, incomplete, or inconsistent platform counters."""
        try:
            parse_utc_timestamp(self.timestamp)
        except ContractError as exc:
            raise ResourceLimitError("Kaggle sample timestamp unavailable") from exc
        for name, value in (
            ("MemAvailable", self.mem_available_kib),
            ("pswpout", self.pswpout_pages),
            ("memory.max", self.memory_max_bytes),
            ("memory.high", self.memory_high_bytes),
            ("memory.current", self.memory_current_bytes),
        ):
            _nonnegative(value, name)
        if self.memory_high_bytes > self.memory_max_bytes:
            raise ResourceLimitError("Kaggle memory.high exceeds memory.max")
        names = tuple(name for name, _ in self.memory_events)
        if len(names) != len(set(names)) or not {"high", "max", "oom", "oom_kill"} <= set(names):
            raise ResourceLimitError("complete unique Kaggle pressure events required")
        for name, value in self.memory_events:
            if not re.fullmatch(r"[a-z_]+", name):
                raise ResourceLimitError("malformed Kaggle pressure event name")
            _nonnegative(value, name)

    @property
    def available_mib(self) -> float:
        """Return the smaller host or visible cgroup headroom."""
        ceiling = min(self.memory_max_bytes, self.memory_high_bytes)
        return min(self.mem_available_kib * 1024, max(0, ceiling - self.memory_current_bytes)) / 2**20


def memory_observation(sample: MemorySample | LinuxSample | KaggleSample) -> JSONObject:
    """Serialize original counters without assigning Linux values macOS labels."""
    raw = asdict(sample)
    if isinstance(sample, LinuxSample):
        raw["cgroup_limits"] = [asdict(item) for item in sample.cgroup_limits]
    if isinstance(sample, KaggleSample):
        raw["memory_events"] = [[name, value] for name, value in sample.memory_events]
    row = object_json(checked_json(raw))
    row["available_mib"] = sample.available_mib
    if isinstance(sample, LinuxSample):
        row.update({"platform": "linux", "contract": LINUX_CONTRACT, "cgroup_hierarchy": "FULL_ROOT_VISIBLE"})
    if isinstance(sample, KaggleSample):
        from frem_s10.runtime.kaggle_contract import CONTRACT

        row.update(
            {
                "platform": "linux",
                "contract": CONTRACT,
                "cgroup_hierarchy": "KAGGLE_NAMESPACE_ROOT",
            }
        )
    return row


def pressure_counter(sample: MemorySample | LinuxSample | KaggleSample) -> int:
    """Return the platform's pressure counter; receipt names keep its actual meaning."""
    return sample.pswpout_pages if isinstance(sample, (LinuxSample, KaggleSample)) else sample.pageouts


def _counter_pairs(text: str, *, name: str) -> tuple[tuple[str, int], ...]:
    rows: list[tuple[str, int]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2 or re.fullmatch(r"[a-z_]+", fields[0]) is None:
            raise ResourceLimitError(f"malformed {name} row")
        rows.append((fields[0], int(_limit_integer(fields[1]) or 0)))
    if not rows:
        raise ResourceLimitError(f"empty {name} counters")
    return tuple(rows)


def sample_kaggle_memory(
    proc_root: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")
) -> KaggleSample:
    """Read the exact visible namespace counters accepted by Kaggle contract v1."""
    from frem_s10.runtime.kaggle_contract import require_platform
    from frem_s10.runtime.preflight import parse_linux_resources

    require_platform()
    if _read_linux_text(proc_root / "self/cgroup").strip() != "0::/":
        raise ResourceLimitError("Kaggle contract requires the observed namespace-root membership")
    maximum = _limit_integer(_read_linux_text(cgroup_root / "memory.max"))
    current = _limit_integer(_read_linux_text(cgroup_root / "memory.current"))
    high_raw = _read_linux_text(cgroup_root / "memory.high").strip()
    high = maximum if high_raw == "max" else _limit_integer(high_raw)
    assert maximum is not None and current is not None and high is not None
    raw = parse_linux_resources(
        _read_linux_text(proc_root / "meminfo"),
        _read_linux_text(proc_root / "vmstat"),
        datetime.now(UTC).isoformat(),
    )
    events = _counter_pairs(_read_linux_text(cgroup_root / "memory.events"), name="memory.events")
    if _read_linux_text(proc_root / "self/cgroup").strip() != "0::/":
        raise ResourceLimitError("Kaggle cgroup membership changed during observation")
    return KaggleSample(
        raw.timestamp,
        raw.mem_available_kib,
        raw.pswpout_pages,
        maximum,
        high,
        current,
        events,
    )


def assess_kaggle(samples: tuple[KaggleSample, ...]) -> dict[str, object]:
    """Assess three Kaggle samples with reserve and pressure-event stability."""
    from frem_s10.runtime.kaggle_contract import CONTRACT

    if len(samples) != 3:
        raise ResourceLimitError("exactly three Kaggle admission samples required")
    reasons: list[str] = []
    if any(sample.available_mib < 1536 for sample in samples):
        reasons.append("KAGGLE_VISIBLE_HEADROOM_BELOW_WORKER_PLUS_512_MIB_RESERVE")
    for first, second in pairwise(samples):
        if first.pswpout_pages != second.pswpout_pages:
            reasons.append("KAGGLE_PSWPOUT_CHANGED")
        if first.memory_events != second.memory_events:
            reasons.append("KAGGLE_MEMORY_EVENTS_CHANGED")
        if (first.memory_max_bytes, first.memory_high_bytes) != (
            second.memory_max_bytes,
            second.memory_high_bytes,
        ):
            reasons.append("KAGGLE_MEMORY_LIMIT_CHANGED")
    times = [parse_utc_timestamp(sample.timestamp) for sample in samples]
    if any(b - a < 1_000_000 for a, b in pairwise(times)):
        reasons.append("KAGGLE_SAMPLE_SPACING_BELOW_ONE_SECOND")
    return {
        "status": "BLOCKED" if reasons else "PASS",
        "platform": "linux",
        "contract": CONTRACT,
        "reasons": list(dict.fromkeys(reasons)),
        "floor_mib": 1536,
        "samples": [memory_observation(sample) for sample in samples],
        "scientific_execution": False,
        "scope": "Kaggle platform admission; worker supervision remains a separate gate",
    }


def _read_linux_text(path: Path) -> str:
    if path.is_symlink():
        raise ResourceLimitError("Linux resource file cannot be a symlink")
    try:
        with path.open("rb") as stream:
            raw = stream.read(_LINUX_MAX_BYTES + 1)
        if len(raw) > _LINUX_MAX_BYTES:
            raise ResourceLimitError("Linux resource file exceeds the bounded input size")
        return raw.decode("ascii")
    except (OSError, UnicodeError) as exc:
        raise ResourceLimitError("Linux resource file unavailable") from exc


def _cgroup_path(text: str) -> str:
    rows = text.splitlines()
    if len(rows) != 1 or not rows[0].startswith("0::"):
        raise ResourceLimitError("exactly one unified cgroup-v2 membership required; v1 is unsupported")
    return _absolute_path(rows[0][3:])


def _mount_unescape(value: str) -> str:
    if re.search(r"\\(?!040|011|012|134)", value):
        raise ResourceLimitError("unsupported cgroup mount escape")
    return re.sub(r"\\(040|011|012|134)", lambda match: chr(int(match.group(1), 8)), value)


def _cgroup_mount(text: str) -> Path:
    mounts: list[Path] = []
    for line in text.splitlines():
        fields = line.split()
        if "-" not in fields:
            raise ResourceLimitError("malformed mountinfo row")
        separator = fields.index("-")
        if separator < 6 or len(fields) < separator + 4:
            raise ResourceLimitError("incomplete mountinfo row")
        kind = fields[separator + 1]
        if kind == "cgroup":
            raise ResourceLimitError("cgroup-v1 or hybrid mounts are unsupported")
        if kind == "cgroup2":
            if _mount_unescape(fields[3]) != "/":
                raise ResourceLimitError("cgroup mount hides ancestor constraints")
            mounts.append(Path(_absolute_path(_mount_unescape(fields[4]))))
    if len(mounts) != 1:
        raise ResourceLimitError("one unambiguous full-root cgroup-v2 mount required")
    return mounts[0]


def _limit_integer(text: str, *, unlimited: bool = False) -> int | None:
    value = text.strip()
    if unlimited and value == "max":
        return None
    if re.fullmatch(r"[0-9]{1,20}", value) is None:
        raise ResourceLimitError("malformed cgroup memory counter")
    return _nonnegative(int(value), "cgroup memory")


def _cgroup_limits(mount: Path, membership: str) -> tuple[CgroupMemoryLimit, ...]:
    if any(path.is_symlink() for path in (mount, *mount.parents)) or not mount.is_dir():
        raise ResourceLimitError("plain cgroup-v2 mount required")
    controllers = _read_linux_text(mount / "cgroup.controllers").split()
    if "memory" not in controllers or len(controllers) != len(set(controllers)):
        raise ResourceLimitError("cgroup-v2 memory controller unavailable")
    if any(
        (mount / name).exists() or (mount / name).is_symlink() for name in ("memory.max", "memory.current")
    ):
        raise ResourceLimitError("namespaced cgroup root hides possible ancestor constraints")
    path = PurePosixPath(membership)
    values: list[CgroupMemoryLimit] = []
    for ancestor in (path, *path.parents):
        if str(ancestor) == "/":
            continue
        directory = mount / str(ancestor).lstrip("/")
        if any(item.is_symlink() for item in (directory, *directory.parents)) or not directory.is_dir():
            raise ResourceLimitError("plain cgroup ancestor directory required")
        maximum = _limit_integer(_read_linux_text(directory / "memory.max"), unlimited=True)
        current = _limit_integer(_read_linux_text(directory / "memory.current"))
        assert current is not None
        values.append(CgroupMemoryLimit(str(ancestor), maximum, current))
    return tuple(values)


def sample_linux_memory(proc_root: Path = Path("/proc")) -> LinuxSample:
    """Read bounded Linux counters and all exposed ancestors; fixtures may use another proc root.

    Namespace/delegated roots with hidden ancestors are unsupported. A passing
    host observation does not verify the separate Linux production environment.
    """
    from frem_s10.runtime.preflight import parse_linux_resources

    membership = _cgroup_path(_read_linux_text(proc_root / "self/cgroup"))
    mount = _cgroup_mount(_read_linux_text(proc_root / "self/mountinfo"))
    limits = _cgroup_limits(mount, membership)
    raw = parse_linux_resources(
        _read_linux_text(proc_root / "meminfo"),
        _read_linux_text(proc_root / "vmstat"),
        datetime.now(UTC).isoformat(),
    )
    if membership != _cgroup_path(_read_linux_text(proc_root / "self/cgroup")) or mount != _cgroup_mount(
        _read_linux_text(proc_root / "self/mountinfo")
    ):
        raise ResourceLimitError("cgroup membership or mount changed during observation")
    return LinuxSample(
        raw.timestamp, raw.mem_available_kib, raw.pswpout_pages, membership, str(mount), limits
    )


def _linux_identity(sample: LinuxSample) -> tuple[object, ...]:
    return (
        sample.cgroup_path,
        sample.cgroup_mount,
        tuple((item.path, item.memory_max_bytes) for item in sample.cgroup_limits),
    )


def assess_linux(samples: tuple[LinuxSample, ...]) -> dict[str, object]:
    """Assess three distinct Linux samples under the explicit conservative contract."""
    if len(samples) != 3:
        raise ResourceLimitError("exactly three Linux admission samples required")
    times = [parse_utc_timestamp(sample.timestamp) for sample in samples]
    reasons = []
    if any(sample.available_mib < 512 for sample in samples):
        reasons.append("LINUX_HOST_OR_CGROUP_HEADROOM_BELOW_512_MIB")
    if any(b.pswpout_pages != a.pswpout_pages for a, b in pairwise(samples)):
        reasons.append("LINUX_PSWPOUT_COUNTER_CHANGED")
    if any(_linux_identity(b) != _linux_identity(a) for a, b in pairwise(samples)):
        reasons.append("LINUX_CGROUP_CONSTRAINT_IDENTITY_CHANGED")
    if any(b - a < 1_000_000 for a, b in pairwise(times)):
        reasons.append("LINUX_SAMPLE_SPACING_BELOW_ONE_SECOND")
    return {
        "status": "BLOCKED" if reasons else "PASS",
        "platform": "linux",
        "contract": LINUX_CONTRACT,
        "reasons": reasons,
        "floor_mib": 512,
        "samples": [memory_observation(sample) for sample in samples],
        "memory_semantics": "min(Linux MemAvailable, each visible cgroup-v2 memory.max-memory.current)",
        "pressure_semantics": "Linux pswpout pages swapped out; not macOS Pageouts",
        "scientific_execution": False,
        "scope": "Linux host admission only; production environment and supervision remain separate gates",
    }


def _linux_row(row: JSONObject) -> LinuxSample:
    expected = {
        "timestamp",
        "mem_available_kib",
        "pswpout_pages",
        "cgroup_path",
        "cgroup_mount",
        "cgroup_limits",
        "available_mib",
        "platform",
        "contract",
        "cgroup_hierarchy",
    }
    if set(row) != expected or (row["platform"], row["contract"], row["cgroup_hierarchy"]) != (
        "linux",
        LINUX_CONTRACT,
        "FULL_ROOT_VISIBLE",
    ):
        raise ResourceLimitError("Linux resource receipt schema or semantics differ")
    raw_limits = row["cgroup_limits"]
    if not isinstance(raw_limits, list) or len(raw_limits) > 127:
        raise ResourceLimitError("bounded cgroup ancestry list required")
    limits: list[CgroupMemoryLimit] = []
    for raw in raw_limits:
        item = object_json(raw)
        if set(item) != {"path", "memory_max_bytes", "memory_current_bytes"}:
            raise ResourceLimitError("exact cgroup constraint schema required")
        maximum = item["memory_max_bytes"]
        limits.append(
            CgroupMemoryLimit(
                _absolute_path(item["path"]),
                None if maximum is None else _nonnegative(maximum, "memory.max"),
                _nonnegative(item["memory_current_bytes"], "memory.current"),
            )
        )
    if not isinstance(row["timestamp"], str):
        raise ResourceLimitError("Linux resource timestamp required")
    return LinuxSample(
        row["timestamp"],
        _nonnegative(row["mem_available_kib"], "MemAvailable"),
        _nonnegative(row["pswpout_pages"], "pswpout"),
        _absolute_path(row["cgroup_path"]),
        _absolute_path(row["cgroup_mount"]),
        tuple(limits),
    )


def _kaggle_row(row: JSONObject) -> KaggleSample:
    expected = {
        "timestamp",
        "mem_available_kib",
        "pswpout_pages",
        "memory_max_bytes",
        "memory_high_bytes",
        "memory_current_bytes",
        "memory_events",
        "available_mib",
        "platform",
        "contract",
        "cgroup_hierarchy",
    }
    if set(row) != expected or row.get("cgroup_hierarchy") != "KAGGLE_NAMESPACE_ROOT":
        raise ResourceLimitError("exact Kaggle memory receipt fields required")
    events = row["memory_events"]
    if not isinstance(events, list):
        raise ResourceLimitError("Kaggle memory events must be a list")
    parsed: list[tuple[str, int]] = []
    for item in events:
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
            raise ResourceLimitError("malformed Kaggle memory event")
        parsed.append((item[0], _nonnegative(item[1], item[0])))
    if not isinstance(row["timestamp"], str):
        raise ResourceLimitError("Kaggle timestamp required")
    return KaggleSample(
        row["timestamp"],
        _nonnegative(row["mem_available_kib"], "MemAvailable"),
        _nonnegative(row["pswpout_pages"], "pswpout"),
        _nonnegative(row["memory_max_bytes"], "memory.max"),
        _nonnegative(row["memory_high_bytes"], "memory.high"),
        _nonnegative(row["memory_current_bytes"], "memory.current"),
        tuple(parsed),
    )


def _memory_row(row: JSONObject) -> MemorySample | LinuxSample | KaggleSample:
    if "platform" in row:
        from frem_s10.runtime.kaggle_contract import CONTRACT

        if row.get("contract") == CONTRACT:
            return _kaggle_row(row)
        return _linux_row(row)
    if set(row) != {
        "timestamp",
        "page_size_bytes",
        "free_pages",
        "speculative_pages",
        "pageouts",
        "available_mib",
    }:
        raise ResourceLimitError("exact macOS memory receipt fields required")
    if not isinstance(row["timestamp"], str):
        raise ResourceLimitError("memory receipt timestamp required")
    size = _nonnegative(row["page_size_bytes"], "page size")
    if not size:
        raise ResourceLimitError("positive page size required")
    return MemorySample(
        row["timestamp"],
        size,
        _nonnegative(row["free_pages"], "free pages"),
        _nonnegative(row["speculative_pages"], "speculative pages"),
        _nonnegative(row["pageouts"], "pageouts"),
    )


def _stable_resource_identity(
    first: MemorySample | LinuxSample | KaggleSample, second: MemorySample | LinuxSample | KaggleSample
) -> None:
    if (
        isinstance(first, LinuxSample)
        and isinstance(second, LinuxSample)
        and (_linux_identity(first) != _linux_identity(second))
    ):
        raise ResourceLimitError("cgroup constraint identity changed")
    if (
        isinstance(first, KaggleSample)
        and isinstance(second, KaggleSample)
        and (first.memory_max_bytes, first.memory_high_bytes, first.memory_events)
        != (second.memory_max_bytes, second.memory_high_bytes, second.memory_events)
    ):
        raise ResourceLimitError("Kaggle resource constraint identity changed")


def audit_memory_rows(rows: list[JSONValue]) -> int:
    """Recompute either platform's three-sample passing receipt and stable pressure counter."""
    if len(rows) != 3:
        raise ResourceLimitError("exactly three resource rows required")
    try:
        documents = [object_json(row) for row in rows]
        samples = [_memory_row(row) for row in documents]
        times = [parse_utc_timestamp(sample.timestamp) for sample in samples]
        for row, sample in zip(documents, samples, strict=True):
            if type(row["available_mib"]) not in (int, float) or row["available_mib"] != sample.available_mib:
                raise ResourceLimitError("resource headroom differs from its raw counters")
            if sample.available_mib < 512:
                raise ResourceLimitError("resource receipt falls below the 512 MiB floor")
        for first, second in pairwise(samples):
            if type(first) is not type(second) or pressure_counter(first) != pressure_counter(second):
                raise ResourceLimitError("resource platform or pressure counter changed")
            _stable_resource_identity(first, second)
        if any(b - a < 1_000_000 for a, b in pairwise(times)):
            raise ResourceLimitError("resource sample timestamps must be at least one second apart")
        return pressure_counter(samples[-1])
    except ContractError as exc:
        raise ResourceLimitError("malformed resource receipt") from exc


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


def sample_memory() -> MemorySample | KaggleSample:
    """Collect one sample on the supported host; unavailable is not admission."""
    if sys.platform == "linux":
        from frem_s10.runtime.kaggle_contract import active

        if active():
            return sample_kaggle_memory()
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
    if sys.platform == "linux":
        from frem_s10.runtime.kaggle_contract import active

        if active():
            kaggle_samples = [sample_kaggle_memory()]
            for _ in range(2):
                time.sleep(1)
                kaggle_samples.append(sample_kaggle_memory())
            return assess_kaggle(tuple(kaggle_samples))
        linux_samples = [sample_linux_memory()]
        for _ in range(2):
            time.sleep(1)
            linux_samples.append(sample_linux_memory())
        return assess_linux(tuple(linux_samples))
    samples = [sample_memory()]
    for _ in range(2):
        time.sleep(1)
        samples.append(sample_memory())
    return assess(tuple(samples))
