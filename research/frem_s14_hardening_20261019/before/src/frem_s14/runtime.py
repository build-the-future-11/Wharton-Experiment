"""Pre-import admission, durable manifests, explicit failures, no overwrite."""

import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamps must include an explicit timezone")
    return int(dt.timestamp() * 1_000_000)


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def memory_snapshot():
    if platform.system() == "Darwin":
        raw = subprocess.check_output(["vm_stat"], text=True)
        page = int(re.search(r"page size of (\d+)", raw)[1])
        counts = {k: int(v) for k, v in re.findall(r"^([^:\n]+):\s+(\d+)\.", raw, re.M)}
        return {
            "available_mib": (counts["Pages free"] + counts["Pages speculative"]) * page / 2**20,
            "pageouts": counts["Pageouts"],
        }
    if platform.system() == "Linux":
        info = dict(re.findall(r"^(\w+):\s+(\d+)", Path("/proc/meminfo").read_text(), re.M))
        available = int(info["MemAvailable"]) / 1024
        # Respect the effective cgroup v2 hierarchy, not only host MemAvailable.
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and "memory" in parts[1].split(","):
                group = Path("/sys/fs/cgroup/memory") / parts[2].lstrip("/")
                for parent in (group, *group.parents):
                    if str(parent).startswith("/sys/fs/cgroup/memory"):
                        maximum, current = (
                            parent / "memory.limit_in_bytes",
                            parent / "memory.usage_in_bytes",
                        )
                        if maximum.exists() and current.exists():
                            available = min(
                                available,
                                (int(maximum.read_text()) - int(current.read_text())) / 2**20,
                            )
            if line.startswith("0::"):
                group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                for parent in (group, *group.parents):
                    if str(parent).startswith("/sys/fs/cgroup"):
                        maximum, current = parent / "memory.max", parent / "memory.current"
                        if maximum.exists() and current.exists():
                            limit = maximum.read_text().strip()
                            if limit != "max":
                                available = min(
                                    available, (int(limit) - int(current.read_text())) / 2**20
                                )
        vm = dict(re.findall(r"^(\w+)\s+(\d+)", Path("/proc/vmstat").read_text(), re.M))
        return {"available_mib": available, "pageouts": int(vm["pswpout"])}
    raise RuntimeError("only Linux/macOS admission is implemented")


def admit():
    samples = []
    for _ in range(3):
        samples.append(memory_snapshot())
        time.sleep(1)
    passed = (
        min(x["available_mib"] for x in samples) >= 512
        and samples[0]["pageouts"] == samples[-1]["pageouts"]
    )
    return {
        "passed": passed,
        "minimum_mib": 512,
        "samples": samples,
        "time": datetime.now(timezone.utc).isoformat(),
    }


def manifest(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinks are forbidden in result bundles")
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(root))] = {
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
    return {"schema": 1, "files": files}


def verify(root):
    saved = read_json(Path(root) / "manifest.json")
    if saved != manifest(root):
        raise ValueError("bundle inventory, size or hash mismatch")
    return {"verified": True, "files": len(saved["files"])}
