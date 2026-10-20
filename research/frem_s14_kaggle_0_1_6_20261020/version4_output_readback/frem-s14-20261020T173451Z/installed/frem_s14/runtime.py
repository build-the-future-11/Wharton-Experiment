"""Pre-import admission, durable manifests, explicit failures, no overwrite."""

import json
import math
import os
import platform
import re
import stat
import subprocess
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamps must include an explicit timezone")
    delta = dt.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def retain_input(source, destination):
    """Retain exact input bytes and their supplier manifest before fitting."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError("input must be a regular file")
    metadata = read_json(str(source) + ".manifest.json")
    expected = digest(source)
    if metadata.get("sha256") != expected:
        raise ValueError("input manifest checksum differs")
    with source.open("rb") as src, destination.open("xb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    if digest(destination) != expected or digest(source) != expected:
        raise ValueError("input changed during retention")
    write_json(str(destination) + ".manifest.json", metadata)
    return destination


def validate_forecast_clock(observed, decisions, origins):
    if len(observed) != len(decisions):
        raise ValueError("observation and decision clocks differ")
    if any(decisions[o] < observed[o] or decisions[o] >= observed[o + 1] for o in origins):
        raise ValueError("decision occurs after a target return; use an explicit as-of adapter")


def identity(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return decode_json(Path(path).read_text())


def decode_json(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            reject(value)
        return number

    return json.loads(
        text, object_pairs_hook=unique, parse_constant=reject, parse_float=finite_float
    )


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
        all(
            type(x["available_mib"]) in (int, float)
            and math.isfinite(x["available_mib"])
            and x["available_mib"] >= 512
            and type(x["pageouts"]) is int
            and x["pageouts"] >= 0
            for x in samples
        )
        and len({x["pageouts"] for x in samples}) == 1
    )
    return {
        "passed": passed,
        "minimum_mib": 512,
        "samples": samples,
        "time": datetime.now(timezone.utc).isoformat(),
    }


def manifest(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("result root must be a real directory")
    files = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("symlinks and special files are forbidden in result bundles")
        if stat.S_ISREG(mode) and path != root / "manifest.json":
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


def resume_seconds(checkpoint, maximum):
    """Retain and verify the prior attempt before charging its consumed time."""
    prior = Path(checkpoint).parent.parent
    verify(prior)
    if (prior / "status.json").exists():
        raise ValueError("completed runs cannot be resumed")
    seconds = read_json(prior / "failure.json")["wall_seconds"]
    if (
        type(seconds) not in (float, int)
        or not math.isfinite(seconds)
        or not 0 <= seconds < maximum
    ):
        raise ValueError("invalid or exhausted prior campaign budget")
    return seconds


def cell_inventory(config, complete, active=None):
    expected = [(arm, seed) for arm in config.arms for seed in config.seeds]
    completed = [(row["arm"], row["seed"]) for row in complete]
    if len(set(completed)) != len(completed) or set(completed) - set(expected):
        raise ValueError("duplicate or unexpected completed cell")
    if active is not None and (active not in expected or active in completed):
        raise ValueError("invalid failed cell")
    return [
        {
            "arm": arm,
            "seed": seed,
            "status": "COMPLETE"
            if (arm, seed) in completed
            else "FAILED"
            if (arm, seed) == active
            else "NOT_STARTED",
        }
        for arm, seed in expected
    ]


def verify_completion(root, config):
    root = Path(root)
    receipt = read_json(root / "status.json")
    expected = [{"arm": a, "seed": s} for a in config.arms for s in config.seeds]
    if (
        (root / "failure.json").exists()
        or receipt.get("status") != "COMPLETE"
        or receipt.get("protected_access") is not False
        or receipt.get("expected_cells") != len(expected)
        or receipt.get("complete_cells") != expected
        or read_json(root / "cells.json") != cell_inventory(config, expected)
    ):
        raise ValueError("incomplete, reordered or contradictory cell receipt")
    return receipt
