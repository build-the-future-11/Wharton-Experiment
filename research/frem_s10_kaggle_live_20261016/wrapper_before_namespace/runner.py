"""CPU-only, finite Kaggle engineering wrapper around the sealed S10 CLI.

No dependency installation, network client, numerical import, or work at import.
The retrieval receipt is an explicit operator attestation, not a remote signature.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DEADLINE = "2026-10-16T16:30:00+00:00"
BUNDLE_SHA256 = "a03c0e1886ece933427b554c555d2e914dff7f9d8df10e5c6caca94a74b2237a"
CANDIDATE_IDENTITY = "6cc8dde2716cfd6085a31d28ee0f9a4cb22adbfa1e6619918a0e6eaa3071a031"
CANDIDATE_DOCUMENT_SHA256 = "578a1137568a750a837468eb731d9a338c03b982540ef03b175abe35f82a685e"
SOURCE_INVENTORY_SHA256 = "826a28bd213f96d496cbcc2b7df004decda0e8e6bc5ecb65d611a9e05da4f697"
TOTAL_RESERVED_SECONDS = 600
CLEANUP_SECONDS = 5
HOST_TIMEOUT_SECONDS = 8
NUMERICAL_TIMEOUT_SECONDS = 75
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def utc():
    return datetime.now(timezone.utc)


def deadline_check(seconds=0, now=None):
    now = now or utc()
    if now.tzinfo is None or seconds < 0:
        raise ValueError("aware clock and nonnegative duration required")
    if (datetime.fromisoformat(DEADLINE) - now).total_seconds() <= seconds + CLEANUP_SECONDS:
        raise RuntimeError("tonight deadline cannot fit command and cleanup; no launch")


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 2**20:
        raise RuntimeError("bounded plain JSON document required")
    return json.loads(path.read_bytes(), object_pairs_hook=unique_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def publish(path, value):
    payload = encoded(value)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise RuntimeError("immutable evidence collision: " + path.name)
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def safe_archive_members(archive, *, maximum=256 * 2**20):
    infos = archive.infolist()
    if len(infos) > 4096 or sum(x.file_size for x in infos) > maximum:
        raise RuntimeError("archive exceeds bounded inventory")
    names = set()
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (not path.parts or path.is_absolute() or ".." in path.parts
                or str(path) != info.filename or "\\" in info.filename
                or info.filename in names or info.is_dir()
                or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                or info.flag_bits & 1):
            raise RuntimeError("unsafe or duplicate archive member")
        names.add(info.filename)
    return infos


def extract_verified(archive_path, destination, expected_sha, *, inventory_name):
    if archive_path.is_symlink() or sha(archive_path.read_bytes()) != expected_sha:
        raise RuntimeError("archive hash mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        infos = safe_archive_members(archive)
        inventory = json.loads(archive.read(inventory_name))
        names = {i.filename for i in infos}
        if names != set(inventory) | {inventory_name}:
            raise RuntimeError("archive and exact inventory disagree")
        for info in infos:
            payload = archive.read(info)
            if info.filename != inventory_name and sha(payload) != inventory[info.filename]:
                raise RuntimeError("archive member hash mismatch")
        if destination.exists():
            observed = set()
            for path in destination.rglob("*"):
                if path.is_symlink():
                    raise RuntimeError("extracted tree contains a symlink")
                if path.is_file():
                    name = path.relative_to(destination).as_posix()
                    observed.add(name)
                    if name not in names or sha(path.read_bytes()) != sha(archive.read(name)):
                        raise RuntimeError("existing extracted source differs")
            if observed != names:
                raise RuntimeError("existing extracted source inventory differs")
            return
        destination.mkdir(parents=True, exist_ok=False)
        for info in infos:
            path = destination / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(archive.read(info))


def verify_source(source):
    if sha((source / "SOURCE_INVENTORY.json").read_bytes()) != SOURCE_INVENTORY_SHA256:
        raise RuntimeError("sealed inventory bytes differ")
    inventory = read_json(source / "SOURCE_INVENTORY.json")
    actual = {p.relative_to(source).as_posix(): sha(p.read_bytes())
              for p in source.rglob("*") if p.is_file() and not p.is_symlink()}
    if any(p.is_symlink() for p in source.rglob("*")):
        raise RuntimeError("source symlink refused")
    actual.pop("SOURCE_INVENTORY.json", None)
    if actual != inventory:
        raise RuntimeError("sealed source inventory mismatch")
    candidate = source / "research/frem_s10_kaggle_20261016/CANDIDATE_IDENTITY.json"
    if sha(candidate.read_bytes()) != CANDIDATE_DOCUMENT_SHA256:
        raise RuntimeError("candidate document mismatch")
    if read_json(candidate)["code_identity"] != CANDIDATE_IDENTITY:
        raise RuntimeError("candidate identity mismatch")


def cpu_only(accelerator):
    if accelerator != "None":
        raise RuntimeError("accelerator None required; S10 has no GPU backend")
    if platform.system() != "Linux":
        raise RuntimeError("Kaggle execution requires Linux; local tests must not launch this path")
    if Path("/dev/nvidia0").exists() or list(Path("/proc/driver/nvidia/gpus").glob("*")):
        raise RuntimeError("GPU device visible; stop platform session and select accelerator None")


def declared_command_reservation():
    metadata = (10 + CLEANUP_SECONDS) + (30 + CLEANUP_SECONDS)
    probe = metadata + (HOST_TIMEOUT_SECONDS + CLEANUP_SECONDS) + (10 + CLEANUP_SECONDS)
    stages = 5 * (HOST_TIMEOUT_SECONDS + NUMERICAL_TIMEOUT_SECONDS + 2 * CLEANUP_SECONDS)
    return {"probe_seconds": probe, "smoke_seconds": metadata + stages,
            "total_seconds": probe + metadata + stages, "cap_seconds": TOTAL_RESERVED_SECONDS}


def compatibility_metadata():
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    reasons = []
    if sys.version_info[:2] != (3, 12):
        reasons.append("PYTHON_3_12_REQUIRED")
    if numpy_version != "2.5.4":
        reasons.append("NUMPY_2_5_4_METADATA_REQUIRED")
    return {"status": "BLOCKED" if reasons else "READY_FOR_CPU_SMOKE_METADATA_ONLY",
            "python": platform.python_version(), "numpy": numpy_version, "reasons": reasons,
            "runtime_import_tested": False, "scientific_status": "NOT_EVALUATED",
            "source": "stdlib wrapper compatibility gate before importing Python3.12 package"}


class Runner:
    def __init__(self, source, session):
        self.source, self.session = source.resolve(), session.resolve()
        self.session.mkdir(parents=True, exist_ok=True)
        self.control = self.session / "control"
        self.control.mkdir(exist_ok=True)
        self.helper_sha = sha(Path(__file__).read_bytes())
        self.binding = {"source_bundle_sha256": BUNDLE_SHA256,
                        "candidate_identity": CANDIDATE_IDENTITY,
                        "runner_sha256": self.helper_sha, "deadline": DEADLINE,
                        "total_reserved_seconds_cap": TOTAL_RESERVED_SECONDS}
        publish(self.control / "binding.json", self.binding)
        bootstraps = list(self.source.parent.glob("bootstrap-*.json"))
        if bootstraps:
            for bootstrap in bootstraps:
                if read_json(bootstrap).get("reserved_seconds") != 60:
                    raise RuntimeError("invalid notebook setup reservation")
                publish(self.control / bootstrap.name, read_json(bootstrap))
        elif not list(self.control.glob("bootstrap-*.json")):
            publish(self.control / "bootstrap-direct.json", {"reserved_seconds": 60})

    def reserved(self):
        starts = sorted(self.control.glob("command-*-start.json"))
        for path in starts:
            end = path.with_name(path.name.replace("-start", "-end"))
            if not end.is_file():
                raise RuntimeError("abandoned command; unknown termination requires review, no retry")
            if read_json(end).get("accepted") is not True:
                raise RuntimeError("prior failed command retained; automatic retry forbidden")
        inherited = self.control / "inherited-reservation.json"
        base = read_json(inherited)["seconds"] if inherited.exists() else 0
        return base + sum(read_json(p)["reserved_seconds"] for p in starts)

    def wall_spent(self):
        total = sum(read_json(p)["reserved_seconds"] for p in self.control.glob("bootstrap-*.json"))
        inherited = self.control / "inherited-reservation.json"
        if inherited.exists():
            total += read_json(inherited)["wall_seconds"]
        for path in self.control.glob("invocation-*-start.json"):
            end = path.with_name(path.name.replace("-start", "-end"))
            if not end.exists():
                raise RuntimeError("abandoned wrapper invocation; review required")
            total += read_json(end)["charged_seconds"]
        return total

    def run(self, name, tail, *, timeout=10, expected=(0,), raw=False):
        verify_source(self.source)
        deadline_check(timeout)
        reserved = timeout + CLEANUP_SECONDS
        if self.reserved() + reserved > TOTAL_RESERVED_SECONDS:
            raise RuntimeError("conservative CPU command reservation cap exhausted")
        key = uuid.uuid4().hex
        log = self.control / f"command-{key}.log"
        argv = [sys.executable, "-u", "-B"] + (tail if raw else ["-m", "frem_s10", *tail])
        start = {"name": name, "argv_tail": tail, "utc": utc().isoformat(),
                 "reserved_seconds": reserved, "timeout_seconds": timeout,
                 "binding": self.binding}
        publish(self.control / f"command-{key}-start.json", start)
        env = os.environ.copy()
        env.update({k: "1" for k in THREADS})
        env.update(PYTHONPATH=str(self.source / "successors/frem_s10/src"),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        env.pop("PYTHONSTARTUP", None)
        begun, child, failure, caught = time.monotonic(), None, None, None
        cleanup_errors = []
        with log.open("xb") as stream:
            try:
                deadline_check(timeout)
                child = subprocess.Popen(argv, cwd=self.source, env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                assert child.stdout is not None
                with selectors.DefaultSelector() as selector:
                    selector.register(child.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        if time.monotonic() - begun >= timeout:
                            raise subprocess.TimeoutExpired(argv, timeout)
                        for event_key, _ in selector.select(timeout=0.1):
                            chunk = os.read(event_key.fileobj.fileno(), 65536)
                            if not chunk:
                                selector.unregister(event_key.fileobj)
                                continue
                            stream.write(chunk)
                            stream.flush()
                            try:
                                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                                sys.stdout.flush()
                            except (BrokenPipeError, OSError):
                                pass  # Durable log still receives every byte.
                child.wait(timeout=max(0.01, timeout - (time.monotonic() - begun)))
            except BaseException as exc:
                failure = type(exc).__name__
                caught = exc
                if child is not None:
                    # A reaped leader may leave descendants holding stdout open.
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    except OSError as cleanup_error:
                        cleanup_errors.append(str(cleanup_error))
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError as cleanup_error:
                        cleanup_errors.append(str(cleanup_error))
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        cleanup_errors.append("child remained alive after cleanup deadline")
            finally:
                if child is not None and child.stdout is not None:
                    child.stdout.close()
                stream.flush()
                os.fsync(stream.fileno())
        result = {"name": name, "returncode": child.returncode if child else None,
                  "failure": failure, "cleanup_errors": cleanup_errors, "elapsed_seconds": time.monotonic() - begun,
                  "utc": utc().isoformat(), "log": log.name,
                  "log_sha256": sha(log.read_bytes()), "binding": self.binding}
        result["accepted"] = failure is None and result["returncode"] in expected
        publish(self.control / f"command-{key}-end.json", result)
        print(json.dumps(result), flush=True)
        if caught is not None:
            raise caught
        if result["returncode"] not in expected:
            raise RuntimeError(f"{name} stopped: see {log.name}; automatic retry forbidden")
        return log

    def metadata(self):
        compatibility = compatibility_metadata()
        if compatibility["status"] == "BLOCKED":
            print(json.dumps(compatibility), flush=True)
            return compatibility, None
        preflight = read_json(self.run("preflight", ["preflight", "--device", "cpu"], expected=(0, 3)))
        if preflight["status"] != "READY_FOR_CPU_SMOKE_METADATA_ONLY":
            return preflight, None
        code = "import json; from frem_s10.runtime.smoke_contract import runtime_identity; print(json.dumps(runtime_identity(),allow_nan=False))"
        identity = read_json(self.run("runtime-identity", ["-c", code], timeout=30, raw=True))
        return preflight, identity

    def probe(self):
        path = self.session / "persistence-probe.json"
        if path.exists():
            existing = read_json(path)
            if existing["binding"] != self.binding:
                raise RuntimeError("existing probe binding differs")
            return existing
        preflight, identity = self.metadata()
        host = (read_json(self.run("probe-host", ["check-host"], timeout=HOST_TIMEOUT_SECONDS, expected=(0, 3)))
                if sys.version_info[:2] == (3, 12) else
                {"status": "NOT_RUN_PYTHON_INCOMPATIBLE", "reason": "maintained package requires Python3.12; no unsupported package import"})
        manifest = self.source / "configs/generated_smoke.json"
        plan = (read_json(self.run("smoke-plan", ["smoke-plan", "--manifest", str(manifest)]))
                if identity is not None else {"status": "SEALED_MANIFEST_READ_ONLY_CLI_NOT_RUN",
                                             "configuration": read_json(manifest),
                                             "configuration_bytes_sha256": sha(manifest.read_bytes())})
        probe = {"schema": "s10-tonight-persistence-probe-v1", "nonce": uuid.uuid4().hex,
                 "created_utc": utc().isoformat(), "binding": self.binding,
                 "preflight": preflight, "runtime_identity": identity,
                 "host_admission": host,
                 "command_reservation_plan": declared_command_reservation(),
                 "plan": plan, "reserved_seconds": self.reserved(),
                 "wrapper_reserved_seconds": self.probe_wall_upper,
                 "scientific_status": "NOT_EVALUATED", "gpu_execution": False,
                 "status": "PERSISTENCE_RETRIEVAL_REQUIRED"}
        publish(path, probe)
        return probe

    def smoke(self, receipt):
        proof = validate_receipt(receipt, helper_sha=self.helper_sha)
        saved = self.control / "retrieval-receipt.json"
        first = not saved.exists()
        if first:
            if (self.session / "smoke").exists():
                raise RuntimeError("unbound smoke directory exists; refuse adoption")
            if not (self.session / "persistence-probe.json").exists():
                publish(self.control / "inherited-reservation.json", {"seconds": proof["reserved_seconds"], "wall_seconds": proof["wrapper_reserved_seconds"]})
            publish(saved, receipt)
        elif read_json(saved) != receipt:
            raise RuntimeError("retrieval receipt changed")
        preflight, identity = self.metadata()
        if preflight["status"] != "READY_FOR_CPU_SMOKE_METADATA_ONLY" or identity != proof["runtime_identity"]:
            raise RuntimeError("retrieved probe and current exact CPU runtime do not match")
        manifest, run = self.source / "configs/generated_smoke.json", self.session / "smoke"
        stages = [
            ("interrupt", ["smoke", "--manifest", str(manifest), "--directory", str(run), "--stop-after", "3", "--execute"], (130,)),
            ("resume", ["smoke-resume", "--manifest", str(manifest), "--directory", str(run), "--execute"], (0,)),
            ("evaluate", ["smoke-evaluate", "--directory", str(run)], (0,)),
            ("completed-reuse", ["smoke-resume", "--manifest", str(manifest), "--directory", str(run), "--execute"], (0,)),
            ("export", ["smoke-export", "--directory", str(run), "--archive", str(self.session / "smoke-export.zip")], (0,)),
        ]
        if (self.control / "stage-export.json").exists():
            raise RuntimeError("completed campaign retained; use status/verified output, no repeated campaign")
        for name, args, expected in stages:
            stage = self.control / f"stage-{name}.json"
            if stage.exists():
                value = read_json(stage)
                log = self.control / value["log"]
                if value["binding"] != self.binding or sha(log.read_bytes()) != value["log_sha256"]:
                    raise RuntimeError("completed stage evidence changed")
                validate_stage_result(name, value["result"])
                continue
            host = read_json(self.run("host-" + name, ["check-host"], timeout=HOST_TIMEOUT_SECONDS))
            if host.get("status") != "PASS":
                raise RuntimeError("fresh host admission failed; no numerical import")
            log = self.run(name, args, timeout=NUMERICAL_TIMEOUT_SECONDS, expected=expected)
            result = final_cli_result(log.read_text())
            validate_stage_result(name, result)
            publish(stage, {"binding": self.binding, "log": log.name, "log_sha256": sha(log.read_bytes()), "result": result})
            self.snapshot()
        outcome = {"status": "GENERATED_CPU_STAGES_COMPLETE_WITH_FINAL_EXPORT_PENDING", "scientific_status": "NOT_EVALUATED",
                   "origins": 12, "variant": "primary", "seed": 11,
                   "gpu_execution": False, "production_qualified": False,
                   "binding": self.binding, "reserved_seconds": self.reserved()}
        publish(self.session / "ENGINEERING_RECEIPT.json", outcome)
        return outcome

    def finalize(self, result):
        export = self.snapshot()
        deadline_check()
        if result and result.get("status") == "GENERATED_CPU_STAGES_COMPLETE_WITH_FINAL_EXPORT_PENDING":
            final = {"status": "GENERATED_CPU_ENGINEERING_COMPLETE", "scientific_status": "NOT_EVALUATED",
                     "binding": self.binding, "full_state_export": export,
                     "platform_session_stopped": "NOT_OBSERVED_REQUIRES_KAGGLE_UI"}
            publish(self.session / "FINALIZATION_RECEIPT.json", final)
            print(json.dumps(final), flush=True)
        return export

    def snapshot(self):
        exports = self.session.parent / "exports"
        exports.mkdir(exist_ok=True)
        inventory = {}
        paths = []
        for path in sorted(self.session.rglob("*")):
            if path.is_symlink():
                raise RuntimeError("state symlink refused")
            if path.is_file() and path.relative_to(self.session).as_posix() not in (".runner.lock", "STATE_INVENTORY.json"):
                paths.append(path)
                inventory[path.relative_to(self.session).as_posix()] = sha(path.read_bytes())
        if sum(p.stat().st_size for p in paths) > 256 * 2**20:
            raise RuntimeError("state export exceeds bound")
        target = exports / ("state-" + uuid.uuid4().hex + ".zip")
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, path.relative_to(self.session).as_posix())
            archive.writestr("STATE_INVENTORY.json", encoded(inventory))
        record = {"archive": target.name, "sha256": sha(target.read_bytes()), "files": len(paths),
                  "binding": self.binding, "created_utc": utc().isoformat(),
                  "resumable": all(p.with_name(p.name.replace("-start", "-end")).is_file()
                                   for p in self.control.glob("invocation-*-start.json")),
                  "restore_note": "Only a terminal snapshot can resume; active invocation snapshots retain unknown time."}
        publish(exports / (target.stem + ".json"), record)
        print(json.dumps({"saved_full_state": record}), flush=True)
        return record


def validate_receipt(receipt, *, helper_sha):
    if set(receipt) != {"schema", "kernel_version", "retrieved_utc", "probe_sha256", "probe", "operator_attestation"}:
        raise RuntimeError("exact external retrieval receipt schema required")
    if receipt["schema"] != "s10-tonight-external-retrieval-v1" or receipt["operator_attestation"] is not True:
        raise RuntimeError("explicit external retrieval attestation required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[1-9][0-9]*", receipt["kernel_version"]):
        raise RuntimeError("exact private notebook owner/slug/version required")
    probe = receipt["probe"]
    if probe.get("schema") != "s10-tonight-persistence-probe-v1" or not re.fullmatch(r"[0-9a-f]{32}", str(probe.get("nonce"))):
        raise RuntimeError("exact persistence probe schema and nonce required")
    if sha(encoded(probe)) != receipt["probe_sha256"]:
        raise RuntimeError("retrieved probe digest mismatch")
    binding = probe["binding"]
    if (binding["source_bundle_sha256"] != BUNDLE_SHA256 or binding["candidate_identity"] != CANDIDATE_IDENTITY
            or binding["runner_sha256"] != helper_sha or binding["deadline"] != DEADLINE
            or binding["total_reserved_seconds_cap"] != TOTAL_RESERVED_SECONDS):
        raise RuntimeError("retrieval receipt source/runner/deadline binding mismatch")
    created, retrieved = datetime.fromisoformat(probe["created_utc"]), datetime.fromisoformat(receipt["retrieved_utc"])
    if created.tzinfo is None or retrieved.tzinfo is None or not created <= retrieved <= utc():
        raise RuntimeError("invalid retrieval chronology")
    if type(probe["reserved_seconds"]) is not int or not 0 <= probe["reserved_seconds"] < TOTAL_RESERVED_SECONDS:
        raise RuntimeError("invalid inherited CPU charge")
    if type(probe.get("wrapper_reserved_seconds")) is not int or not 180 <= probe["wrapper_reserved_seconds"] < TOTAL_RESERVED_SECONDS:
        raise RuntimeError("bounded probe wrapper and bootstrap reservation required")
    return probe


def final_cli_result(text):
    decoder = json.JSONDecoder()
    found = None
    for index, character in enumerate(text):
        if character == "{":
            try:
                value, end = decoder.raw_decode(text, index)
            except ValueError:
                continue
            if isinstance(value, dict) and not text[end:].strip():
                found = value
    if found is None:
        raise RuntimeError("missing final maintained CLI JSON result")
    return found


def validate_stage_result(name, result):
    expected = {"interrupt": "INTERRUPTED", "resume": "COMPLETE", "evaluate": "COMPLETE",
                "completed-reuse": "COMPLETE_REUSED", "export": "COMPLETE"}[name]
    if result.get("status") != expected:
        raise RuntimeError("stage status does not match declared operation")
    if name != "export":
        if result.get("completed") != (3 if name == "interrupt" else 12) or result.get("planned") != 12:
            raise RuntimeError("stage origin boundary mismatch")
        if result.get("scientific_status") != "NOT_EVALUATED":
            raise RuntimeError("stage scientific scope changed")
    if name == "completed-reuse" and result.get("validated") is not True:
        raise RuntimeError("completed cell was not validated")
    if name in ("evaluate", "export") and result.get("evidence_scope") != "GENERATED_ENGINEERING":
        raise RuntimeError("stage evidence scope changed")


def alarm_expired(_signal, _frame):
    raise TimeoutError("overall active wall/deadline watchdog expired")


class TerminationRequested(RuntimeError):
    """Do not subclass InterruptedError: selectors may swallow that exception."""


def termination_requested(_signal, _frame):
    raise TerminationRequested("platform/outer supervisor requested termination; child cleanup required")


def attest(downloaded_probe, kernel_version, output, acknowledged, expected_sha=None):
    if not acknowledged:
        raise RuntimeError("operator must explicitly attest external private-version retrieval")
    probe = read_json(downloaded_probe)
    if not isinstance(expected_sha, str) or sha(downloaded_probe.read_bytes()) != expected_sha:
        raise RuntimeError("downloaded probe differs from externally observed producer digest")
    if downloaded_probe.read_bytes() != encoded(probe):
        raise RuntimeError("downloaded probe bytes differ from canonical producer output")
    receipt = {"schema": "s10-tonight-external-retrieval-v1", "kernel_version": kernel_version,
               "retrieved_utc": utc().isoformat(), "probe_sha256": sha(downloaded_probe.read_bytes()),
               "probe": probe, "operator_attestation": True}
    validate_receipt(receipt, helper_sha=sha(Path(__file__).read_bytes()))
    publish(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "smoke", "status", "attest"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--accelerator", default="None")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--downloaded-probe", type=Path)
    parser.add_argument("--kernel-version")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ack-retrieved", action="store_true")
    parser.add_argument("--expected-probe-sha256")
    args = parser.parse_args()
    if args.mode == "attest":
        print(json.dumps(attest(args.downloaded_probe, args.kernel_version, args.output, args.ack_retrieved, args.expected_probe_sha256), indent=2))
        return 0
    if args.mode == "status":
        for path in sorted(args.session.glob("*RECEIPT.json")):
            print(json.dumps(read_json(path), indent=2))
        print("Read-only saved evidence; platform session termination must be checked in Kaggle UI.")
        return 0
    deadline_check()
    cpu_only(args.accelerator)
    signal.signal(signal.SIGALRM, alarm_expired)
    signal.signal(signal.SIGTERM, termination_requested)
    signal.setitimer(signal.ITIMER_REAL, min(10, (datetime.fromisoformat(DEADLINE) - utc()).total_seconds() - CLEANUP_SECONDS))
    verify_source(args.source)
    runner = Runner(args.source, args.session)
    with (runner.session / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_wall = runner.wall_spent()
        remaining = TOTAL_RESERVED_SECONDS - previous_wall
        if args.mode == "smoke" and not (runner.session / "persistence-probe.json").exists() and not (runner.control / "inherited-reservation.json").exists():
            receipt_for_budget = read_json(args.receipt)
            proof_for_budget = validate_receipt(receipt_for_budget, helper_sha=runner.helper_sha)
            remaining -= proof_for_budget["wrapper_reserved_seconds"]
        allowance = min(110 if args.mode == "probe" else remaining - 10, remaining - 10,
                        (datetime.fromisoformat(DEADLINE) - utc()).total_seconds() - CLEANUP_SECONDS)
        if allowance <= 25:
            raise RuntimeError("insufficient wrapper wall budget including final export")
        invocation = uuid.uuid4().hex
        runner.probe_wall_upper = previous_wall + 120
        started = time.monotonic()
        publish(runner.control / f"invocation-{invocation}-start.json", {"utc": utc().isoformat(), "upper_seconds": allowance})
        signal.signal(signal.SIGALRM, alarm_expired)
        signal.setitimer(signal.ITIMER_REAL, allowance - 20)
        result = None
        try:
            result = runner.probe() if args.mode == "probe" else runner.smoke(read_json(args.receipt))
            print(json.dumps(result, indent=2, allow_nan=False), flush=True)
            if args.mode == "probe":
                print(json.dumps({"probe_sha256": sha(encoded(result)), "path": "session/persistence-probe.json"}), flush=True)
        finally:
            elapsed = time.monotonic() - started
            publish(runner.control / f"invocation-{invocation}-end.json", {"charged_seconds": math.ceil(elapsed) + 30, "elapsed_seconds": elapsed,
                    "includes_setup_reservation_seconds": 10, "includes_final_export_reservation_seconds": 20})
            export_remaining = min(20, allowance - elapsed,
                                   (datetime.fromisoformat(DEADLINE) - utc()).total_seconds() - CLEANUP_SECONDS)
            if export_remaining > 0:
                signal.setitimer(signal.ITIMER_REAL, export_remaining)
                try:
                    runner.finalize(result)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            else:
                signal.setitimer(signal.ITIMER_REAL, 0)
                raise RuntimeError("deadline leaves no export time; no final platform completion claim")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__, "reason": str(exc),
                          "scientific_status": "NOT_EVALUATED"}), flush=True)
        raise SystemExit(2)
