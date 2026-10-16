"""Export only verified generated-run artifacts, never repository or market data."""

import zipfile
from pathlib import Path

from frem_s10.errors import IntegrityError
from frem_s10.identities import JSONObject, canonical_bytes, digest_bytes
from frem_s10.runtime.artifacts import _plain_directory
from frem_s10.runtime.journal import inspect_journal
from frem_s10.runtime.smoke import _check_sessions, _evaluate_owned, _threads
from frem_s10.runtime.smoke_checkpoint import read_plain, run_lock


def export_smoke(directory: Path, destination: Path) -> JSONObject:
    """Create one new local ZIP after reconstructing all committed scores."""
    _threads()
    _plain_directory(destination.absolute().parent)
    with run_lock(directory):
        _check_sessions(directory)
        report = _evaluate_owned(directory)
        if report["status"] != "COMPLETE":
            raise IntegrityError("export requires a complete generated exercise")
        names, inventory = _inventory(directory)
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(names):
                archive.write(directory / name, name)
            archive.writestr("evaluation.json", canonical_bytes(report) + b"\n")
            archive.writestr("inventory.json", canonical_bytes(inventory) + b"\n")
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise IntegrityError("export archive failed CRC verification")
            for name, identity in inventory.items():
                if digest_bytes(archive.read(name)) != identity:
                    raise IntegrityError("export member differs from verified generated artifact")
    return {
        "status": "COMPLETE",
        "archive": str(destination),
        "artifacts": len(names),
        "bytes": destination.stat().st_size,
        "evidence_scope": "GENERATED_ENGINEERING",
    }


def _inventory(directory: Path) -> tuple[set[str], JSONObject]:
    names = {"run.json", "identity.json"}
    for path in directory.glob("step-*.json"):
        commit = read_plain(path)
        names.update((path.name, str(commit["path"])))
    for path in directory.glob("metrics-*.jsonl"):
        inspect_journal(path)
        names.add(path.name)
    names.update(path.name for path in directory.glob("status-*.json"))
    names.update(path.name for path in directory.glob("session-*.json"))
    inventory: JSONObject = {}
    total = 0
    for name in sorted(names):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise IntegrityError("export only accepts regular generated artifact files")
        total += path.stat().st_size
        if total > 256 * 2**20:
            raise IntegrityError("generated export exceeds 256 MiB bound")
        inventory[name] = digest_bytes(path.read_bytes())
    return names, inventory
