"""Offline exact-source wheel/build inspection and clean installed-package smoke.

Called only by the fixed engineering verifier after resource admission. All
commands inherit its denied-network/raw-data sandbox. No research is executed.
"""

import argparse
import base64
import configparser
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path

from frem_s10.errors import IntegrityError, S10Error
from frem_s10.identities import (
    JSONObject,
    canonical_bytes,
    checked_json,
    digest_bytes,
    object_json,
    parse_json,
)
from frem_s10.runtime.artifacts import _plain_directory, publish_json
from frem_s10.runtime.provenance import PACKAGE, expected_code_files, observe_environment

COMMAND_NAMES = ("build", "clean-environment", "install", "smoke", "help", "plan")

NUMPY_WHEEL = Path(
    "research/frem_s10_repair_20261015/package_verification_attempt4/numpy-2.5.4-cp312-cp312-macosx_14_0_arm64.whl"
)
NUMPY_SHA256 = "9cb18a327b49c5c337f972b03682f6a49855525faaf3c0d3e9c96cd0fd8880a8"
SMOKE = """import json, sys
import numpy
import frem_s10
import frem_s10.runtime.pipeline
import frem_s10.runtime.continuation
import frem_s10.experiment.model_archive
from importlib.resources import files
p = files("frem_s10")
s = json.loads(p.joinpath("science.json").read_text())
assert s["design_id"] == "FREM-S10-DESIGN-4-CL16"
assert "site-packages" in str(p) and "successors/frem_s10/src" not in str(p)
assert not ({"frem", "torch", "pandas", "sklearn"} & set(sys.modules))
print(json.dumps({"module": str(p), "numpy": numpy.__version__, "design_id": s["design_id"], "status": "PASS"}))
"""


def _run(name: str, command: tuple[str, ...], *, output: Path, cwd: Path) -> None:
    started = time.monotonic()
    failure: str | None = None
    with (output / f"{name}.log").open("xb") as log:
        try:
            completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            exit_code = completed.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            exit_code, failure = -1, f"{type(exc).__name__}: {exc}"
            log.write(f"\nPACKAGING_COMMAND_FAILED: {failure}\n".encode())
    publish_json(
        output,
        f"{name}.json",
        {
            "name": name,
            "argv": list(command),
            "cwd": str(cwd),
            "exit_code": exit_code,
            "wall_seconds": time.monotonic() - started,
            "failure": failure,
            "log_sha256": digest_bytes((output / f"{name}.log").read_bytes()),
        },
    )
    if exit_code:
        raise IntegrityError(f"offline packaging command failed: {name}")


def _safe_member(name: str) -> None:
    parts = Path(name).parts
    if not name or "\\" in name or "\x00" in name or Path(name).as_posix() != name.rstrip("/"):
        raise IntegrityError("canonical POSIX archive member required")
    if (
        Path(name).is_absolute()
        or ".." in parts
        or {".venv", ".env", "research", "outputs", ".git"} & set(parts)
    ):
        raise IntegrityError("package archive contains forbidden or escaping member")
    if Path(name).name in ("SPY.json", "TLT.json", "GLD.json"):
        raise IntegrityError("package must not distribute source market snapshots")


@dataclass(frozen=True)
class PackageContract:
    """The supported frozen distribution inputs, without inferred extra data files."""

    files: dict[str, bytes]
    payloads: dict[str, bytes]
    project: dict[str, object]
    root_name: str


def _contract(source: Path) -> PackageContract:
    paths = expected_code_files(source)
    base = source / PACKAGE
    files = {path.relative_to(base).as_posix(): path.read_bytes() for path in paths}
    project = tomllib.loads(files["pyproject.toml"].decode())["project"]
    if project["name"] != "frem-s10" or project["version"] != "0.1.0":
        raise IntegrityError("fixed S10 distribution identity required")
    if project.get("readme") != "README.md" or project.get("dependencies") != ["numpy==2.5.4"]:
        raise IntegrityError("fixed readme and dependency contract required")
    payloads = {name.removeprefix("src/"): data for name, data in files.items() if name.startswith("src/")}
    return PackageContract(files, payloads, project, "frem_s10-0.1.0")


def _members(archive: zipfile.ZipFile) -> dict[str, bytes]:
    rows = archive.infolist()
    if (
        len({row.filename for row in rows}) != len(rows)
        or len(rows) > 2048
        or sum(row.file_size for row in rows) > 16_777_216
    ):
        raise IntegrityError("duplicate or excessive wheel members")
    values: dict[str, bytes] = {}
    for row in rows:
        _safe_member(row.filename)
        kind = (row.external_attr >> 16) & 0o170000
        if row.is_dir() or kind not in (0, 0o100000) or row.file_size > 1_048_576:
            raise IntegrityError("wheel requires bounded regular file members")
        values[row.filename] = archive.read(row)
    return values


class _EntryParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _entry_points(raw: bytes, contract: PackageContract) -> None:
    parser = _EntryParser(interpolation=None, strict=True)
    try:
        parser.read_string(raw.decode())
    except (UnicodeError, configparser.Error) as exc:
        raise IntegrityError("malformed wheel entry points") from exc
    if (
        parser.defaults()
        or parser.sections() != ["console_scripts"]
        or dict(parser["console_scripts"]) != contract.project.get("scripts")
    ):
        raise IntegrityError("wheel entry points differ from frozen project scripts")
    expected = "[console_scripts]\n" + "".join(
        f"{name} = {value}\n" for name, value in sorted(parser["console_scripts"].items())
    )
    if raw != expected.encode():
        raise IntegrityError("unbound entry point metadata bytes are unsupported")


def _metadata(raw: bytes, contract: PackageContract) -> None:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    allowed = {
        "Metadata-Version",
        "Name",
        "Version",
        "Summary",
        "Requires-Python",
        "Description-Content-Type",
        "Requires-Dist",
    }
    if (
        set(message.keys()) != allowed
        or any(len(message.get_all(key, [])) != 1 for key in allowed)
        or message.defects
    ):
        raise IntegrityError("exact supported package metadata headers required")
    project = contract.project
    expected = {
        "Metadata-Version": "2.4",
        "Name": project["name"],
        "Version": project["version"],
        "Summary": project.get("description", ""),
        "Description-Content-Type": "text/markdown",
        "Requires-Dist": "numpy==2.5.4",
    }
    if any(message[key] != value for key, value in expected.items()):
        raise IntegrityError("package metadata differs from frozen project")
    if sorted(str(message["Requires-Python"]).split(",")) != sorted(
        str(project["requires-python"]).split(",")
    ):
        raise IntegrityError("package Python requirement differs from frozen project")
    if message.get_payload() != contract.files["README.md"].decode():
        raise IntegrityError("package description differs from frozen README")


def _wheel_metadata(values: dict[str, bytes], contract: PackageContract) -> bytes:
    prefix = f"{contract.root_name}.dist-info/"
    names = {prefix + name for name in ("METADATA", "WHEEL", "entry_points.txt", "top_level.txt", "RECORD")}
    if set(values) != set(contract.payloads) | names:
        raise IntegrityError("wheel inventory differs from exact package and five metadata files")
    if any(values[name] != payload for name, payload in contract.payloads.items()):
        raise IntegrityError("wheel source payload differs from frozen source")
    _metadata(values[prefix + "METADATA"], contract)
    _entry_points(values[prefix + "entry_points.txt"], contract)
    headers = BytesParser(policy=policy.default).parsebytes(values[prefix + "WHEEL"])
    expected = {
        "Wheel-Version": "1.0",
        "Root-Is-Purelib": "true",
        "Tag": "py3-none-any",
        "Generator": "setuptools (84.0.0)",
    }
    if (
        set(headers.keys()) != set(expected)
        or any(len(headers.get_all(key, [])) != 1 for key in headers)
        or any(headers[key] != value for key, value in expected.items())
        or headers.get_payload() != ""
    ):
        raise IntegrityError("unsupported wheel platform or install layout")
    if values[prefix + "top_level.txt"] != b"frem_s10\n":
        raise IntegrityError("wheel top-level import identity differs")
    _record(values, prefix + "RECORD")
    return values[prefix + "METADATA"]


def _record(values: dict[str, bytes], name: str) -> None:
    rows = list(csv.reader(io.StringIO(values[name].decode())))
    if len(rows) != len(values) or any(len(row) != 3 for row in rows):
        raise IntegrityError("wheel RECORD lacks exact complete rows")
    observed: set[str] = set()
    for member, identity, size in rows:
        if member in observed or member not in values:
            raise IntegrityError("duplicate or unbound wheel RECORD member")
        observed.add(member)
        digest = base64.urlsafe_b64encode(hashlib.sha256(values[member]).digest()).decode().rstrip("=")
        expected = ("", "") if member == name else (f"sha256={digest}", str(len(values[member])))
        if (identity, size) != expected:
            raise IntegrityError("wheel RECORD hash or size differs from exact payload")


def _sdist_values(path: Path, contract: PackageContract) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    seen: set[str] = set()
    total = 0
    with tarfile.open(path) as archive:
        for member in archive:
            _safe_member(member.name)
            parts = Path(member.name).parts
            if member.name in seen or len(seen) >= 2048 or parts[0] != contract.root_name:
                raise IntegrityError("duplicate, excessive or unbound source distribution member")
            seen.add(member.name)
            if not member.isfile() and not member.isdir():
                raise IntegrityError("source distribution requires regular files/directories")
            if member.isfile():
                total += member.size
                stream = archive.extractfile(member)
                if member.size > 1_048_576 or total > 16_777_216 or stream is None:
                    raise IntegrityError("source distribution payload exceeds bound or is unavailable")
                with stream:
                    values[Path(*parts[1:]).as_posix()] = stream.read()
    return values


def _sdist(path: Path, contract: PackageContract, metadata: bytes, entry_points: bytes) -> None:
    values = _sdist_values(path, contract)
    egg = "src/frem_s10.egg-info/"
    generated = {
        "PKG-INFO": metadata,
        "setup.cfg": b"[egg_info]\ntag_build = \ntag_date = 0\n\n",
        egg + "PKG-INFO": metadata,
        egg + "dependency_links.txt": b"\n",
        egg + "entry_points.txt": entry_points,
        egg + "requires.txt": b"numpy==2.5.4\n",
        egg + "top_level.txt": b"frem_s10\n",
    }
    source_names = (
        set(contract.files) | {name for name in generated if name.startswith(egg)} | {egg + "SOURCES.txt"}
    )
    if set(values) != set(contract.files) | set(generated) | {egg + "SOURCES.txt"}:
        raise IntegrityError("source distribution inventory differs from exact frozen build inventory")
    if any(values[name] != payload for name, payload in (contract.files | generated).items()):
        raise IntegrityError("source distribution payload differs from frozen source or fixed metadata")
    rows = values[egg + "SOURCES.txt"].decode().splitlines()
    if len(rows) != len(set(rows)) or set(rows) != source_names:
        raise IntegrityError("source manifest differs from exact frozen build inventory")


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inspect(wheel: Path, sdist: Path, source: Path) -> JSONObject:
    contract = _contract(source)
    if wheel.name != f"{contract.root_name}-py3-none-any.whl" or sdist.name != f"{contract.root_name}.tar.gz":
        raise IntegrityError("fixed platform-independent wheel and source archive names required")
    with zipfile.ZipFile(wheel) as archive:
        values = _members(archive)
    metadata = _wheel_metadata(values, contract)
    _sdist(sdist, contract, metadata, values[f"{contract.root_name}.dist-info/entry_points.txt"])
    return {
        "schema": "s10-package-inspection-v1",
        "status": "PASS",
        "source_payloads": checked_json(
            {name: digest_bytes(payload) for name, payload in contract.payloads.items()}
        ),
        "frozen_build_files": checked_json(
            {name: digest_bytes(payload) for name, payload in contract.files.items()}
        ),
        "wheel_sha256": _hash(wheel),
        "sdist_sha256": _hash(sdist),
    }


def verify_package(source_root: Path, output: Path) -> JSONObject:
    """Build and install only known local wheels, retaining every command and artifact."""
    source = _plain_directory(source_root)
    environment = observe_environment(source)
    if output.exists():
        raise IntegrityError("new packaging output directory required")
    output.mkdir()
    _plain_directory(output)
    try:
        return _verify_package(source, output, environment)
    except Exception as exc:
        publish_json(
            output,
            "failure.json",
            {
                "schema": "s10-package-failure-v1",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_execution": False,
            },
        )
        raise


def _verify_package(source: Path, output: Path, environment: JSONObject) -> JSONObject:
    numpy_wheel = source / NUMPY_WHEEL
    if _hash(numpy_wheel) != NUMPY_SHA256:
        raise IntegrityError("retained locked NumPy wheel is missing or changed")
    uv = shutil.which("uv")
    if uv is None:
        raise IntegrityError("observed offline installer unavailable")
    python = str(Path(sys.executable).absolute())
    _run(
        "build",
        (python, "-m", "build", "--no-isolation", "--outdir", str(output / "dist"), str(source / PACKAGE)),
        output=output,
        cwd=source,
    )
    wheels, sdists = list((output / "dist").glob("*.whl")), list((output / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise IntegrityError("exactly one wheel and source distribution required")
    inspection = _inspect(wheels[0], sdists[0], source)
    publish_json(output, "archive-inspection.json", inspection)
    _run(
        "clean-environment",
        (uv, "venv", "--offline", "--python", python, str(output / "clean_env")),
        output=output,
        cwd=output,
    )
    requirements = output / "requirements.txt"
    requirements.write_text(
        f"{wheels[0].as_uri()} --hash=sha256:{inspection['wheel_sha256']}\n{numpy_wheel.as_uri()} --hash=sha256:{NUMPY_SHA256}\n"
    )
    _run(
        "install",
        (
            uv,
            "pip",
            "install",
            "--python",
            str(output / "clean_env/bin/python"),
            "--offline",
            "--no-index",
            "--require-hashes",
            "-r",
            str(requirements),
        ),
        output=output,
        cwd=output,
    )
    _run("smoke", (str(output / "clean_env/bin/python"), "-I", "-B", "-c", SMOKE), output=output, cwd=output)
    _run("help", (str(output / "clean_env/bin/s10-research"), "--help"), output=output, cwd=output)
    _run(
        "plan",
        (
            str(output / "clean_env/bin/s10-research"),
            "plan",
            "--manifest",
            str(source / PACKAGE / "src/frem_s10/science.json"),
            "--output",
            str(output / "plan.json"),
        ),
        output=output,
        cwd=output,
    )
    receipt = inspection | {
        "clean_install": True,
        "network": "PARENT_VERIFIER_SANDBOX_PROOF_REQUIRED",
        "network_isolation_attested": False,
        "scientific_execution": False,
        "installer_path": uv,
        "installer_sha256": _hash(Path(uv)),
        "environment_sha256": digest_bytes(canonical_bytes(environment)),
        "commands": [{"name": name, "sha256": _hash(output / f"{name}.json")} for name in COMMAND_NAMES],
        "archive_inspection_sha256": _hash(output / "archive-inspection.json"),
    }
    publish_json(output, "receipt.json", receipt)
    return receipt


def audit_package_receipt(output: Path, source_root: Path) -> JSONObject:
    """Check compact command/source links; parent must prove actual sandboxed execution."""
    directory, source = _plain_directory(output), _plain_directory(source_root)
    receipt = _small_object(directory / "receipt.json")
    contract = _contract(source)
    expected = checked_json({name: digest_bytes(payload) for name, payload in contract.files.items()})
    if (
        receipt.get("schema") != "s10-package-inspection-v1"
        or receipt.get("status") != "PASS"
        or receipt.get("frozen_build_files") != expected
        or receipt.get("clean_install") is not True
        or receipt.get("scientific_execution") is not False
    ):
        raise IntegrityError("packaging receipt does not bind this candidate source")
    if (
        receipt.get("network") != "PARENT_VERIFIER_SANDBOX_PROOF_REQUIRED"
        or receipt.get("network_isolation_attested") is not False
    ):
        raise IntegrityError("packaging cannot self-attest parent network boundaries")
    _audit_package_commands(directory, receipt)
    _audit_archive_bytes(directory, receipt, contract.root_name)
    inspection = _small_object(directory / "archive-inspection.json")
    if _hash(directory / "archive-inspection.json") != receipt.get("archive_inspection_sha256") or any(
        receipt.get(key) != value for key, value in inspection.items()
    ):
        raise IntegrityError("packaging inspection link differs from completed receipt")
    return receipt


def _audit_archive_bytes(directory: Path, receipt: JSONObject, root_name: str) -> None:
    archives = _plain_directory(directory / "dist")
    names = {f"{root_name}-py3-none-any.whl": "wheel_sha256", f"{root_name}.tar.gz": "sdist_sha256"}
    if {path.name for path in archives.iterdir()} != set(names):
        raise IntegrityError("packaging output inventory changed after inspection")
    for name, key in names.items():
        path = archives / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 33_554_432:
            raise IntegrityError("bounded regular distribution archives required")
        if _hash(path) != receipt.get(key):
            raise IntegrityError("distribution bytes changed after inspected build")


def _small_object(path: Path) -> JSONObject:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_048_576:
        raise IntegrityError("bounded regular packaging metadata required")
    return object_json(parse_json(path.read_bytes()))


def _audit_package_commands(directory: Path, receipt: JSONObject) -> None:
    rows = {name: _small_object(directory / f"{name}.json") for name in COMMAND_NAMES}
    expected = [{"name": name, "sha256": _hash(directory / f"{name}.json")} for name in COMMAND_NAMES]
    if receipt.get("commands") != expected:
        raise IntegrityError("all exact packaging command receipts required")
    for name in COMMAND_NAMES:
        row = rows[name]
        log = directory / f"{name}.log"
        if log.is_symlink() or not log.is_file() or log.stat().st_size > 4_194_304:
            raise IntegrityError("bounded regular packaging command log required")
        if (
            row.get("name") != name
            or row.get("exit_code") != 0
            or row.get("failure") is not None
            or row.get("log_sha256") != _hash(directory / f"{name}.log")
        ):
            raise IntegrityError("packaging command failed or retained logs changed")


def main(argv: list[str] | None = None) -> int:
    """Execute only the fixed offline engineering packaging operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_package(args.source_root, args.output)
    except (S10Error, OSError, subprocess.SubprocessError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 6
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
