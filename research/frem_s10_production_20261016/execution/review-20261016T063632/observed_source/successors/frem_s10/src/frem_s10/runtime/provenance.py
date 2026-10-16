"""Content-addressed local code/environment observations without numerical imports."""

import importlib.metadata
import importlib.util
import platform
import re
import shutil
import sys
import sysconfig
import tomllib
from pathlib import Path
from zoneinfo import TZPATH

from frem_s10.errors import ContractError, IntegrityError
from frem_s10.identities import JSONObject, JSONValue, canonical_bytes, digest_bytes, object_json
from frem_s10.runtime.artifacts import _plain_directory, publish_json, read_verified_json
from frem_s10.runtime.declarations import DESIGN, SCOPES, _reference

THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
PACKAGE = Path("successors/frem_s10")
OBJECTS = Path("research/frem_s10_objects")


def save_object(artifact_root: Path, value: JSONValue) -> JSONObject:
    """Publish canonical bytes once under their exact SHA256-derived basename."""
    root = _plain_directory(artifact_root)
    objects = root / OBJECTS
    objects.mkdir(parents=True, exist_ok=True)
    _plain_directory(objects)
    payload = canonical_bytes(value) + b"\n"
    identity = digest_bytes(payload)
    name = identity + ".json"
    path = objects / name
    if path.exists():
        read_verified_json(path, identity, maximum_bytes=16_777_216)
    else:
        actual = publish_json(objects, name, value)
        if actual != identity:
            raise IntegrityError("content-addressed publication identity mismatch")
    return {"path": str(OBJECTS / name), "sha256": identity}


def read_object(reference: JSONValue, artifact_root: Path) -> JSONObject:
    """Read a bounded exact reference without interpreting supplied module paths."""
    path, identity = _reference(reference, _plain_directory(artifact_root))
    return object_json(read_verified_json(path, identity, maximum_bytes=16_777_216))


def expected_code_files(source_root: Path) -> tuple[Path, ...]:
    """Enumerate the complete supported package, test and build-contract inventory."""
    root = _plain_directory(source_root)
    package = root / PACKAGE
    sources = tuple(sorted((package / "src/frem_s10").rglob("*.py")))
    tests = tuple(sorted((package / "tests").rglob("*.py")))
    if not sources or not tests:
        raise IntegrityError("complete source and test inventories required")
    _validate_inventory(package)
    paths = (
        sources
        + tests
        + tuple(
            package / name
            for name in (
                "pyproject.toml",
                "uv.lock",
                "README.md",
                "MANIFEST.in",
                "src/frem_s10/science.json",
                "src/frem_s10/py.typed",
            )
        )
    )
    for path in paths:
        if path.is_symlink() or any(part.is_symlink() for part in path.parents) or not path.is_file():
            raise IntegrityError("source inventory must consist of regular non-symlink files")
        if path.stat().st_size > 1_048_576:
            raise IntegrityError("source inventory member exceeds declared bound")
    return tuple(sorted(paths))


def _validate_inventory(package: Path) -> None:
    """Reject source/test payloads which are absent from the explicit freeze."""
    allowed_data = {package / "src/frem_s10/science.json", package / "src/frem_s10/py.typed"}
    for base in (package / "src/frem_s10", package / "tests"):
        if base.is_symlink():
            raise IntegrityError("source/test directory symlinks are unsupported")
        for member in base.rglob("*"):
            if member.is_symlink():
                raise IntegrityError("source/test directory symlinks are unsupported")
            if (
                member.is_file()
                and "__pycache__" not in member.parts
                and member.suffix != ".py"
                and member not in allowed_data
            ):
                raise IntegrityError("unregistered source/test payload is outside the exact frozen inventory")


def code_document(source_root: Path) -> JSONObject:
    """Hash all current supported code; omitting a member cannot establish a freeze."""
    root = _plain_directory(source_root)
    return {
        "schema": "s10-code-manifest-v1",
        "files": [
            {"path": str(path.relative_to(root)), "sha256": digest_bytes(path.read_bytes())}
            for path in expected_code_files(root)
        ],
    }


def create_code_manifest(artifact_root: Path, source_root: Path) -> JSONObject:
    """Record exact source identities before verification or scientific declaration."""
    return save_object(artifact_root, code_document(source_root))


def audit_code(reference: JSONValue, *, artifact_root: Path, source_root: Path) -> None:
    """Reject modified, extra or omitted package/test/build members."""
    if read_object(reference, artifact_root) != code_document(source_root):
        raise IntegrityError("complete current code inventory differs from verified snapshot")


def _timezone_identity() -> JSONObject:
    for base in TZPATH:
        path = Path(base) / "America/New_York"
        if path.is_file():
            return {"zone": "America/New_York", "path": str(path), "sha256": digest_bytes(path.read_bytes())}
    raise IntegrityError("observed America/New_York timezone data unavailable")


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.lower())


def _installed_versions(source_root: Path) -> list[tuple[str, str]]:
    """Require the entire resolved supported lock, with no silently upgraded tools."""
    distributions = list(importlib.metadata.distributions())
    versions = sorted((_normalized_name(dist.metadata["Name"]), dist.version) for dist in distributions)
    locked = tomllib.loads((source_root / PACKAGE / "uv.lock").read_text())
    expected = {item["name"]: item["version"] for item in locked["package"] if item["name"] != "colorama"}
    if dict(versions) != expected:
        raise IntegrityError("installed package versions differ from the complete resolved macOS lock")
    for name in {name for name, _ in versions}:
        rows = [dist for dist in distributions if _normalized_name(dist.metadata["Name"]) == name]
        if len(rows) == 1:
            continue
        # The editable project's build leaves same-version egg-info in src.
        # Only this exact source/venv pair is permitted; other duplicates fail.
        roots = {Path(str(dist.locate_file(""))).resolve() for dist in rows}
        expected_roots = {
            (source_root / PACKAGE / "src").resolve(),
            (Path(sys.executable).absolute().parent.parent / "lib/python3.12/site-packages").resolve(),
        }
        if (
            name != "frem-s10"
            or len(rows) != 2
            or roots != expected_roots
            or len({dist.version for dist in rows}) != 1
        ):
            raise IntegrityError("ambiguous duplicate installed package metadata")
    return sorted(set(versions))


def observe_environment(source_root: Path) -> JSONObject:
    """Observe the actual supported interpreter, installed versions and timezone bytes."""
    spec = importlib.util.find_spec("frem_s10")
    expected = source_root.absolute() / PACKAGE / "src/frem_s10/__init__.py"
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected.resolve():
        raise IntegrityError("verification interpreter does not import the declared local source package")
    if sys.version_info[:2] != (3, 12) or platform.system() != "Darwin" or platform.machine() != "arm64":
        raise IntegrityError("this production environment requires actually verified Python3.12 macOSarm64")
    versions = _installed_versions(source_root)
    return {
        "schema": "s10-environment-observation-v1",
        "python": sys.version,
        "executable": str(Path(sys.executable).absolute()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "implementation": platform.python_implementation(),
        "soabi": str(sysconfig.get_config_var("SOABI")),
        "packages": [[name, version] for name, version in versions],
        "source_origin": str(expected),
        "lock_sha256": digest_bytes((source_root / PACKAGE / "uv.lock").read_bytes()),
        "timezone": _timezone_identity(),
        "offline_installer": _installer_identity(),
        "required_worker_thread_environment": dict(THREAD_ENV),
    }


def _installer_identity() -> JSONObject:
    path = shutil.which("uv")
    if path is None:
        raise IntegrityError("fixed offline installer is unavailable")
    resolved = Path(path).resolve(strict=True)
    return {"path": path, "resolved_path": str(resolved), "sha256": digest_bytes(resolved.read_bytes())}


def create_environment_receipt(artifact_root: Path, source_root: Path, *, evidence_scope: str) -> JSONObject:
    """Persist actual observations; a matching receipt alone never grants execution."""
    if evidence_scope not in SCOPES:
        raise ContractError("explicit declared evidence scope required")
    observation = save_object(artifact_root, observe_environment(source_root))
    return save_object(
        artifact_root,
        {
            "schema": "s10-environment-receipt-v1",
            "design_id": DESIGN,
            "evidence_scope": evidence_scope,
            "status": "PASS",
            "identity": observation["sha256"],
        },
    )


def audit_environment(
    reference: JSONValue, *, artifact_root: Path, source_root: Path, evidence_scope: str
) -> None:
    """Re-observe installed state rather than trusting a caller's PASS string."""
    receipt = read_object(reference, artifact_root)
    if set(receipt) != {"schema", "design_id", "evidence_scope", "status", "identity"}:
        raise IntegrityError("exact environment receipt schema required")
    identity = receipt["identity"]
    if not isinstance(identity, str) or len(identity) != 64:
        raise IntegrityError("environment observation identity required")
    expected: JSONObject = {
        "schema": "s10-environment-receipt-v1",
        "design_id": DESIGN,
        "evidence_scope": evidence_scope,
        "status": "PASS",
        "identity": identity,
    }
    if receipt != expected:
        raise IntegrityError("environment receipt declaration mismatch")
    observed = read_object({"path": str(OBJECTS / f"{identity}.json"), "sha256": identity}, artifact_root)
    if observed != observe_environment(source_root):
        raise IntegrityError("current interpreter, dependencies or timezone differ from verified observation")
