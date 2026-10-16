"""Execute and audit exact local engineering checks with retained command evidence."""

import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

from frem_s10.errors import ContractError, IntegrityError
from frem_s10.identities import JSONObject, JSONValue, checked_json, object_json
from frem_s10.runtime.artifacts import _plain_directory
from frem_s10.runtime.declarations import BOUND_ROLES, DESIGN, SCOPES, _reference, _schema
from frem_s10.runtime.provenance import (
    PACKAGE,
    THREAD_ENV,
    audit_code,
    audit_environment,
    read_object,
    save_object,
)

CHECK_NAMES = ("collection", "tests", "lint", "format", "types")


def verification_commands(source_root: Path, work: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the fixed supported command family; never execute manifest-supplied argv."""
    package = str(PACKAGE)
    config = f"{package}/pyproject.toml"
    python = str(Path(sys.executable).absolute())
    binary = Path(sys.executable).absolute().parent
    pytest = (python, "-B", "-m", "pytest", "-c", config, f"{package}/tests", "-q", "-p", "no:cacheprovider")
    sources = (f"{package}/src/frem_s10", f"{package}/tests")
    return (
        ("collection", (*pytest, "--collect-only")),
        ("tests", (*pytest, "-W", "error", "--basetemp", str(work / "fixtures"))),
        ("lint", (str(binary / "ruff"), "check", "--config", config, *sources)),
        ("format", (str(binary / "ruff"), "format", "--check", "--config", config, *sources)),
        (
            "types",
            (str(binary / "mypy"), "--config-file", config, "--cache-dir", str(work / "mypy"), *sources),
        ),
    )


def _environment() -> dict[str, str]:
    env = dict(os.environ, **THREAD_ENV, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return env


def _work_path(value: JSONValue, root: Path) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ContractError("relative verification output directory required")
    return _plain_directory(root / value)


def _capture(name: str, command: tuple[str, ...], source: Path, root: Path) -> JSONObject:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, cwd=source, env=_environment(), capture_output=True, text=True, timeout=300
        )
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        stderr += "\nVERIFICATION_COMMAND_TIMEOUT\n"
    except OSError as exc:
        code, stdout, stderr = 127, "", f"VERIFICATION_LAUNCH_FAILED: {type(exc).__name__}: {exc}"
    output = save_object(root, {"schema": "s10-command-output-v1", "stdout": stdout, "stderr": stderr})
    return {
        "name": name,
        "argv": list(command),
        "cwd": str(source),
        "exit_code": code,
        "wall_seconds": time.monotonic() - started,
        "cumulative_child_peak_rss_bytes_macos": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        "output": output,
    }


def _observed_counts(commands: list[JSONObject], root: Path) -> tuple[int, int]:
    collection = read_object(commands[0]["output"], root)
    test = read_object(commands[1]["output"], root)
    collected = re.search(r"(\d+) tests? collected", str(collection["stdout"]))
    passed = re.search(r"(\d+) passed(?: in |,)", str(test["stdout"]))
    if (
        collected is None
        or passed is None
        or re.search(r"\d+ (?:skipped|xfailed|xpassed|failed|errors?)", str(test["stdout"]))
    ):
        raise IntegrityError("complete collected and passed test counts without omissions required")
    first, second = int(collected[1]), int(passed[1])
    if first != second or first < 274:
        raise IntegrityError("test inventory is incomplete or regressed below retained274 baseline")
    return first, second


def _prerequisites(
    *,
    artifact_root: Path,
    source_root: Path,
    code_ref: JSONValue,
    environment_ref: JSONValue,
    bindings: JSONObject,
    evidence_scope: str,
) -> None:
    if evidence_scope not in SCOPES or set(bindings) != set(BOUND_ROLES):
        raise ContractError("fixed scope and complete five-role verification bindings required")
    if (
        _reference(code_ref, artifact_root)[1] != bindings["code"]
        or _reference(environment_ref, artifact_root)[1] != bindings["environment"]
    ):
        raise IntegrityError("verification code/environment references differ from bound identities")
    audit_code(code_ref, artifact_root=artifact_root, source_root=source_root)
    audit_environment(
        environment_ref, artifact_root=artifact_root, source_root=source_root, evidence_scope=evidence_scope
    )


def run_verification(
    *,
    artifact_root: Path,
    source_root: Path,
    code_ref: JSONValue,
    environment_ref: JSONValue,
    bindings: JSONObject,
    operation: str,
    evidence_scope: str,
    output_directory: str,
    qualification: JSONValue = None,
) -> JSONObject:
    """Run fixed bounded checks and publish actual results; no market sources are opened.

    Failed command outputs are committed as objects immediately and retained.
    A source change during verification invalidates the result. Campaign-scale
    qualification is a separate generated resource-gated prerequisite.
    """
    root, source = _plain_directory(artifact_root), _plain_directory(source_root)
    if operation not in ("prepare-precision", "development-campaign"):
        raise ContractError("fixed verification operation required")
    _prerequisites(
        artifact_root=root,
        source_root=source,
        code_ref=code_ref,
        environment_ref=environment_ref,
        bindings=bindings,
        evidence_scope=evidence_scope,
    )
    if not output_directory or Path(output_directory).is_absolute() or ".." in Path(output_directory).parts:
        raise ContractError("new relative verification directory required")
    work = root / output_directory
    work.mkdir(parents=True, exist_ok=False)
    _plain_directory(work)
    commands = []
    for name, command in verification_commands(source, work):
        row = _capture(name, command, source, root)
        save_object(root, row)
        commands.append(row)
        if row["exit_code"] != 0:
            break
    status = "FAIL"
    reason = "a required command did not complete successfully"
    counts: tuple[int, int] | None = None
    if len(commands) == len(CHECK_NAMES) and all(row["exit_code"] == 0 for row in commands):
        try:
            _prerequisites(
                artifact_root=root,
                source_root=source,
                code_ref=code_ref,
                environment_ref=environment_ref,
                bindings=bindings,
                evidence_scope=evidence_scope,
            )
            counts = _observed_counts(commands, root)
            status, reason = "PASS", "all fixed commands and current identity checks passed"
        except (IntegrityError, ContractError) as exc:
            reason = str(exc)
    evidence = save_object(
        root,
        {
            "schema": "s10-verification-evidence-v2",
            "status": status,
            "reason": reason,
            "operation": operation,
            "evidence_scope": evidence_scope,
            "bindings": bindings,
            "code": code_ref,
            "environment": environment_ref,
            "work_directory": output_directory,
            "commands": checked_json(commands),
            "counts": None if counts is None else list(counts),
            "qualification": qualification,
        },
    )
    result = save_object(
        root,
        {
            "schema": "s10-test-result-v2",
            "status": status,
            "code_sha256": bindings["code"],
            "environment_sha256": bindings["environment"],
            "evidence": evidence,
        },
    )
    return save_object(
        root,
        {
            "schema": "s10-verification-receipt-v1",
            "design_id": DESIGN,
            "operation": operation,
            "evidence_scope": evidence_scope,
            "bindings": bindings,
            "status": status,
            "logs": [result],
        },
    )


def _audit_commands(evidence: JSONObject, root: Path, source: Path) -> tuple[int, int]:
    work = _work_path(evidence["work_directory"], root)
    records = evidence["commands"]
    if not isinstance(records, list) or len(records) != len(CHECK_NAMES):
        raise IntegrityError("all fixed verification commands required")
    rows = [object_json(row) for row in records]
    for row, (name, command) in zip(rows, verification_commands(source, work), strict=True):
        _schema(
            row,
            {
                "name",
                "argv",
                "cwd",
                "exit_code",
                "wall_seconds",
                "cumulative_child_peak_rss_bytes_macos",
                "output",
            },
            "command observation",
        )
        if (
            row["name"] != name
            or row["argv"] != list(command)
            or row["cwd"] != str(source)
            or row["exit_code"] != 0
        ):
            raise IntegrityError("verification command or observed success mismatch")
        output = read_object(row["output"], root)
        _schema(output, {"schema", "stdout", "stderr"}, "command output")
        if (
            output["schema"] != "s10-command-output-v1"
            or not isinstance(output["stdout"], str)
            or not isinstance(output["stderr"], str)
        ):
            raise IntegrityError("retained stdout/stderr required")
    return _observed_counts(rows, root)


def audit_verification(reference: JSONValue, *, artifact_root: Path, source_root: Path) -> JSONObject:
    """Reconcile exact current identities and observed commands before production use."""
    root, source = _plain_directory(artifact_root), _plain_directory(source_root)
    receipt = read_object(reference, root)
    _schema(
        receipt,
        {"schema", "design_id", "operation", "evidence_scope", "bindings", "status", "logs"},
        "verification receipt",
    )
    if (
        receipt["schema"] != "s10-verification-receipt-v1"
        or receipt["design_id"] != DESIGN
        or receipt["status"] != "PASS"
    ):
        raise IntegrityError("current passing design4 verification required")
    refs = receipt["logs"]
    if not isinstance(refs, list) or len(refs) != 1:
        raise IntegrityError("exact complete verification result required")
    result = read_object(refs[0], root)
    _schema(
        result, {"schema", "status", "code_sha256", "environment_sha256", "evidence"}, "observed test result"
    )
    if result["schema"] != "s10-test-result-v2" or result["status"] != "PASS":
        raise IntegrityError("a structural v1 PASS cannot certify production verification")
    evidence = read_object(result["evidence"], root)
    _schema(
        evidence,
        {
            "schema",
            "status",
            "reason",
            "operation",
            "evidence_scope",
            "bindings",
            "code",
            "environment",
            "work_directory",
            "commands",
            "counts",
            "qualification",
        },
        "verification evidence",
    )
    bindings = object_json(receipt["bindings"])
    if evidence["schema"] != "s10-verification-evidence-v2" or any(
        evidence[key] != receipt[key] for key in ("operation", "evidence_scope", "bindings", "status")
    ):
        raise IntegrityError("command evidence differs from bound verification receipt")
    if result["code_sha256"] != bindings["code"] or result["environment_sha256"] != bindings["environment"]:
        raise IntegrityError("test result identity mismatch")
    scope = receipt["evidence_scope"]
    if not isinstance(scope, str):
        raise IntegrityError("verification evidence scope required")
    _prerequisites(
        artifact_root=root,
        source_root=source,
        code_ref=evidence["code"],
        environment_ref=evidence["environment"],
        bindings=bindings,
        evidence_scope=scope,
    )
    counts = _audit_commands(evidence, root, source)
    if evidence["counts"] != list(counts):
        raise IntegrityError("retained collection/pass counts disagree")
    if receipt["operation"] == "development-campaign":
        from frem_s10.runtime.qualification import audit_qualification

        audit_qualification(
            evidence["qualification"],
            artifact_root=root,
            source_root=source,
            code_sha256=str(bindings["code"]),
            environment_sha256=str(bindings["environment"]),
            calendar_sha256=str(bindings["calendar"]),
        )
    elif receipt["operation"] != "prepare-precision":
        raise IntegrityError("unsupported verification operation")
    return {
        "status": "PASS",
        "tests_collected": counts[0],
        "tests_passed": counts[1],
        "bindings": bindings,
        "evidence_scope": scope,
        "operation": receipt["operation"],
    }
