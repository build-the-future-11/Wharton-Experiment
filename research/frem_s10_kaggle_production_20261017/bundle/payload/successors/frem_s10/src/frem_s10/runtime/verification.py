"""Execute and audit exact local engineering checks with retained command evidence."""

import json
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from frem_s10.errors import ContractError, IntegrityError
from frem_s10.identities import JSONObject, JSONValue, checked_json, object_json, parse_json
from frem_s10.metadata import finite_float, strict_int
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
from frem_s10.runtime.resources import audit_memory_rows, check_host

CHECK_NAMES = ("collection", "tests", "lint", "format", "types", "packaging", "mutations", "boundary_probe")
BOUNDARY_PROBE = """import errno, os, pathlib, socket, sys
try:
    with socket.socket() as connection:
        connection.connect(("127.0.0.1", 1))
except OSError as error:
    assert error.errno in (errno.EPERM, errno.EACCES), repr(error)
else:
    raise AssertionError("network boundary absent")
for denied in sys.argv[1:4]:
    try:
        list(os.scandir(denied))
    except PermissionError:
        pass
    else:
        raise AssertionError("source evidence boundary absent: " + denied)
assert list(pathlib.Path(sys.argv[4]).iterdir()) == []
print("NETWORK_DENIED DATA_OUTPUTS_RESEARCH_DENIED PROTECTED_FIXTURE_EMPTY")
"""


def verification_commands(source_root: Path, work: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the fixed supported command family; never execute manifest-supplied argv."""
    package = str(PACKAGE)
    config = f"{package}/pyproject.toml"
    python = str(Path(sys.executable).absolute())
    binary = Path(sys.executable).absolute().parent
    interpreter = (python, "-B", "-X", f"pycache_prefix={work / 'empty_bytecode'}")
    pytest = (*interpreter, "-m", "pytest", "-c", config, f"{package}/tests", "-q", "-p", "no:cacheprovider")
    sources = (f"{package}/src/frem_s10", f"{package}/tests")
    commands = (
        ("collection", (*pytest, "--collect-only")),
        (
            "tests",
            (
                *pytest,
                "-W",
                "error",
                "--basetemp",
                str(work / "fixtures"),
                "--s10-receipt",
                str(work / "pytest-observations.json"),
            ),
        ),
        ("lint", (str(binary / "ruff"), "check", "--config", config, *sources)),
        ("format", (str(binary / "ruff"), "format", "--check", "--config", config, *sources)),
        (
            "types",
            (str(binary / "mypy"), "--config-file", config, "--cache-dir", str(work / "mypy"), *sources),
        ),
        (
            "packaging",
            (
                *interpreter,
                "-m",
                "frem_s10.runtime.packaging",
                "--source-root",
                str(source_root),
                "--output",
                str(work / "package"),
            ),
        ),
        (
            "mutations",
            (
                *interpreter,
                "-m",
                "frem_s10.runtime.mutation_check",
                "--source-root",
                str(source_root),
                "--output",
                str(work / "mutations"),
            ),
        ),
        (
            "boundary_probe",
            (
                *interpreter,
                "-c",
                BOUNDARY_PROBE,
                *(str(source_root / item) for item in ("data", "outputs", "research")),
                str(work / "protected_empty"),
            ),
        ),
    )
    from frem_s10.runtime.packaging import NUMPY_WHEEL

    if platform.system() == "Linux":
        from frem_s10.runtime import kaggle_contract

        if kaggle_contract.active():
            sealed = os.environ.get("FREM_S10_SEALED_INPUT")
            if not sealed:
                raise IntegrityError("exact sealed Kaggle input path required")
            kaggle_contract.require_sealed_input(Path(sealed))
            from frem_s10.runtime.packaging import dependency_wheel

            wheel, _ = dependency_wheel(source_root)
            probe = (
                *interpreter,
                "-c",
                kaggle_contract.BOUNDARY_PROBE,
                sealed,
                *(str(source_root / item) for item in ("data", "outputs", "research")),
                str(wheel),
                str(work / "protected_empty"),
            )
            return tuple((name, probe if name == "boundary_probe" else command) for name, command in commands)
        from frem_s10.runtime.isolation import linux_prefix, linux_probe_arguments
        from frem_s10.runtime.packaging import dependency_wheel

        wheel, _ = dependency_wheel(source_root)
        prefix = linux_prefix(source_root, work, wheel)
        probe = (*interpreter, *linux_probe_arguments(source_root, work, wheel))
        return tuple(
            (name, (*prefix, *(probe if name == "boundary_probe" else command))) for name, command in commands
        )
    if platform.system() != "Darwin":
        raise IntegrityError("verification isolation is unsupported on this platform")

    denied = "".join(
        f"(subpath {json.dumps(str(source_root / item))})" for item in ("data", "outputs", "research")
    )
    policy = (
        f"(version 1)(allow default)(deny network*)(deny file-read* {denied})"
        f"(allow file-read* (subpath {json.dumps(str(work))})"
        f"(literal {json.dumps(str(source_root / NUMPY_WHEEL))}))"
    )
    return tuple((name, ("/usr/bin/sandbox-exec", "-p", policy, *command)) for name, command in commands)


def _environment() -> dict[str, str]:
    env = dict(os.environ, **THREAD_ENV, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return env


def _probe_message() -> str:
    if platform.system() == "Linux":
        from frem_s10.runtime import kaggle_contract

        if kaggle_contract.active():
            return kaggle_contract.BOUNDARY_MESSAGE
        from frem_s10.runtime.isolation import LINUX_PROBE_MESSAGE

        return LINUX_PROBE_MESSAGE
    return "NETWORK_DENIED DATA_OUTPUTS_RESEARCH_DENIED PROTECTED_FIXTURE_EMPTY"


def _rss_observation() -> tuple[str, int]:
    value = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    if platform.system() == "Linux":
        return "cumulative_child_peak_rss_bytes_linux", value * 1024
    return "cumulative_child_peak_rss_bytes_macos", value


def _work_path(value: JSONValue, root: Path) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ContractError("relative verification output directory required")
    return _plain_directory(root / value)


def _capture(name: str, command: tuple[str, ...], source: Path, root: Path) -> JSONObject:
    started = time.monotonic()
    try:
        with subprocess.Popen(
            command,
            cwd=source,
            env=_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(
                    timeout=600 if name in ("tests", "packaging", "mutations") else 120
                )
                code = process.returncode
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=5)
                code, stderr = 124, stderr + "\nVERIFICATION_COMMAND_TIMEOUT_OWNED_GROUP_STOPPED\n"
            if _stop_remaining_group(process.pid):
                code, stderr = 125, stderr + "\nVERIFICATION_COMMAND_LEFT_DESCENDANTS_OWNED_GROUP_STOPPED\n"
    except OSError as exc:
        code, stdout, stderr = 127, "", f"VERIFICATION_LAUNCH_FAILED: {type(exc).__name__}: {exc}"
    output = save_object(root, {"schema": "s10-command-output-v1", "stdout": stdout, "stderr": stderr})
    rss_field, rss_bytes = _rss_observation()
    return {
        "name": name,
        "argv": list(command),
        "cwd": str(source),
        "exit_code": code,
        "wall_seconds": time.monotonic() - started,
        rss_field: rss_bytes,
        "output": output,
    }


def _stop_remaining_group(group: int) -> bool:
    """Stop descendants in this command's new session, never another workload."""
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    with suppress(ProcessLookupError):
        os.killpg(group, signal.SIGKILL)
    return True


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
    catalog_ref: JSONValue,
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
    (work / "protected_empty").mkdir()
    (work / "empty_bytecode").mkdir()
    from frem_s10.runtime.isolation import prepare_linux_fixture

    prepare_linux_fixture(work)
    host = checked_json(check_host())
    host_ref = save_object(root, host)
    commands = []
    if object_json(host)["status"] == "PASS":
        for name, command in verification_commands(source, work):
            row = _capture(name, command, source, root)
            save_object(root, row)
            commands.append(row)
            if row["exit_code"] != 0:
                break
    status = "FAIL"
    reason = (
        "host resource gate blocked before engineering execution"
        if not commands
        else "a required command did not complete successfully"
    )
    observations = _save_local_metadata(work / "pytest-observations.json", root)
    package_receipt = _save_local_metadata(work / "package/receipt.json", root)
    mutation_receipt = _save_local_metadata(work / "mutations/receipt.json", root)
    evidence: JSONObject = {
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
        "counts": None,
        "qualification": qualification,
        "host": host_ref,
        "source_verification": None,
        "catalog": catalog_ref,
        "pytest_observations": observations,
        "packaging": package_receipt,
        "mutations": mutation_receipt,
    }
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
            counts = _audit_commands(evidence, root, source)
            _audit_host(host_ref, root)
            _audit_scoped_checks(evidence, root, source, counts)
            evidence["counts"] = list(counts)
            status, reason = "PASS", "all fixed commands and current identity checks passed"
        except (IntegrityError, ContractError) as exc:
            reason = str(exc)
    evidence.update({"status": status, "reason": reason})
    result = save_object(
        root,
        {
            "schema": "s10-test-result-v2",
            "status": status,
            "code_sha256": bindings["code"],
            "environment_sha256": bindings["environment"],
            "evidence": save_object(root, evidence),
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
    rss_field = _rss_observation()[0]
    for row, (name, command) in zip(rows, verification_commands(source, work), strict=True):
        _schema(
            row,
            {
                "name",
                "argv",
                "cwd",
                "exit_code",
                "wall_seconds",
                rss_field,
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
        if name == "boundary_probe" and output["stdout"].strip() != _probe_message():
            raise IntegrityError("network/data isolation was not actually observed")
        wall = finite_float(row["wall_seconds"], name="observed command wall", strictly_positive=True)
        if wall > (605 if name in ("tests", "packaging", "mutations") else 125):
            raise IntegrityError("engineering command exceeded its declared timeout plus cleanup")
        rss = strict_int(row[rss_field], name="observed command cumulative RSS", minimum=1)
        if rss > (512 if name in ("types", "packaging", "mutations", "boundary_probe") else 256) * 2**20:
            raise IntegrityError("engineering cumulative child RSS exceeded its conservative ceiling")
    return _observed_counts(rows, root)


def _audit_host(reference: JSONValue, root: Path) -> None:
    host = read_object(reference, root)
    samples = host.get("samples")
    if host.get("status") != "PASS" or not isinstance(samples, list) or len(samples) != 3:
        raise IntegrityError("three admitted host samples before complete engineering execution required")
    try:
        audit_memory_rows(samples)
    except Exception as exc:
        raise IntegrityError("engineering preflight resource admission failed") from exc


def _save_local_metadata(path: Path, root: Path) -> JSONValue:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_777_216:
        raise IntegrityError("bounded regular verification metadata required")
    return save_object(root, parse_json(path.read_bytes()))


def _pytest_observations(
    evidence: JSONObject, root: Path, counts: tuple[int, int]
) -> tuple[tuple[str, ...], dict[str, str]]:
    observations = read_object(evidence["pytest_observations"], root)
    _schema(observations, {"schema", "collected", "outcomes", "exit_code"}, "pytest observations")
    if observations["schema"] != "s10-pytest-observations-v1" or observations["exit_code"] != 0:
        raise IntegrityError("actual successful pytest observations required")
    collected, raw = observations["collected"], object_json(observations["outcomes"])
    if not isinstance(collected, list) or any(not isinstance(node, str) for node in collected):
        raise IntegrityError("exact collected pytest node IDs required")
    nodes = tuple(str(node) for node in collected)
    if len(nodes) != counts[0] or len(set(nodes)) != len(nodes) or set(raw) != set(nodes):
        raise IntegrityError("pytest counts, collected identities and observed outcomes disagree")
    if any(value != "PASSED" for value in raw.values()):
        raise IntegrityError("every collected test must pass without skip or xfail")
    rows = evidence["commands"]
    if not isinstance(rows, list):
        raise IntegrityError("actual collection command required")
    collection = read_object(object_json(rows[0])["output"], root)
    listed = tuple(
        line for line in str(collection["stdout"]).splitlines() if line.startswith("tests/") and "::" in line
    )
    if listed != nodes:
        raise IntegrityError("separate collection and executed pytest node identities differ")
    return nodes, dict.fromkeys(nodes, "PASSED")


def _audit_packaging(evidence: JSONObject, root: Path, source: Path) -> None:
    from frem_s10.runtime.packaging import audit_package_receipt

    receipt = read_object(evidence["packaging"], root)
    rows = evidence["commands"]
    if not isinstance(rows, list):
        raise IntegrityError("actual packaging execution required")
    row = next((object_json(raw) for raw in rows if object_json(raw)["name"] == "packaging"), None)
    if row is None:
        raise IntegrityError("fixed packaging command missing")
    output = read_object(row["output"], root)
    if parse_json(str(output["stdout"])) != receipt:
        raise IntegrityError("saved package receipt differs from actual packaging command output")
    if (
        receipt.get("status") != "PASS"
        or receipt.get("clean_install") is not True
        or receipt.get("scientific_execution") is not False
    ):
        raise IntegrityError("exact-source build and clean offline installation must pass")
    if audit_package_receipt(_work_path(evidence["work_directory"], root) / "package", source) != receipt:
        raise IntegrityError("package archive/command/source evidence differs from retained observation")


def _qualification_obligations(evidence: JSONObject, root: Path, source: Path) -> tuple[str, ...]:
    if evidence["operation"] == "prepare-precision":
        if evidence["qualification"] is not None:
            raise IntegrityError("pilot verification cannot borrow a campaign qualification")
        return ("TEST-086", "TEST-093", "TEST-096")
    if evidence["operation"] != "development-campaign":
        raise IntegrityError("unsupported verification operation")
    from frem_s10.runtime.qualification import audit_qualification

    bindings = object_json(evidence["bindings"])
    audit_qualification(
        evidence["qualification"],
        artifact_root=root,
        source_root=source,
        code_sha256=str(bindings["code"]),
        environment_sha256=str(bindings["environment"]),
        calendar_sha256=str(bindings["calendar"]),
    )
    return ("TEST-081", "TEST-086", "TEST-090", "TEST-093", "TEST-096")


def _audit_mutations(evidence: JSONObject, root: Path, source: Path) -> None:
    from frem_s10.runtime.mutation_check import audit_mutation_receipt

    receipt = read_object(evidence["mutations"], root)
    rows = evidence["commands"]
    if not isinstance(rows, list):
        raise IntegrityError("actual fixed mutation command required")
    row = next((object_json(raw) for raw in rows if object_json(raw)["name"] == "mutations"), None)
    if row is None:
        raise IntegrityError("fixed mutation command missing")
    output = read_object(row["output"], root)
    if parse_json(str(output["stdout"])) != receipt:
        raise IntegrityError("saved mutation receipt differs from actual command output")
    if audit_mutation_receipt(_work_path(evidence["work_directory"], root) / "mutations", source) != receipt:
        raise IntegrityError("mutation proof links differ from the observed final-candidate check")


def _audit_scoped_checks(evidence: JSONObject, root: Path, source: Path, counts: tuple[int, int]) -> None:
    from frem_s10.runtime.catalog import REQUIREMENT_IDS, validate_operation_catalog
    from frem_s10.runtime.preservation import audit_preservation

    nodes, outcomes = _pytest_observations(evidence, root, counts)
    _audit_packaging(evidence, root, source)
    _audit_mutations(evidence, root, source)
    external = _qualification_obligations(evidence, root, source)
    catalog = read_object(evidence["catalog"], root)
    audit_preservation(catalog.get("preservation_audits"), artifact_root=root)
    readiness = validate_operation_catalog(
        catalog,
        operation="prepare-precision"
        if evidence["operation"] == "prepare-precision"
        else "run-development-campaign",
        target_platform="Linux"
        if platform.system() == "Linux"
        and __import__("frem_s10.runtime.kaggle_contract", fromlist=["active"]).active()
        else "macOS",
        requirements=REQUIREMENT_IDS,
        collected_nodeids=nodes,
        test_outcomes=outcomes,
        externally_verified_obligations=external,
    )
    if not readiness.complete:
        raise IntegrityError(
            "operation-specific verification obligations incomplete: " + "; ".join(readiness.gaps)
        )


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
            "host",
            "source_verification",
            "catalog",
            "pytest_observations",
            "packaging",
            "mutations",
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
    _audit_host(evidence["host"], root)
    if evidence["counts"] != list(counts):
        raise IntegrityError("retained collection/pass counts disagree")
    _audit_scoped_checks(evidence, root, source, counts)
    _audit_reuse(evidence, root, source)
    return {
        "status": "PASS",
        "tests_collected": counts[0],
        "tests_passed": counts[1],
        "bindings": bindings,
        "evidence_scope": scope,
        "operation": receipt["operation"],
    }


def _audit_reuse(evidence: JSONObject, root: Path, source: Path) -> None:
    reference = evidence["source_verification"]
    if reference is None:
        return
    if evidence["operation"] != "development-campaign":
        raise IntegrityError("only campaign qualification may reuse earlier exact engineering evidence")
    prior, original = _evidence(reference, root)
    if prior.get("operation") != "prepare-precision" or original.get("source_verification") is not None:
        raise IntegrityError("verification reuse must originate from one independently executed pilot gate")
    audit_verification(reference, artifact_root=root, source_root=source)
    for key in (
        "code",
        "environment",
        "work_directory",
        "commands",
        "counts",
        "host",
        "evidence_scope",
        "catalog",
        "pytest_observations",
        "packaging",
        "mutations",
    ):
        if original.get(key) != evidence[key]:
            raise IntegrityError("reused verification changed its actual command evidence")
    old, new = object_json(original["bindings"]), object_json(evidence["bindings"])
    if any(old[role] != new[role] for role in BOUND_ROLES if role != "data"):
        raise IntegrityError(
            "reused verification changed a verified code/config/calendar/environment identity"
        )


def _evidence(reference: JSONValue, root: Path) -> tuple[JSONObject, JSONObject]:
    receipt = read_object(reference, root)
    refs = receipt.get("logs")
    if not isinstance(refs, list) or len(refs) != 1:
        raise IntegrityError("one complete observed verification record required")
    result = read_object(refs[0], root)
    return receipt, read_object(result.get("evidence"), root)


def rebind_campaign_verification(
    prior_ref: JSONValue,
    *,
    artifact_root: Path,
    source_root: Path,
    bindings: JSONObject,
    qualification: JSONValue,
) -> JSONObject:
    """Reuse successful unchanged code checks, adding actual generated qualification.

    This creates a new scoped receipt without pretending to rerun the commands.
    Only the fixed source value cutoff may differ; source admission is separately
    rechecked by the production freeze and worker.
    """
    audit_verification(prior_ref, artifact_root=artifact_root, source_root=source_root)
    prior, evidence = _evidence(prior_ref, artifact_root)
    if prior["operation"] != "prepare-precision":
        raise IntegrityError("campaign verification must derive from the original observed pilot gate")
    updated = evidence | {
        "operation": "development-campaign",
        "bindings": bindings,
        "qualification": qualification,
        "source_verification": prior_ref,
        "reason": "same observed engineering checks plus separately supervised generated qualification",
    }
    result = save_object(
        artifact_root,
        {
            "schema": "s10-test-result-v2",
            "status": "PASS",
            "code_sha256": bindings["code"],
            "environment_sha256": bindings["environment"],
            "evidence": save_object(artifact_root, updated),
        },
    )
    receipt = save_object(
        artifact_root, prior | {"operation": "development-campaign", "bindings": bindings, "logs": [result]}
    )
    audit_verification(receipt, artifact_root=artifact_root, source_root=source_root)
    return receipt
