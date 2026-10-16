"""Fixed generated semantic mutations of isolated exact-candidate copies.

The parent verifier provides aggregate resource and network boundaries. This
producer additionally requires fresh host admission before launching any test.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from frem_s10.errors import IntegrityError, ResourceLimitError, S10Error
from frem_s10.identities import JSONObject, JSONValue, checked_json, object_json, parse_json
from frem_s10.runtime.artifacts import _plain_directory, publish_json
from frem_s10.runtime.provenance import PACKAGE, THREAD_ENV, expected_code_files
from frem_s10.runtime.resources import check_host


@dataclass(frozen=True)
class Mutant:
    """One unchanged reviewed semantic edit and its fixed discriminating test."""

    name: str
    file: str
    before: str
    after: str
    test: str
    extra_edits: tuple[tuple[str, str], ...] = ()


MUTANTS = (
    Mutant(
        "maturity_equality",
        "model/feedback.py",
        "if outcome.available_us >= decision_us:",
        "if outcome.available_us > decision_us:",
        "tests/test_model_audit.py::test_current_target_has_no_issuance_input_and_feedback_is_delayed",
    ),
    Mutant(
        "omit_exit_cost",
        "portfolio/policy.py",
        "terminal = (1.0 - one_way_cost) * forecast.g + forecast.d",
        "terminal = forecast.g + forecast.d",
        "tests/test_policy_benchmarks_audit.py::test_price_only_exit_fee_preserves_dividend_cash",
    ),
    Mutant(
        "omit_cross_expert_terms",
        "math/scores.py",
        "weights @ self.a - 0.5 * weights @ self.b @ weights",
        "weights @ self.a - 0.5 * weights @ np.diag(self.b)",
        "tests/test_math_audit.py::test_energy_score_cross_terms_and_law_invariance",
    ),
    Mutant(
        "publish_before_consumption_validation",
        "runtime/worker.py",
        '    if (\n        marker.get("state"),',
        '    publish_json(output, "premature-output.json", {"mutant": True})\n    if (\n        marker.get("state"),',
        "tests/test_production_runtime.py::test_worker_refuses_consumed_identity_from_another_output_before_import_or_values",
    ),
    Mutant(
        "rescore_current_library",
        "model/engine.py",
        "pending[item.record_id], item, decision_us=request.decision_us, transform=state.transform",
        "replace(pending[item.record_id], experts=issue_experts(state.library, "
        "pending[item.record_id].request.x, decision_us=request.decision_us, transform=state.transform, "
        "config=state.config)), item, decision_us=request.decision_us, transform=state.transform",
        "tests/test_model_audit.py::test_issued_forecast_is_not_recomputed_after_library_changes",
    ),
    Mutant(
        "clip_infeasible_face",
        "math/simplex.py",
        "if not feasible:\n        return alpha, FaceCertificate(face, condition, False, False, "
        "float(np.abs(alpha).max()), multiplier)",
        "if not feasible:\n        alpha = np.maximum(alpha, 0)\n        alpha /= float(alpha.sum())",
        "tests/test_catalog_numerics_audit.py::"
        "test_tiny_negative_simplex_face_is_corrected_only_within_tolerance[1e-10-False]",
    ),
    Mutant(
        "interruption_as_resumable_success",
        "runtime/supervisor.py",
        'result: dict[str, JSONValue] = {\n        "state": state,',
        'result: dict[str, JSONValue] = {\n        "state": "COMPLETE" if state == "INTERRUPTED" else state,'
        '\n        "resume_permitted": state == "INTERRUPTED",',
        "tests/test_supervisor_audit.py::test_cancel_terminates_owned_child",
    ),
    Mutant(
        "post_trade_dividend_ownership",
        "portfolio/ledger.py",
        "    credited = state.shares * dividend\n",
        "    global _MUTATED_LAST_DIVIDEND\n    _MUTATED_LAST_DIVIDEND = dividend\n    credited = np.zeros(3)\n",
        "tests/test_ledger_audit.py::test_new_buyers_do_not_receive_opening_dividend_and_repeated_marks_charge_nothing",
        (
            (
                "        shares=shares,\n        cash=solution.cash,\n        fees=state.fees + solution.fee,",
                "        shares=shares,\n        cash=solution.cash + float((shares * _MUTATED_LAST_DIVIDEND).sum()),"
                "\n        dividends=state.dividends + shares * _MUTATED_LAST_DIVIDEND,"
                "\n        fees=state.fees + solution.fee,",
            ),
            (
                '        kind="FINAL_LIQUIDATION" if final_liquidation else "REBALANCE",'
                "\n        price_pnl=np.zeros(3),\n        dividends=np.zeros(3),",
                '        kind="FINAL_LIQUIDATION" if final_liquidation else "REBALANCE",'
                "\n        price_pnl=np.zeros(3),\n        dividends=shares * _MUTATED_LAST_DIVIDEND,",
            ),
        ),
    ),
    Mutant(
        "charge_hypothetical_liquidation_fee",
        "portfolio/ledger.py",
        "    value = state.cash + (1 - one_way_cost) * float(state.risky_dollars.sum())\n",
        "    value = state.cash + (1 - one_way_cost) * float(state.risky_dollars.sum())\n"
        '    object.__setattr__(state, "cash", state.cash - one_way_cost * float(state.risky_dollars.sum()))\n'
        '    object.__setattr__(state, "fees", state.fees + one_way_cost * float(state.risky_dollars.sum()))\n',
        "tests/test_ledger_audit.py::test_entering_share_dividend_before_endpoint_sale_and_negative_path",
    ),
    Mutant(
        "pre_fee_target_dollars",
        "math/fees.py",
        "turnover = np.sum(np.abs(q[:, None] * weights - current), axis=1)",
        "turnover = np.sum(np.abs(nav * weights - current), axis=1)",
        "tests/test_math_audit.py::test_exact_fee_roots_and_batch_agree_with_decimal",
        (
            ("risky = q[:, None] * weights", "risky = nav * weights"),
            ("cash_after = q * (1.0 - np.sum(weights, axis=1))", "cash_after = q - np.sum(risky, axis=1)"),
        ),
    ),
    Mutant(
        "omit_benchmark_control",
        "experiment/matrix.py",
        '            result.extend(CellSpec(fold, "benchmark", arm, cost) for arm in BENCHMARKS)',
        '            result.extend(CellSpec(fold, "benchmark", arm, cost) for arm in BENCHMARKS if arm != "equal100")',
        "tests/test_experiment_audit.py::test_exact_192_registry_and_no_combined_or_cost_shifted_ablations",
    ),
)

BUDGET_SECONDS = 600.0


def _baseline_tests() -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.test for item in MUTANTS))


def _recipes() -> list[JSONValue]:
    return [
        {
            "name": item.name,
            "file": item.file,
            "before": item.before,
            "after": item.after,
            "test": item.test,
            "extra_edits": [list(edit) for edit in item.extra_edits],
        }
        for item in MUTANTS
    ]


RUNNER = """import json,pathlib,sys
import frem_s10,pytest
origin=pathlib.Path(frem_s10.__file__).resolve()
expected=pathlib.Path.cwd()/"src/frem_s10/__init__.py"
assert origin == expected, (origin,expected)
class Observations:
    def __init__(self):
        self.nodes=[]; self.phases={}; self.exceptions={}
    def pytest_collection_finish(self,session):
        self.nodes=[item.nodeid for item in session.items]
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self,item,call):
        if call.excinfo is not None:
            self.exceptions.setdefault(item.nodeid,{})[call.when]={"type":call.excinfo.type.__module__+"."+call.excinfo.type.__qualname__,"message":str(call.excinfo.value)}
        yield
    def pytest_runtest_logreport(self,report):
        self.phases.setdefault(report.nodeid,{})[report.when]=report.outcome
observations=Observations()
code=int(pytest.main(sys.argv[1:],plugins=[observations]))
value={"schema":"s10-mutation-observations-v1","origin":str(origin),"collected":observations.nodes,"phases":observations.phases,"exceptions":observations.exceptions,"exit_code":code}
with pathlib.Path("observations.json").open("x") as stream:
    json.dump(value,stream,sort_keys=True,allow_nan=False)
raise SystemExit(code)
"""


def _hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError("regular mutation evidence file required")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _small(path: Path) -> JSONObject:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_194_304:
        raise IntegrityError("bounded regular mutation metadata required")
    return object_json(parse_json(path.read_bytes()))


def _inventory(source: Path) -> JSONObject:
    return {
        path.relative_to(source / PACKAGE).as_posix(): _hash(path) for path in expected_code_files(source)
    }


def _copy(source: Path, target: Path, inventory: JSONObject) -> None:
    target.mkdir(exist_ok=False)
    for name, identity in inventory.items():
        path = source / PACKAGE / name
        if _hash(path) != identity:
            raise IntegrityError("candidate source changed during isolated copying")
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def _mutated(source: Path, item: Mutant) -> bytes:
    raw = (source / PACKAGE / "src/frem_s10" / item.file).read_bytes()
    for before, after in ((item.before, item.after), *item.extra_edits):
        encoded = before.encode()
        if raw.count(encoded) != 1:
            raise IntegrityError(f"reviewed mutation anchor is not unique: {item.name}")
        raw = raw.replace(encoded, after.encode(), 1)
    return raw


def _command(target: Path, tests: tuple[str, ...]) -> tuple[str, ...]:
    return (
        sys.executable,
        "-B",
        "-X",
        f"pycache_prefix={target / 'empty_bytecode'}",
        "-c",
        RUNNER,
        "-c",
        "pyproject.toml",
        "-q",
        "-p",
        "no:cacheprovider",
        "-W",
        "error",
        "--basetemp",
        str(target / "fixtures"),
        *tests,
    )


def _environment(target: Path) -> dict[str, str]:
    return dict(
        os.environ,
        **THREAD_ENV,
        PYTHONPATH=str(target / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )


def _observations(value: JSONObject, target: Path, tests: tuple[str, ...], *, mutant: bool) -> None:
    if (
        value.get("schema") != "s10-mutation-observations-v1"
        or value.get("origin") != str(target / "src/frem_s10/__init__.py")
        or value.get("collected") != list(tests)
    ):
        raise IntegrityError("mutation test collection or isolated import origin differs")
    phases, exceptions = object_json(value.get("phases")), object_json(value.get("exceptions"))
    expected = {"setup": "passed", "call": "failed" if mutant else "passed", "teardown": "passed"}
    if phases != dict.fromkeys(tests, expected) or value.get("exit_code") != int(mutant):
        raise IntegrityError("mutation run did not execute exact expected phases")
    if not mutant:
        if exceptions:
            raise IntegrityError("baseline must have no exceptions")
        return
    _assertion(exceptions, tests)


def _assertion(exceptions: JSONObject, tests: tuple[str, ...]) -> None:
    if len(tests) != 1 or set(exceptions) != set(tests):
        raise IntegrityError("only the exact target assertion may kill a mutation")
    phases = object_json(exceptions[tests[0]])
    if set(phases) != {"call"}:
        raise IntegrityError("setup/import/teardown errors cannot kill a semantic mutation")
    failure = object_json(phases["call"])
    if failure.get("type") == "builtins.AssertionError":
        return
    if failure.get("type") != "_pytest.outcomes.Failed" or not str(failure.get("message")).startswith(
        "DID NOT RAISE"
    ):
        raise IntegrityError("non-assertion test errors cannot kill a semantic mutation")


def _run(target: Path, tests: tuple[str, ...], label: str, *, mutant: bool, remaining: float) -> JSONObject:
    command, started, failure = _command(target, tests), time.monotonic(), None
    timeout = min(60.0, remaining)
    if timeout <= 0:
        raise ResourceLimitError("fixed mutation wall-time budget exhausted before dispatch")
    with (target / "pytest.log").open("xb") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=target,
                env=_environment(target),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            code = completed.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            code, failure = -1, f"{type(exc).__name__}: {exc}"
            log.write(f"\nMUTATION_RUN_FAILED: {failure}\n".encode())
    command_id = publish_json(
        target,
        "command.json",
        {
            "argv": list(command),
            "cwd": str(target),
            "pythonpath": str(target / "src"),
            "thread_environment": dict(THREAD_ENV),
            "exit_code": code,
            "failure": failure,
            "wall_seconds": time.monotonic() - started,
            "timeout_seconds": timeout,
        },
    )
    valid, observations_id = False, None
    try:
        if code != int(mutant) or (target / "pytest.log").stat().st_size > 4_194_304:
            raise IntegrityError("unexpected mutation exit code or oversized log")
        _observations(_small(target / "observations.json"), target, tests, mutant=mutant)
        observations_id, valid = _hash(target / "observations.json"), True
    except (S10Error, OSError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    return {
        "label": label,
        "tests": list(tests),
        "command_sha256": command_id,
        "observations_sha256": observations_id,
        "log_sha256": _hash(target / "pytest.log"),
        "status": "KILLED" if valid and mutant else "BASELINE_PASSED" if valid else "INVALID_OR_SURVIVED",
        "failure": failure,
    }


def run_mutation_check(source_root: Path, output: Path) -> JSONObject:
    """Execute fixed reviewed edits only in fresh copies after host admission."""
    source = _plain_directory(source_root)
    _plain_directory(output.parent)
    output.mkdir(exist_ok=False)
    output = _plain_directory(output)
    started = time.monotonic()
    try:
        host = checked_json(check_host())
        publish_json(output, "host.json", host)
        if object_json(host).get("status") != "PASS":
            raise ResourceLimitError("fresh host admission refused generated mutation execution")
        result = _execute(source, output, started)
        if result["status"] != "PASS":
            raise IntegrityError("fixed candidate mutation campaign failed; every run retained")
        return audit_mutation_receipt(output, source)
    except Exception as exc:
        publish_json(
            output,
            "failure.json",
            {
                "schema": "s10-mutation-failure-v1",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scope": "GENERATED_ENGINEERING",
            },
        )
        raise


def _execute(source: Path, output: Path, started: float) -> JSONObject:
    if not 10 <= len(MUTANTS) <= 20 or len({item.name for item in MUTANTS}) != len(MUTANTS):
        raise IntegrityError("the reviewed campaign must contain 10 to 20 distinct fixed mutations")
    inventory = _inventory(source)
    baseline = output / "baseline"
    _copy(source, baseline, inventory)
    rows: list[JSONValue] = [
        _run(
            baseline,
            _baseline_tests(),
            "baseline",
            mutant=False,
            remaining=BUDGET_SECONDS - (time.monotonic() - started),
        )
    ]
    if object_json(rows[0])["status"] == "BASELINE_PASSED":
        for item in MUTANTS:
            remaining = BUDGET_SECONDS - (time.monotonic() - started)
            if remaining <= 0:
                break
            target = output / item.name
            _copy(source, target, inventory)
            (target / "src/frem_s10" / item.file).write_bytes(_mutated(source, item))
            rows.append(_run(target, (item.test,), item.name, mutant=True, remaining=remaining))
    elapsed = time.monotonic() - started
    success = (
        len(rows) == 1 + len(MUTANTS)
        and elapsed <= BUDGET_SECONDS
        and all(object_json(row)["status"] == "KILLED" for row in rows[1:])
        and _inventory(source) == inventory
    )
    result: JSONObject = {
        "schema": "s10-semantic-mutants-v2",
        "status": "PASS" if success else "FAIL",
        "scope": "GENERATED_ENGINEERING",
        "source_files": inventory,
        "mutations": _recipes(),
        "runs": rows,
        "host_sha256": _hash(output / "host.json"),
        "parent_resource_and_network_proof_required": True,
        "scientific_support_claimed": False,
        "wall_seconds": elapsed,
        "budget_seconds": BUDGET_SECONDS,
    }
    publish_json(output, "receipt.json", result)
    return result


def audit_mutation_receipt(output: Path, source_root: Path) -> JSONObject:
    """Recheck exact candidate/copy/command/assertion identities, without rerunning tests."""
    directory, source = _plain_directory(output), _plain_directory(source_root)
    value, inventory = _small(directory / "receipt.json"), _inventory(source)
    if (
        (
            value.get("schema"),
            value.get("status"),
            value.get("scope"),
            value.get("scientific_support_claimed"),
            value.get("parent_resource_and_network_proof_required"),
        )
        != ("s10-semantic-mutants-v2", "PASS", "GENERATED_ENGINEERING", False, True)
        or value.get("source_files") != inventory
        or value.get("mutations") != _recipes()
    ):
        raise IntegrityError("mutation receipt differs from exact candidate and fixed generated scope")
    if (
        value.get("host_sha256") != _hash(directory / "host.json")
        or _small(directory / "host.json").get("status") != "PASS"
    ):
        raise IntegrityError("mutation host admission evidence differs")
    rows = value.get("runs")
    _audit_budget(value)
    if not isinstance(rows, list) or len(rows) != 1 + len(MUTANTS):
        raise IntegrityError("exact baseline and every fixed mutation run required")
    _audit_run(
        directory / "baseline",
        object_json(rows[0]),
        _baseline_tests(),
        inventory,
        mutant=False,
    )
    for item, row in zip(MUTANTS, rows[1:], strict=True):
        changed = inventory | {
            "src/frem_s10/" + item.file: hashlib.sha256(_mutated(source, item)).hexdigest()
        }
        _audit_run(directory / item.name, object_json(row), (item.test,), changed, mutant=True)
    return value


def _audit_budget(value: JSONObject) -> None:
    elapsed = value.get("wall_seconds")
    if (
        type(elapsed) not in (int, float)
        or not isinstance(elapsed, (int, float))
        or not 0 <= elapsed <= BUDGET_SECONDS
        or value.get("budget_seconds") != BUDGET_SECONDS
    ):
        raise IntegrityError("fixed mutation cumulative wall-time budget was not met")


def _audit_copy(target: Path, inventory: JSONObject) -> None:
    actual: set[str] = set()
    for path in target.rglob("*"):
        relative = path.relative_to(target)
        if path.is_symlink():
            raise IntegrityError("isolated mutation copy contains a symlink")
        if relative.parts[0] in {"fixtures", "empty_bytecode"}:
            continue
        if path.is_file():
            actual.add(relative.as_posix())
    allowed = set(inventory) | {"command.json", "observations.json", "pytest.log"}
    if actual != allowed:
        raise IntegrityError("isolated mutation copy contains omitted or additional files")
    for name, expected in inventory.items():
        if _hash(target / name) != expected:
            raise IntegrityError("isolated mutation copy differs from exact reviewed edit")


def _audit_run(
    target: Path, row: JSONObject, tests: tuple[str, ...], inventory: JSONObject, *, mutant: bool
) -> None:
    _plain_directory(target)
    if (row.get("label"), row.get("tests"), row.get("status"), row.get("failure")) != (
        target.name,
        list(tests),
        "KILLED" if mutant else "BASELINE_PASSED",
        None,
    ):
        raise IntegrityError("mutation run identity or successful assertion evidence differs")
    for name in ("command", "observations"):
        _small(target / f"{name}.json")
        if row.get(f"{name}_sha256") != _hash(target / f"{name}.json"):
            raise IntegrityError("mutation metadata identity changed")
    command = _small(target / "command.json")
    timeout, elapsed = command.get("timeout_seconds"), command.get("wall_seconds")
    if (
        type(timeout) not in (int, float)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout <= 60
        or type(elapsed) not in (int, float)
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
    ):
        raise IntegrityError("mutation command timing evidence differs")
    if (
        command.get("argv") != list(_command(target, tests))
        or command.get("cwd") != str(target)
        or command.get("pythonpath") != str(target / "src")
        or command.get("thread_environment") != dict(THREAD_ENV)
        or command.get("exit_code") != int(mutant)
        or command.get("failure") is not None
    ):
        raise IntegrityError("mutation command or isolated environment differs")
    log = target / "pytest.log"
    if log.stat().st_size > 4_194_304 or row.get("log_sha256") != _hash(log):
        raise IntegrityError("mutation log identity or byte bound differs")
    _audit_copy(target, inventory)
    _observations(_small(target / "observations.json"), target, tests, mutant=mutant)


def main(argv: list[str] | None = None) -> int:
    """Run the fixed mutation operation only; arbitrary supplied edits are unsupported."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_mutation_check(args.source_root, args.output)
    except (S10Error, OSError, subprocess.SubprocessError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 6
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
