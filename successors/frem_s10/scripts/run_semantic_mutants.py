"""Run four fixed generated-only semantic mutations in retained isolated copies."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mutant:
    """One exact reviewed edit and a discriminating generated test."""

    name: str
    file: str
    before: str
    after: str
    test: str


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
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(source: Path, target: Path) -> None:
    target.mkdir(exist_ok=False)
    for name in ("src", "tests"):
        shutil.copytree(
            source / name, target / name, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
        )
    shutil.copy2(source / "pyproject.toml", target / "pyproject.toml")


def _run(target: Path, tests: tuple[str, ...], label: str) -> tuple[int, int, int]:
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(target / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    command = (
        sys.executable,
        "-c",
        "import pathlib,sys,frem_s10,pytest; "
        "assert pathlib.Path(frem_s10.__file__).resolve().is_relative_to(pathlib.Path.cwd()/'src'); "
        "raise SystemExit(pytest.main(sys.argv[1:]))",
        "-q",
        "-p",
        "no:cacheprovider",
        "--junitxml=result.xml",
        *tests,
    )
    with (target / "pytest.log").open("xb") as log:
        result = subprocess.run(
            command, cwd=target, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False
        )
    xml = ET.parse(target / "result.xml")
    failures = len(xml.findall(".//testcase/failure"))
    errors = len(xml.findall(".//testcase/error"))
    (target / "command.json").write_text(
        json.dumps(
            {
                "label": label,
                "command": command,
                "returncode": result.returncode,
                "scope": "GENERATED_ENGINEERING",
            },
            indent=2,
        )
        + "\n"
    )
    return result.returncode, failures, errors


def run(source: Path, output: Path) -> dict[str, object]:
    """Require passing unmodified references; collection/import errors never kill a mutant."""
    source = source.resolve(strict=True)
    output = output.absolute()
    output.mkdir(exist_ok=False)
    before = {item.file: _hash(source / "src/frem_s10" / item.file) for item in MUTANTS}
    baseline = output / "baseline"
    _copy(source, baseline)
    code, failures, errors = _run(baseline, tuple(item.test for item in MUTANTS), "baseline")
    if code != 0 or failures or errors:
        raise RuntimeError(
            "unmodified mutant reference tests failed; retained baseline prevents false kill claims"
        )
    rows: list[dict[str, object]] = []
    for item in MUTANTS:
        target = output / item.name
        _copy(baseline, target)
        path = target / "src/frem_s10" / item.file
        text = path.read_text()
        if text.count(item.before) != 1:
            raise RuntimeError(f"mutant source anchor is not unique: {item.name}")
        path.write_text(text.replace(item.before, item.after))
        code, failures, errors = _run(target, (item.test,), item.name)
        rows.append(
            {
                **asdict(item),
                "original_sha256": before[item.file],
                "mutated_sha256": _hash(path),
                "returncode": code,
                "failures": failures,
                "errors": errors,
                "status": "KILLED" if code == 1 and failures == 1 and errors == 0 else "SURVIVED_OR_INVALID",
            }
        )
    after = {item.file: _hash(source / "src/frem_s10" / item.file) for item in MUTANTS}
    result: dict[str, object] = {
        "schema": "s10-semantic-mutants-v1",
        "scope": "GENERATED_ENGINEERING",
        "baseline_passed": True,
        "source_unchanged": before == after,
        "mutants": rows,
        "all_killed": all(row["status"] == "KILLED" for row in rows),
        "independent_empirical_verification": False,
    }
    (output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["all_killed"] or not result["source_unchanged"]:
        raise RuntimeError("mutation or source invariance gate failed; all copies and logs retained")
    return result


def main() -> None:
    """Run only the fixed four generated tests, preserving every isolated source copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
