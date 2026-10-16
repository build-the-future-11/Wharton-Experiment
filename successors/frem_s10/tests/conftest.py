"""Opt-in exact pytest observations, with no import-time filesystem activity."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the explicit exclusive output requested by verification."""
    parser.addoption("--s10-receipt", action="store", default=None)


def pytest_configure(config: pytest.Config) -> None:
    """Install observation hooks only when the caller selected a receipt path."""
    selected: object = config.getoption("--s10-receipt")
    if selected is not None:
        if not isinstance(selected, str) or not selected:
            raise pytest.UsageError("--s10-receipt requires a nonempty path")
        config.pluginmanager.register(Observations(Path(selected)), "s10-observations")


@dataclass(eq=False)
class Observations:
    """Collect every phase so teardown errors cannot become passing evidence."""

    path: Path
    collected: tuple[str, ...] = ()
    phases: dict[str, dict[str, str]] = field(default_factory=dict)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Retain every exact node ID, including parameter cases and skipped tests."""
        self.collected = tuple(item.nodeid for item in session.items)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Record failures, skips and xfail/xpass distinctly across all phases."""
        expected_failure: object = report.__dict__.get("wasxfail")
        outcome = "PASSED" if report.passed else "FAILED" if report.failed else "SKIPPED"
        if expected_failure is not None:
            outcome = "XFAILED" if report.skipped else "FAILED"
        self.phases.setdefault(report.nodeid, {})[report.when] = outcome

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
        """Exclusively publish observations; filesystem errors fail visibly."""
        outcomes = {node: observed_outcome(self.phases.get(node, {})) for node in self.collected}
        payload = {
            "schema": "s10-pytest-observations-v1",
            "collected": list(self.collected),
            "outcomes": outcomes,
            "exit_code": int(exitstatus),
        }
        with self.path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def observed_outcome(phases: dict[str, str]) -> str:
    """Incomplete execution and any failing phase prevent a passing result."""
    if "FAILED" in phases.values():
        return "FAILED"
    if "XFAILED" in phases.values():
        return "XFAILED"
    if "SKIPPED" in phases.values():
        return "SKIPPED"
    return "PASSED" if phases == dict.fromkeys(("setup", "call", "teardown"), "PASSED") else "FAILED"
