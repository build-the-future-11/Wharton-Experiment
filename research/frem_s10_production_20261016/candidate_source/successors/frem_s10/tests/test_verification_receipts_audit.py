"""Small adversarial checks for actual test, packaging and admission evidence."""

from pathlib import Path

import pytest

from frem_s10.errors import IntegrityError
from frem_s10.identities import JSONObject, checked_json
from frem_s10.runtime.provenance import save_object
from frem_s10.runtime.verification import _audit_host, _audit_packaging, _pytest_observations


def _evidence(root: Path) -> JSONObject:
    nodes = ["tests/test_a.py::test_a[0]", "tests/test_a.py::test_a[1]"]
    observations = save_object(
        root,
        {
            "schema": "s10-pytest-observations-v1",
            "collected": checked_json(nodes),
            "outcomes": dict.fromkeys(nodes, "PASSED"),
            "exit_code": 0,
        },
    )
    collection = save_object(
        root,
        {
            "schema": "s10-command-output-v1",
            "stdout": "\n".join(nodes) + "\n\n2 tests collected in 0.01s\n",
            "stderr": "",
        },
    )
    return {"pytest_observations": observations, "commands": [{"name": "collection", "output": collection}]}


def test_exact_parameter_cases_must_match_independent_collection(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    nodes, outcomes = _pytest_observations(evidence, tmp_path, (2, 2))
    assert len(nodes) == 2 and set(outcomes.values()) == {"PASSED"}
    evidence["commands"] = [
        {
            "name": "collection",
            "output": save_object(
                tmp_path,
                {
                    "schema": "s10-command-output-v1",
                    "stdout": "tests/test_a.py::test_a[0]\n\n2 tests collected in 0.01s\n",
                    "stderr": "",
                },
            ),
        }
    ]
    with pytest.raises(IntegrityError, match="node identities differ"):
        _pytest_observations(evidence, tmp_path, (2, 2))


@pytest.mark.parametrize("damage", ["skip", "duplicate", "orphan", "exit"])
def test_summary_pass_count_cannot_override_observation_failure(tmp_path: Path, damage: str) -> None:
    evidence = _evidence(tmp_path)
    nodes = ["tests/test_a.py::test_a[0]", "tests/test_a.py::test_a[1]"]
    outcomes = dict.fromkeys(nodes, "PASSED")
    if damage == "skip":
        outcomes[nodes[1]] = "SKIPPED"
    elif damage == "duplicate":
        nodes[1] = nodes[0]
    elif damage == "orphan":
        outcomes["tests/uncollected.py::test_extra"] = "PASSED"
    evidence["pytest_observations"] = save_object(
        tmp_path,
        {
            "schema": "s10-pytest-observations-v1",
            "collected": checked_json(nodes),
            "outcomes": checked_json(outcomes),
            "exit_code": int(damage == "exit"),
        },
    )
    with pytest.raises(IntegrityError):
        _pytest_observations(evidence, tmp_path, (2, 2))


def test_packaging_status_requires_matching_observed_command(tmp_path: Path) -> None:
    receipt: JSONObject = {"status": "PASS", "clean_install": True, "scientific_execution": False}
    evidence: JSONObject = {
        "packaging": save_object(tmp_path, receipt),
        "commands": [
            {
                "name": "packaging",
                "output": save_object(
                    tmp_path,
                    {
                        "stdout": '{"status":"FAIL","clean_install":false,"scientific_execution":false}',
                        "stderr": "",
                    },
                ),
            }
        ],
    }
    with pytest.raises(IntegrityError, match="differs"):
        _audit_packaging(evidence, tmp_path, tmp_path)


@pytest.mark.parametrize("damage", ["low", "pageout", "clock", "reported_memory"])
def test_host_pass_string_cannot_override_observed_gate(tmp_path: Path, damage: str) -> None:
    rows: list[JSONObject] = [
        {
            "timestamp": f"2026-10-16T00:00:0{i}+00:00",
            "page_size_bytes": 16384,
            "free_pages": 32768,
            "speculative_pages": 0,
            "pageouts": 10,
            "available_mib": 512.0,
        }
        for i in range(3)
    ]
    if damage == "low":
        rows[1].update({"free_pages": 32767, "available_mib": 32767 / 64})
    elif damage == "pageout":
        rows[2]["pageouts"] = 11
    elif damage == "clock":
        rows[1]["timestamp"] = rows[0]["timestamp"]
    else:
        rows[0]["available_mib"] = 1024.0
    reference = save_object(tmp_path, {"status": "PASS", "samples": checked_json(rows)})
    with pytest.raises(IntegrityError):
        _audit_host(reference, tmp_path)
