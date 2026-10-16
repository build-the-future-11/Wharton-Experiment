"""Generated schema, durable-failure and real phase-interruption assertions."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from test_data_audit import _contract, _dates, _panel, _write_sources
from test_supervisor_audit import run_generated, setup_generated

from frem_s10.data.admission import admit_sources
from frem_s10.data.features import build_origin_records
from frem_s10.data.source_reader import ScopedSourceReader
from frem_s10.errors import ArtifactIOError, ContractError, DataAdmissionError
from frem_s10.identities import object_json, parse_json
from frem_s10.portfolio.actions import Valuation
from frem_s10.runtime.artifacts import publish_json
from frem_s10.runtime.journal import write_failure_receipt


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema_version", 0),
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", "1"),
        ("currency", "EUR"),
        ("price_units", "adjusted_total_return"),
        ("dividend_units", "reinvested"),
        ("action_convention", "terminal_zero_supported"),
        ("availability_convention", "same_open"),
    ),
)
def test_every_source_convention_and_schema_rejects_unsupported_values(
    field: str, value: str | int | bool
) -> None:
    """TEST-004/006: schema and every declared source convention are strict."""
    contract = _contract(_dates(weeks=5))
    with pytest.raises(ContractError):
        if field == "schema_version":
            replace(contract, schema_version=cast(int, value))
        else:
            assert isinstance(value, str)
            constructors = {
                "currency": lambda: replace(contract, currency=value),
                "price_units": lambda: replace(contract, price_units=value),
                "dividend_units": lambda: replace(contract, dividend_units=value),
                "action_convention": lambda: replace(contract, action_convention=value),
                "availability_convention": lambda: replace(contract, availability_convention=value),
            }
            constructors[field]()


def test_source_asset_order_and_terminal_event_cannot_be_silently_reinterpreted(tmp_path: Path) -> None:
    """TEST-002/007: wrong asset axis and explicit terminal metadata both reject."""
    dates = _dates(weeks=5)
    reader, identities = _write_sources(tmp_path, dates)
    with pytest.raises(DataAdmissionError, match="ordered"):
        admit_sources(reader, (identities[1], identities[0], identities[2]), contract=_contract(dates))
    path = tmp_path / "SPY.json"
    raw = object_json(parse_json(path.read_bytes()))
    raw["terminal_events"] = [{"date": dates[-1].isoformat(), "liquidation_value": 0.0}]
    payload = json.dumps(raw).encode()
    path.write_bytes(payload)
    updated = (replace(identities[0], sha256=hashlib.sha256(payload).hexdigest()), *identities[1:])
    with pytest.raises(ContractError, match="fields"):
        admit_sources(ScopedSourceReader(tmp_path, updated), updated, contract=_contract(dates))


def test_current_quote_affects_valuation_but_not_available_context() -> None:
    """TEST-012: the current execution quote is excluded from historical features."""
    panel = _panel()
    original = build_origin_records(panel)[0]
    prices = panel.open.copy()
    prices[original.anchor_index] *= 2
    changed = build_origin_records(replace(panel, open=prices))[0]
    np.testing.assert_array_equal(original.x, changed.x)
    assert original.record_id == changed.record_id
    shares = np.array([1.0, 2.0, 0.0])
    before = Valuation(shares * panel.open[original.anchor_index], 10.0)
    after = Valuation(shares * prices[original.anchor_index], 10.0)
    assert after.nav - 10 == 2 * (before.nav - 10)
    assert not np.array_equal(original.g, changed.g)


def test_failure_before_temp_creation_retains_no_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEST-074: disk failure before a temporary file cannot publish a result."""

    def failed(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError("constructed disk full before temp")

    monkeypatch.setattr(tempfile, "mkstemp", failed)
    with pytest.raises(ArtifactIOError, match="existing bytes retained") as caught:
        publish_json(tmp_path, "success.json", {"state": "COMPLETE"})
    assert isinstance(caught.value.__cause__, OSError)
    assert tuple(tmp_path.iterdir()) == ()


def test_failure_of_last_durable_destination_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEST-079: unavailable fallback storage cannot be reported as saved evidence."""
    retained = tmp_path / "retained.json"
    retained.write_text('{"state":"CONSUMED"}\n')
    before = retained.read_bytes()

    def denied(*args: object, **kwargs: object) -> int:
        raise PermissionError("constructed storage unavailable")

    monkeypatch.setattr(os, "open", denied)
    with pytest.raises(ArtifactIOError, match="failure receipt unavailable") as caught:
        write_failure_receipt(tmp_path, {"state": "FAILED"})
    assert isinstance(caught.value.__cause__, PermissionError)
    assert not (tmp_path / "failure.json").exists() and retained.read_bytes() == before


@pytest.mark.parametrize("phase", ("forecast", "ledger", "analysis"))
def test_sigint_after_real_generated_phase_retains_prefix_and_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """TEST-077: SIGINT follows actual generated phases and cannot auto-resume."""
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    package = Path(__file__).resolve().parents[1]
    retained = output / "attempt" / "phase-evidence"
    code = f"""
import sys, pathlib, os, signal
sys.path[:0] = {json.dumps([str(package / "src"), str(package / "tests")])}
from test_model_audit import first_transition
from test_catalog_ledger_audit import fixed_path
from frem_s10.experiment.model_archive import ModelArchiveWriter
from frem_s10.runtime.artifacts import publish_json
from frem_s10.analysis.bootstrap import relative_gain
import numpy as np
root = pathlib.Path({str(retained)!r})
root.mkdir()
(root / 'model').mkdir()
step = first_transition()
archive = ModelArchiveWriter(root / 'model', experiment_id='generated-interruption', evidence_scope='GENERATED_ENGINEERING', binding_id='a'*64, variants=('primary',))
archive.sink('primary', step)
publish_json(root, 'forecast.json', {{'state_id':step.state.record_id,'scope':'GENERATED_ENGINEERING'}})
if {phase!r} == 'forecast': os.kill(os.getpid(), signal.SIGINT)
state, events, marks = fixed_path(100.0, 0.01)
publish_json(root, 'ledger.json', {{'state_id':state.record_id,'nav':state.nav,'scope':'GENERATED_ENGINEERING'}})
if {phase!r} == 'ledger': os.kill(os.getpid(), signal.SIGINT)
gain = relative_gain(np.array([0.8,0.9]), np.ones(2))
publish_json(root, 'analysis.json', {{'gain':gain,'scope':'GENERATED_ENGINEERING'}})
os.kill(os.getpid(), signal.SIGINT)
"""
    result = run_generated(tmp_path, output, markers, identity, code, wall=15)
    assert result["state"] == "FAILED" and result["consumed"] is True
    assert result["worker_exit_code"] != 0 and len(tuple(markers.iterdir())) == 1
    assert (retained / "forecast.json").exists()
    assert (retained / "ledger.json").exists() is (phase != "forecast")
    assert (retained / "analysis.json").exists() is (phase == "analysis")
    assert "KeyboardInterrupt" in (output / "attempt/worker.stderr").read_text()


def test_warning_and_unexpected_chained_failure_remain_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEST-082: warnings and both causes survive a nonzero generated worker exit."""
    output, markers, identity = setup_generated(tmp_path, monkeypatch)
    code = """
import warnings
warnings.warn('constructed-warning-marker', RuntimeWarning)
try:
    raise ValueError('constructed-root-cause')
except ValueError as error:
    raise RuntimeError('constructed-outer-cause') from error
"""
    result = run_generated(tmp_path, output, markers, identity, code)
    assert result["state"] == "FAILED" and result["worker_exit_code"] != 0
    diagnostic = (output / "attempt/worker.stderr").read_text()
    assert all(
        text in diagnostic
        for text in (
            "constructed-warning-marker",
            "constructed-root-cause",
            "constructed-outer-cause",
            "Traceback",
            "direct cause",
        )
    )


def test_misspelled_pytest_marker_is_a_collection_error(tmp_path: Path) -> None:
    """TEST-085: strict marker configuration cannot silently omit misspelled tests."""
    test = tmp_path / "test_marker.py"
    test.write_text("import pytest\n@pytest.mark.s10_marker_typo\ndef test_typo():\n    assert True\n")
    config = Path(__file__).resolve().parents[1] / "pyproject.toml"
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-c",
            str(config),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            str(test),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0 and "s10_marker_typo" in result.stdout + result.stderr
