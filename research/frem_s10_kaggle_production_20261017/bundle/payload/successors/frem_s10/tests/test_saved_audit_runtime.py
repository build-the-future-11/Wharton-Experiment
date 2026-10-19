"""Bounded saved-audit admission and lifecycle tests; no campaign replay executes."""

import subprocess
import sys
from pathlib import Path

import pytest

from frem_s10.cli import main
from frem_s10.errors import ExecutionStateError, IntegrityError, ResourceLimitError
from frem_s10.identities import JSONObject, canonical_bytes
from frem_s10.runtime import saved_audit
from frem_s10.runtime.artifacts import publish_json
from frem_s10.runtime.consumption import consume
from frem_s10.runtime.journal import Journal
from frem_s10.runtime.pipeline import _finalization_failure
from frem_s10.runtime.receipts import read_unconsumed_refusal
from frem_s10.runtime.supervisor import Budget


def refusal(output: Path) -> JSONObject:
    value: JSONObject = {
        "state": "REJECTED_UNCONSUMED",
        "consumed": False,
        "error_type": "ResourceLimitError",
        "worker_dispatched": False,
        "replay_permitted": False,
    }
    journal = Journal(output / "timeline.jsonl")
    journal.append({"state": "PREFLIGHT"})
    journal.append(value)
    publish_json(output, "completion.json", value)
    return value


def test_retained_resource_refusal_cannot_hide_consumption(tmp_path: Path) -> None:
    output, markers = tmp_path / "output", tmp_path / "markers"
    output.mkdir()
    markers.mkdir()
    expected = refusal(output)
    assert (
        read_unconsumed_refusal(output, consumption_root=markers, operation="saved-audit", freeze_id="a" * 64)
        == expected
    )
    consume(
        markers, operation="saved-audit", freeze_id="a" * 64, attempt_id="attempt", output_path=str(output)
    )
    with pytest.raises(IntegrityError, match="after consumption"):
        read_unconsumed_refusal(output, consumption_root=markers, operation="saved-audit", freeze_id="a" * 64)


def test_resource_refusal_with_partial_lifecycle_is_not_retryable(tmp_path: Path) -> None:
    output, markers = tmp_path / "output", tmp_path / "markers"
    output.mkdir()
    markers.mkdir()
    refusal(output)
    with (output / "timeline.jsonl").open("ab") as stream:
        stream.write(b"partial")
    with pytest.raises(IntegrityError, match="partial"):
        read_unconsumed_refusal(output, consumption_root=markers, operation="precision", freeze_id="a" * 64)


def test_finalization_failure_retains_consumed_disposition(tmp_path: Path) -> None:
    result = _finalization_failure(tmp_path, "campaign", "c" * 64, IntegrityError("bad saved metric"))
    assert result["state"] == "FAILED_CONSUMED" and result["consumed"] is True
    assert result["automatic_retry_permitted"] is False
    original = (tmp_path / "finalization-failure.json").read_bytes()
    retry = _finalization_failure(tmp_path, "campaign", "c" * 64, IntegrityError("different failure"))
    assert "retention_error" in retry
    assert (tmp_path / "finalization-failure.json").read_bytes() == original


def test_saved_audit_cli_requires_explicit_execution() -> None:
    with pytest.raises(SystemExit) as failure:
        main(["audit", "--manifest", "unused.json"])
    assert failure.value.code == 2


def test_saved_audit_module_is_import_light() -> None:
    command = "import sys; import frem_s10.runtime.saved_audit; assert 'numpy' not in sys.modules; assert 'frem_s10.runtime.results' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-B", "-c", command], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_saved_audit_worker_rejects_wrong_owned_output_before_publication(tmp_path: Path) -> None:
    markers, output = tmp_path / "markers", tmp_path / "output"
    markers.mkdir()
    output.mkdir()
    consume(
        markers,
        operation="saved-audit",
        freeze_id="a" * 64,
        attempt_id="attempt",
        output_path=str(tmp_path / "different"),
    )
    with pytest.raises(ExecutionStateError, match="exact consumed output"):
        saved_audit.execute_saved_audit(
            request_path=tmp_path / "missing.json",
            request_sha256="a" * 64,
            artifact_root=tmp_path,
            source_root=tmp_path,
            consumption_root=markers,
            attempt_id="attempt",
            output=output,
        )
    assert not tuple(output.iterdir())


@pytest.mark.parametrize("raise_refusal", (False, True))
def test_saved_audit_uses_fixed_guarded_dispatch_and_preserves_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raise_refusal: bool
) -> None:
    markers, outputs = tmp_path / "markers", tmp_path / "outputs"
    markers.mkdir()
    outputs.mkdir()
    reference: JSONObject = {"path": "request.json", "sha256": "a" * 64}

    def request(reference: object, root: Path, source: Path) -> tuple[JSONObject, str]:
        return {}, "a" * 64

    def dispatch(**kwargs: object) -> JSONObject:
        assert kwargs["operation"] == "saved-audit"
        budget = kwargs["budget"]
        assert isinstance(budget, Budget) and (budget.rss_mib, budget.wall_seconds, budget.output_bytes) == (
            256,
            3600,
            2 * 2**30,
        )
        command = kwargs["command"]
        assert isinstance(command, tuple)
        assert command[1:3] == ("-B", "-X")
        assert command[3] == f"pycache_prefix={outputs / 'attempt' / 'empty_bytecode'}"
        assert command[4:6] == ("-m", "frem_s10.runtime.saved_audit")
        output = outputs / "attempt"
        output.mkdir()
        result = refusal(output)
        if raise_refusal:
            raise ResourceLimitError("generated host refusal")
        return result

    monkeypatch.setattr(saved_audit, "_request", request)
    monkeypatch.setattr(saved_audit, "execute_operation", dispatch)
    manifest: JSONObject = {
        "schema": "s10-saved-audit-launch-v1",
        "request": reference,
        "artifact_root": str(tmp_path),
        "source_root": str(tmp_path),
        "consumption_root": str(markers),
        "output_root": str(outputs),
        "attempt_id": "attempt",
    }
    path = tmp_path / "launch.json"
    path.write_bytes(canonical_bytes(manifest))
    result = saved_audit.run_saved_audit(path)
    assert result["state"] == "REJECTED_UNCONSUMED" and result["consumed"] is False
    assert not tuple(markers.iterdir())


def test_invalid_saved_audit_manifest_cannot_supply_arbitrary_command(tmp_path: Path) -> None:
    path = tmp_path / "launch.json"
    path.write_bytes(canonical_bytes({"schema": "s10-saved-audit-launch-v1", "command": ["arbitrary"]}))
    from frem_s10.errors import ContractError

    with pytest.raises(ContractError, match="exact schema"):
        saved_audit.run_saved_audit(path)
