import json

from rejepa.experiments import (
    ExperimentCell,
    ExperimentManifest,
    completion_status,
    load_manifest,
    run_manifest,
    save_manifest,
)
from rejepa.mandate import ClientMandate
from rejepa.reporting import StrategyEvidence, render_strategy_markdown


def _manifest() -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id="engineering-only",
        phase="compatibility",
        dataset_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        protected_start=None,
        cells=(ExperimentCell("cell-1", "rejepa_full", 11, 0.1),),
    )


def test_manifest_hash_round_trip_and_freeze(tmp_path) -> None:
    manifest = _manifest().freeze()
    path = tmp_path / "manifest.json"
    save_manifest(path, manifest)
    loaded = load_manifest(path)
    assert loaded == manifest
    assert loaded.sha256 == manifest.sha256


def test_unfrozen_manifest_cannot_run(tmp_path) -> None:
    try:
        run_manifest(_manifest(), tmp_path, lambda cell: {"ok": True})
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("Unfrozen manifest ran")


def test_confirmatory_manifest_requires_protected_boundary() -> None:
    manifest = ExperimentManifest("confirmatory", "confirmatory", "a" * 64, "b" * 64, None, _manifest().cells)
    try:
        manifest.freeze()
    except ValueError as error:
        assert "protected boundary" in str(error)
    else:
        raise AssertionError("Confirmatory manifest froze without a boundary")


def test_runner_retains_complete_and_failed_receipts(tmp_path) -> None:
    cells = (
        ExperimentCell("ok", "baseline", 11, 0.1),
        ExperimentCell("bad", "candidate", 11, 0.1),
    )
    manifest = ExperimentManifest("eng", "compatibility", "a" * 64, "b" * 64, None, cells).freeze()

    def execute(cell: ExperimentCell) -> dict[str, object]:
        if cell.cell_id == "bad":
            raise RuntimeError("retained failure")
        return {"metric": 1.0}

    receipts = run_manifest(manifest, tmp_path, execute)
    status = completion_status(manifest, receipts)
    assert status["completed_cells"] == 1
    assert status["failed_cells"] == ["bad"]
    assert json.loads((tmp_path / "bad.receipt.json").read_text())["status"] == "FAILED"


def test_strategy_report_blocks_unsubstantiated_holdings() -> None:
    evidence = StrategyEvidence("NOT_EVALUATED", False, False, False, (("FAKE", 1.0),))
    report = render_strategy_markdown(ClientMandate(), evidence)
    assert "**BLOCKED:**" in report
    assert "FAKE: 100.0%" not in report
