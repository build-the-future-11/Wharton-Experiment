import json

from rejepa.cli import build_parser, main
from rejepa.engineering_data import ENGINEERING_LABEL


def test_cli_exposes_every_required_command() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "data-validate",
        "baseline-train",
        "jepa-train",
        "simulator-train",
        "memory-experiment",
        "full-train",
        "walk-forward",
        "portfolio-optimize",
        "report-generate",
        "synthetic-generate",
        "engineering-smoke",
    ):
        assert command in help_text


def test_report_cli_writes_blocked_evidence_draft(tmp_path, capsys) -> None:
    destination = tmp_path / "nested" / "strategy.md"
    assert main(["report-generate", str(destination)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "COMPLETE"
    assert "BLOCKED" in destination.read_text(encoding="utf-8")


def test_synthetic_dataset_and_engineering_smoke_run_end_to_end(tmp_path, capsys) -> None:
    data = tmp_path / "data"
    assert (
        main(
            [
                "synthetic-generate",
                str(data),
                "--assets",
                "3",
                "--dates",
                "48",
                "--context",
                "6",
                "--horizon",
                "2",
            ]
        )
        == 0
    )
    generated = json.loads(capsys.readouterr().out)
    assert generated["result"]["label"] == ENGINEERING_LABEL
    assert (
        main(
            [
                "data-validate",
                str(data / "panel.csv"),
                "--manifest",
                str(data / "panel.csv.manifest.json"),
            ]
        )
        == 0
    )
    validated = json.loads(capsys.readouterr().out)
    assert validated["result"]["valid"]
    assert validated["result"]["manifest_verified"]
    output = tmp_path / "smoke"
    assert main(["engineering-smoke", str(data / "training.npz"), str(output), "--epochs", "1"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["result"]["status"] == "ENGINEERING_COMPATIBILITY_PASS"
    assert receipt["result"]["matrix_status"]["complete"]
    assert (output / "ENGINEERING_RECEIPT.json").exists()
    assert (output / "manifest.json").exists()
    assert len(list((output / "cells").glob("*.receipt.json"))) == 2
