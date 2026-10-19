from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _notebook_sources() -> list[str]:
    notebook = json.loads((ROOT / "kaggle" / "frem_s12_campaign.ipynb").read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_kaggle_notebook_code_is_parseable() -> None:
    for source in _notebook_sources():
        ast.parse(source)


def test_kaggle_install_is_isolated_and_offline() -> None:
    source = "\n".join(_notebook_sources())
    assert 'PACKAGE_VERSION = "0.1.5"' in source
    assert 'SITE = Path("/kaggle/working/frem-s12-site")' in source
    assert '"--no-index"' in source
    assert '"--no-deps"' in source
    assert '"--no-compile"' in source
    assert '"--target"' in source
    assert '"--force-reinstall"' not in source
    assert '"PYTHONPATH": str(SITE)' in source
    assert '"PYTHONWARNINGS": "error"' in source


def test_kaggle_notebook_fails_closed_and_manifests_outputs() -> None:
    source = "\n".join(_notebook_sources())
    assert "Output root must be absent for a fresh run" in source
    assert "Kaggle input contains forbidden symlinks" in source
    assert '"input_inventory_sha256"' in source
    assert 'write_new_json(OUTPUT / "command.json", command_result)' in source
    assert '"frem-s12-kaggle-run-manifest-v1"' in source
    assert 'write_new_json(OUTPUT / "run-manifest.json", manifest)' in source
    assert "if postcondition_error is not None:" in source
    assert "Kaggle output contains forbidden symlinks or special files" in source
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f'"{variable}": "1"' in source


def test_p0v5_changes_delivery_only() -> None:
    protocol = json.loads((ROOT / "src" / "frem_s12" / "protocol_p0_v5.json").read_text(encoding="utf-8"))
    assert protocol["version"] == 5
    assert protocol["base_protocol_sha256"] == (
        "8c746364f07224b4790c7424f07eceb7da6a2d407c522e250c4a8c073b7d3002"
    )
    assert protocol["qualification_timing_repeats"] == 3
    assert protocol["model"] == "unchanged_from_p0v4"
    assert protocol["cells"] == "unchanged_from_p0v4"
    assert protocol["thresholds"] == "unchanged_from_p0v4"
    assert protocol["inference"] == "unchanged_from_p0v4"


def test_p0v6_changes_evidence_validation_only() -> None:
    protocol = json.loads((ROOT / "src" / "frem_s12" / "protocol_p0_v6.json").read_text())
    assert protocol["version"] == 6
    assert protocol["market_outcomes_observed_before_amendment"] is False
    assert protocol["model"] == "unchanged_from_p0v5"
    assert protocol["cells"] == "unchanged_from_p0v5"
    assert protocol["thresholds"] == "unchanged_from_p0v5"
    assert protocol["inference"] == "unchanged_from_p0v5"


def test_p0v7_closes_evidence_chain_without_changing_science() -> None:
    protocol = json.loads((ROOT / "src" / "frem_s12" / "protocol_p0_v7.json").read_text())
    assert protocol["version"] == 7
    assert protocol["base_protocol_sha256"] == (
        "c540f44bf9ca46164af3b62d3e8d01f2c99a359d435c452c7138a1e7bdd088f2"
    )
    assert protocol["market_outcomes_observed_before_amendment"] is False
    assert protocol["artifact_schema"] == 3
    assert protocol["checkpoint_schema"] == 3
    assert protocol["model"] == "unchanged_from_p0v6"
    assert protocol["cells"] == "unchanged_from_p0v6"
    assert protocol["thresholds"] == "unchanged_from_p0v6"
    assert protocol["inference"] == "unchanged_from_p0v6"


def test_p0v8_closes_global_consumption_and_policy_reference_gaps() -> None:
    protocol = json.loads((ROOT / "src" / "frem_s12" / "protocol_p0_v8.json").read_text())
    assert protocol["version"] == 8
    assert protocol["market_outcomes_observed_before_amendment"] is False
    assert protocol["model"] == "unchanged_from_p0v7"
    assert protocol["cells"] == "unchanged_from_p0v7"
