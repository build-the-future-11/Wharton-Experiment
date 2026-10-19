from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frem_s12.admission import (
    QUALIFICATION_REPEATS,
    _read_only_mount,
    _runtime_supported,
    qualify_environment,
    qualify_kaggle_environment,
    require_execution_receipts,
    validate_kaggle_environment_receipt,
)
from frem_s12.artifacts import read_fold, write_fold
from frem_s12.campaign import FOLDS
from frem_s12.consumption import claim_consumption
from frem_s12.errors import ConsumptionError, ContractError
from frem_s12.evidence import atomic_json, read_json, sha256_file
from frem_s12.experiment import Fold, run_fold
from frem_s12.generated import generated_panel
from frem_s12.model import ModelConfig
from frem_s12.scores import Law, energy_score, variogram_score


def test_qualification_is_json_safe() -> None:
    value = qualify_environment(benchmark_size=64)
    assert value["qualified"] is True
    assert value["deterministic"] is True
    assert value["scientific_admitted"] is False
    assert value["scope"] == "local_engineering"


def test_runtime_allowlist_is_exact() -> None:
    assert _runtime_supported((3, 12), "2.5.4") is True
    assert _runtime_supported((3, 13), "2.1.3") is True
    assert _runtime_supported((3, 12), "2.5.3") is False
    assert _runtime_supported((3, 13), "2.5.4") is False
    assert _runtime_supported((3, 14), "2.1.3") is False


def test_kaggle_qualification_uses_three_repeats() -> None:
    assert QUALIFICATION_REPEATS == 3


def test_kaggle_admission_receipt_is_semantically_reconstructed() -> None:
    source = "a" * 64
    search_seconds = [1.0, 2.0, 3.0]
    fixed_seconds = [4.0, 5.0, 6.0]
    projected_cpu = (3.0 * 144 + 6.0 * 120) / 3600
    projected_shard = 6.0 * 33 / 3600
    projected_bytes = 200 * 33
    environment: dict[str, Any] = {
        "schema": "frem-s12-environment-qualification-v1",
        "qualified": True,
        "scientific_admitted": True,
        "scope": "kaggle_scientific",
        "source_identity": source,
        "kaggle_detected": True,
        "runtime_ok": True,
        "python_ok": True,
        "numpy_ok": True,
        "input_read_only": True,
        "network_disabled_observed": True,
        "projection_ok": True,
        "expected_numpy": "2.1.3",
        "runtime": {"python": "3.13.15", "numpy": "2.1.3"},
        "benchmark": {
            "executed": True,
            "worst_declared_trial": 23,
            "latest_fixed_fold": "H",
            "repeats": 3,
            "timing_statistic": "maximum",
            "artifact_bytes_stable": True,
            "search_cells": 144,
            "fixed_cells": 120,
            "generated_search_seconds_samples": search_seconds,
            "generated_search_seconds": 3.0,
            "generated_fixed_seconds_samples": fixed_seconds,
            "generated_fixed_seconds": 6.0,
            "search_artifact_bytes_samples": [100, 100, 100],
            "search_artifact_bytes": 100,
            "fixed_artifact_bytes_samples": [200, 200, 200],
            "fixed_artifact_bytes": 200,
            "projected_cpu_hours": projected_cpu,
            "projected_eight_shard_hours": projected_shard,
            "projected_largest_shard_bytes": projected_bytes,
        },
        "limits": {
            "total_cpu_hours": 120,
            "active_shard_hours": 8.5,
            "retained_shard_bytes": 15 * 2**30,
            "working_free_bytes": 10_000,
        },
        "gpu": {"admitted": False},
    }
    validate_kaggle_environment_receipt(environment, expected_source_identity=source)
    environment["benchmark"]["projected_cpu_hours"] = 0.0
    with pytest.raises(ContractError, match="does not reconstruct"):
        validate_kaggle_environment_receipt(environment, expected_source_identity=source)


def test_writable_directory_is_not_a_read_only_mount(tmp_path: Path) -> None:
    read_only, observation = _read_only_mount(tmp_path)
    assert read_only is False
    assert observation


def test_non_kaggle_host_cannot_issue_scientific_receipt(tmp_path: Path) -> None:
    value = qualify_kaggle_environment(kaggle_root=tmp_path)
    assert value["qualified"] is False
    assert value["scientific_admitted"] is False
    assert value["scope"] == "kaggle_scientific"


def test_local_receipt_cannot_unlock_scientific_cell(tmp_path: Path) -> None:
    environment = tmp_path / "environment.json"
    data = tmp_path / "data.json"
    acquisition = tmp_path / "acquisition.json"
    archive = tmp_path / "archive.zip"
    atomic_json(environment, qualify_environment(benchmark_size=64))
    atomic_json(data, {})
    atomic_json(acquisition, {})
    archive.write_bytes(b"not reached")
    with pytest.raises(ContractError, match="environment"):
        require_execution_receipts(
            environment_receipt=environment,
            data_receipt=data,
            acquisition_receipt=acquisition,
            archive=archive,
            family="size_bm",
        )


def test_generated_fold_retains_all_evidence() -> None:
    panel = generated_panel(months=220, seed=9)
    fold = Fold("X", 199601, 199612, 199701, 199712)
    config = ModelConfig(fourier_width=8, reservoir_width=8, engram_capacity=60)
    result = run_fold(panel, fold, seed=11, config=config)
    assert result.scores.shape == (12, 9)
    assert result.router_weights.shape == (12, 5)
    assert result.targets.shape == (12, 9, 7)
    assert result.holdings.shape == (12, 9, 7)
    assert result.outcomes.shape == (12, 6)
    assert result.validation_scores.shape == (12, 9)
    assert np.allclose(result.router_weights.sum(axis=1), 1)
    for row in range(12):
        for arm in range(9):
            length = int(result.forecast_lengths[row, arm])
            law = Law(
                result.forecast_atoms[row, arm, :length],
                result.forecast_probabilities[row, arm, :length],
            )
            assert energy_score(law, result.outcomes[row]) == pytest.approx(
                result.scores[row, arm], abs=1e-15
            )
            assert variogram_score(law, result.outcomes[row]) == pytest.approx(
                result.variograms[row, arm], abs=1e-15
            )


def test_checkpoint_resume_is_bit_exact() -> None:
    panel = generated_panel(months=220, seed=9)
    fold = Fold("X", 199601, 199612, 199701, 199712)
    config = ModelConfig(fourier_width=8, reservoir_width=8, engram_capacity=60)
    complete = run_fold(panel, fold, seed=11, config=config)
    captured: dict[str, Any] = {}

    class Stop(RuntimeError):
        pass

    def stop(state: dict[str, Any]) -> None:
        captured.update(state)
        raise Stop

    with pytest.raises(Stop):
        run_fold(
            panel,
            fold,
            seed=11,
            config=config,
            checkpoint_callback=stop,
            checkpoint_every=120,
        )
    resumed = run_fold(panel, fold, seed=11, config=config, resume_state=captured)
    np.testing.assert_array_equal(complete.scores, resumed.scores)
    np.testing.assert_array_equal(complete.variograms, resumed.variograms)
    np.testing.assert_array_equal(complete.wealth, resumed.wealth)
    np.testing.assert_array_equal(complete.validation_outcomes, resumed.validation_outcomes)
    np.testing.assert_array_equal(complete.validation_forecast_atoms, resumed.validation_forecast_atoms)
    np.testing.assert_array_equal(complete.forecast_atoms, resumed.forecast_atoms)
    np.testing.assert_array_equal(complete.targets, resumed.targets)
    np.testing.assert_array_equal(complete.holdings, resumed.holdings)


def test_fold_artifact_hash_and_no_overwrite(tmp_path: Path) -> None:
    panel = generated_panel(months=432, seed=4)
    result = run_fold(
        panel,
        FOLDS[0],
        seed=11,
        config=ModelConfig(fourier_width=8, reservoir_width=8, engram_capacity=60),
    )
    output = tmp_path / "fold"
    write_fold(output, result, provenance={"generated": True})
    restored = read_fold(output)
    np.testing.assert_array_equal(restored.scores, result.scores)
    np.testing.assert_array_equal(restored.forecast_atoms, result.forecast_atoms)
    with pytest.raises(FileExistsError):
        write_fold(output, result, provenance={"generated": True})
    (output / "arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_fold(output)


def test_fold_reader_reconstructs_scores_and_policy_paths(tmp_path: Path) -> None:
    panel = generated_panel(months=432, seed=7)
    result = run_fold(
        panel,
        FOLDS[0],
        seed=11,
        config=ModelConfig(fourier_width=8, reservoir_width=8, engram_capacity=60),
    )
    output = tmp_path / "fold"
    write_fold(output, result, provenance={"generated": True})
    with np.load(output / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["scores"] = arrays["scores"].copy()
    arrays["scores"][0, 0] += 0.01
    np.savez_compressed(output / "arrays.npz", **arrays)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"]["arrays.npz"] = sha256_file(output / "arrays.npz")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="energy scores do not reconstruct"):
        read_fold(output)


def test_fold_reader_rejects_calendar_validation_and_control_tampering(tmp_path: Path) -> None:
    panel = generated_panel(months=432, seed=12)
    result = run_fold(
        panel,
        FOLDS[0],
        seed=11,
        config=ModelConfig(fourier_width=8, reservoir_width=8, engram_capacity=60),
    )

    calendar = tmp_path / "calendar"
    write_fold(calendar, result, provenance={"engineering_only": True})
    with np.load(calendar / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["months"] = arrays["months"].copy()
    arrays["months"][0] = 199912
    np.savez_compressed(calendar / "arrays.npz", **arrays)
    manifest = read_json(calendar / "manifest.json")
    manifest["files"]["arrays.npz"] = sha256_file(calendar / "arrays.npz")
    (calendar / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"months are invalid|calendar differs"):
        read_fold(calendar)

    validation = tmp_path / "validation"
    write_fold(validation, result, provenance={"engineering_only": True})
    with np.load(validation / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["validation_scores"] = arrays["validation_scores"].copy()
    arrays["validation_scores"][0, 0] += 0.1
    np.savez_compressed(validation / "arrays.npz", **arrays)
    manifest = read_json(validation / "manifest.json")
    manifest["files"]["arrays.npz"] = sha256_file(validation / "arrays.npz")
    (validation / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="validation scores do not reconstruct"):
        read_fold(validation)

    control = tmp_path / "control"
    write_fold(control, result, provenance={"engineering_only": True})
    summary = read_json(control / "summary.json")
    summary["selected_non_frem"] = next(
        name for name in ("gaussian", "student_t", "var", "causal_mlp") if name != result.selected_non_frem
    )
    (control / "summary.json").write_text(json.dumps(summary))
    manifest = read_json(control / "manifest.json")
    manifest["files"]["summary.json"] = sha256_file(control / "summary.json")
    (control / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="selected control does not reconstruct"):
        read_fold(control)


def test_resume_rejects_foreign_identity() -> None:
    panel = generated_panel(months=220, seed=9)
    fold = Fold("X", 199601, 199612, 199701, 199712)
    config = ModelConfig(fourier_width=8, reservoir_width=8, engram_capacity=60)
    with pytest.raises(ContractError, match="identity"):
        run_fold(
            panel,
            fold,
            seed=11,
            config=config,
            resume_state={"identity": {"family": "wrong", "fold": "X", "seed": 11}},
        )


def test_consumption_is_one_use_and_exact_resume_only(tmp_path: Path) -> None:
    identity: dict[str, int | str] = {"family": "size_bm", "fold": "A", "seed": 11}
    gates: dict[str, object] = {"environment": "abc", "data": "def"}
    first = claim_consumption(tmp_path, phase="fixed", identity=identity, gate_hashes=gates)
    assert first["resumed"] is False
    with pytest.raises(ConsumptionError, match="already consumed"):
        claim_consumption(tmp_path, phase="fixed", identity=identity, gate_hashes=gates)
    resumed = claim_consumption(tmp_path, phase="fixed", identity=identity, gate_hashes=gates, resume=True)
    assert resumed["sha256"] == first["sha256"]
    with pytest.raises(ConsumptionError, match="differs"):
        claim_consumption(
            tmp_path,
            phase="fixed",
            identity=identity,
            gate_hashes={"environment": "changed"},
            resume=True,
        )
