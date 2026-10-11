"""Prospective integrity tests, now covered by the retained runtime receipt."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from frem import reporting
from frem.artifacts import file_digest, read_json


def manifest(phase="confirmatory"):
    return {
        "status": "COMPLETE",
        "phase": phase,
        "run_id": "fixture_only",
        "configuration": {
            "models": ["frem", "no_memory"],
            "seeds": [11, 29],
            "evaluation": {},
            "data": {"horizon": 5},
        },
        "identity": {"freeze_sha256": "fixture", "code": {"sha256": "fixture", "files": {}}},
    }


def protocol():
    return {
        "primary_comparison": {
            "candidate": "frem",
            "control": "no_memory",
            "minimum_practical_improvement_nats": 0.01,
        },
        "mechanism_comparisons": [
            "fifo_memory",
            "uniform_memory",
            "no_spectral",
            "no_jepa",
            "no_reservoir",
            "no_controller",
            "no_gate",
        ],
        "statistics": {
            "block_dates": 20,
            "bootstrap_repetitions": 5000,
            "bootstrap_seed": 1901,
            "confidence": 0.95,
            "sensitivity_block_dates": [5, 60],
        },
    }


def settings(record=None, **overrides):
    supplied = {
        "block_length": None,
        "n_resamples": None,
        "confidence": None,
        "threshold": None,
        "bootstrap_seed": None,
    }
    supplied.update(overrides)
    return reporting._settings(record or manifest(), protocol(), **supplied)


def predictions(seed=11, model="frem"):
    day = 86_400_000_000_000
    first = pd.Timestamp("2020-01-01T20:00:00Z").value
    times = np.asarray([first, first, first + 3_600_000_000_000, first + day])
    return pd.DataFrame(
        {
            "time": times,
            "asset_id": [0, 1, 0, 0],
            "label_time": times + 3 * day,
            "entry_time": times + day,
            "exit_time": times + 2 * day,
            "actual": [0.01, -0.01, 0.02, -0.02],
            "prediction": 0.0,
            "scale": 0.1,
            "student_df": [4.0, 5.0, 6.0, 7.0],
            "seed": seed,
            "model": model,
            "source_id": "fixture_only",
        }
    )


@pytest.mark.parametrize(
    "override",
    [
        {"block_length": 30},
        {"n_resamples": 1000},
        {"confidence": 0.9},
        {"threshold": 0.0},
        {"bootstrap_seed": 1902},
    ],
)
def test_confirmatory_settings_cannot_change_frozen_inference(override):
    with pytest.raises(ValueError, match="Confirmatory"):
        settings(**override)


def test_prescribed_sensitivity_block_never_becomes_primary():
    chosen = settings(block_length=60)
    assert chosen["primary_block"] is False
    assert chosen["prescribed"]["block_length"] == 20
    assert chosen["bootstrap_seed"] == 1901


def test_fractional_resample_count_is_not_silently_truncated():
    with pytest.raises(ValueError, match="integer"):
        settings(n_resamples=5000.5)


def test_development_uses_its_saved_configuration_defaults():
    record = manifest("development")
    record["configuration"]["evaluation"] = {"bootstrap_repetitions": 2000, "bootstrap_block_dates": 10}
    chosen = settings(record)
    assert chosen["n_resamples"] == 2000 and chosen["block_length"] == 10
    assert chosen["prescribed"]["n_resamples"] == 5000


@pytest.mark.parametrize("column", ["actual", "label_time", "entry_time", "exit_time"])
def test_pairing_requires_identical_outcomes_and_timing(column):
    left = predictions()
    right = left.copy()
    right.loc[0, column] += 1
    with pytest.raises(ValueError, match=column):
        reporting._matching_evidence(left, right)


def test_pairing_never_inner_joins_away_missing_rows():
    left = predictions()
    with pytest.raises(ValueError, match="Unmatched"):
        reporting._matching_evidence(left, left.iloc[:-1])


def test_prediction_reader_requires_every_configured_seed(tmp_path):
    (tmp_path / "frem" / "seed_11").mkdir(parents=True)
    predictions().to_csv(tmp_path / "frem" / "seed_11" / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete seed family"):
        reporting._predictions(tmp_path, "frem", manifest())


def test_prediction_reader_requires_complete_seed_observation_grid(tmp_path):
    for seed in [11, 29]:
        cell = tmp_path / "frem" / f"seed_{seed}"
        cell.mkdir(parents=True)
        frame = predictions(seed)
        if seed == 29:
            frame = frame.iloc[:-1]
        frame.to_csv(cell / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="Unmatched"):
        reporting._predictions(tmp_path, "frem", manifest())


def test_complete_status_does_not_excuse_missing_matrix_cells(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()), encoding="utf-8")
    (tmp_path / "completed_cells.json").write_text(
        json.dumps([{"model": "frem", "seed": 11}, {"model": "frem", "seed": 29}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="registry"):
        reporting._complete_run(tmp_path)


def test_protocol_archive_must_match_saved_content_hash(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol()), encoding="utf-8")
    record = manifest("development")
    record["identity"]["code"]["files"]["research/protocol.json"] = file_digest(path)
    changed = protocol()
    changed["statistics"]["confidence"] = 0.9
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol differs"):
        reporting._protocol_for_run(tmp_path, record)


def test_family_command_refuses_partial_predeclared_family(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "_complete_run", lambda run: manifest())
    monkeypatch.setattr(reporting, "_protocol_for_run", lambda run, record: (protocol(), "fixture"))
    with pytest.raises(ValueError, match="complete prescribed mechanism family"):
        reporting.compare_family(tmp_path)


def test_report_checks_common_evidence_before_creating_artifacts(tmp_path, monkeypatch):
    record = manifest("development")
    record["configuration"]["seeds"] = [11]
    monkeypatch.setattr(reporting, "_complete_run", lambda run: record)
    monkeypatch.setattr(reporting, "_protocol_for_run", lambda run, record: (protocol(), "fixture"))

    def read_predictions(run, model, record):
        result = predictions(model=model).assign(nll=1.0)
        if model == "no_memory":
            result.loc[0, "actual"] += 1.0
        return result

    monkeypatch.setattr(reporting, "_predictions", read_predictions)
    destination = tmp_path / "never_created"
    with pytest.raises(ValueError, match="actual"):
        reporting.render_report(tmp_path, destination)
    assert not destination.exists()


def test_different_analysis_settings_preserve_distinct_artifacts(tmp_path):
    first = {"settings": {"block_length": 20}, "source_prediction_sha256": {"fixture": "a"}}
    second = copy.deepcopy(first)
    second["settings"]["block_length"] = 60
    one = reporting._preserve_analysis(tmp_path, "fixture_only", first)
    two = reporting._preserve_analysis(tmp_path, "fixture_only", second)
    assert one != two
    assert read_json(one) == first and read_json(two) == second
