"""Execution-gate and perturbation contracts, now verified at runtime."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from frem.config import validate_config
from frem.data import WindowData
from frem.experiments import _archive_protocol, _remaining, execute_suite, suite_plan
from frem.robustness import DAY_NS, held_out_assets, perturb_inputs, recent_training_fraction


def configuration() -> dict:
    return {
        "schema_version": 1,
        "name": "contract",
        "data": {"path": "data/panel.parquet", "context": 8, "horizon": 2},
        "split": {"protected_start": "2025-01-01T00:00:00Z", "embargo_bars": 0},
        "model": {},
        "training": {"epochs": 1, "max_wall_seconds": 20},
        "evaluation": {},
        "models": ["frem", "no_memory"],
        "seeds": [3],
        "budget": {"max_runs": 2, "max_wall_seconds": 60},
        "output_dir": "outputs/runs",
    }


def suite(variants: list[dict]) -> dict:
    return {"max_variants": 3, "max_total_runs": 6, "max_wall_seconds": 120, "variants": variants}


def panel_windows(values: np.ndarray, names: list[str]) -> WindowData:
    count = values.shape[0]
    return WindowData(
        values,
        np.zeros((count, 2, values.shape[-1])),
        np.linspace(-0.02, 0.02, count),
        np.arange(count, dtype=np.int64) + 1,
        np.arange(count, dtype=np.int64) + 4,
        np.zeros(count, dtype=np.int64),
        names,
        ["old_asset", "new_asset"],
        metadata={"source": "contract_fixture"},
    )


def test_validation_normalizes_paths_without_mutating_input(tmp_path) -> None:
    original = configuration()
    original["data"]["macro_path"] = "data/macro.csv"
    snapshot = copy.deepcopy(original)
    checked = validate_config(original, root=tmp_path)
    assert original == snapshot
    assert checked["data"]["path"] == str(tmp_path / "data" / "panel.parquet")
    assert checked["data"]["macro_path"] == str(tmp_path / "data" / "macro.csv")
    assert checked["output_dir"] == str(tmp_path / "outputs" / "runs")


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "models", ["frem", "frem"]),
        (None, "seeds", [3, 3]),
        (None, "schema_version", 1.0),
        ("budget", "max_runs", 1),
        ("budget", "max_wall_seconds", float("inf")),
        ("training", "max_wall_seconds", float("nan")),
        ("evaluation", "noise_std", float("nan")),
        ("split", "protected_start", "2025-01-01"),
    ],
)
def test_invalid_execution_configuration_is_rejected(section, key, value, tmp_path) -> None:
    cfg = configuration()
    (cfg if section is None else cfg[section])[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg, root=tmp_path)


def test_suite_resolves_late_paths_and_records_separate_budget(tmp_path) -> None:
    base = validate_config(configuration(), root=tmp_path)
    declaration = suite(
        [{"name": "other_panel", "data": {"path": "alternate/panel.parquet"}, "output_dir": "alternate/runs"}]
    )
    plan = suite_plan(base, declaration, root=tmp_path)
    variant = plan["variants"][0]
    assert variant["data"]["path"] == str(tmp_path / "alternate" / "panel.parquet")
    assert variant["output_dir"] == str(tmp_path / "alternate" / "runs")
    assert plan["requested_runs"] == 2
    assert plan["requested_runs_if_primary_and_suite_each_execute_once"] == 4
    assert not plan["executed"]


@pytest.mark.parametrize(
    "variants",
    [
        [{"name": "same"}, {"name": "same", "evaluation": {"noise_std": 0.1}}],
        [{"name": "first"}, {"name": "same_effect", "models": ["no_memory", "frem"]}],
        [{"name": "more_seeds", "seeds": [3, 7]}],
        [{"name": "bad_models", "models": ["frem", "frem"]}],
    ],
)
def test_suite_revalidates_every_merge_and_rejects_duplicates(variants, tmp_path) -> None:
    with pytest.raises(ValueError):
        suite_plan(configuration(), suite(variants), root=tmp_path)


def test_later_invalid_suite_variant_prevents_every_execution(monkeypatch, tmp_path) -> None:
    declaration = suite([{"name": "valid_first"}, {"name": "over_budget", "seeds": [3, 7]}])
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(declaration), encoding="utf-8")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("An experiment began before the whole suite was validated")

    monkeypatch.setattr("frem.experiments.execute_matrix", forbidden)
    with pytest.raises(ValueError, match="max_runs"):
        execute_suite(configuration(), source, root=tmp_path)
    assert calls == []


def test_protocol_archive_preserves_bytes_and_rejects_changed_source(tmp_path) -> None:
    research = tmp_path / "research"
    research.mkdir()
    content = '{\r\n  "note": "π prospective"\r\n}\r\n'.encode()
    source = research / "protocol.json"
    source.write_bytes(content)
    identity = {"code": {"files": {"research/protocol.json": hashlib.sha256(content).hexdigest()}}}
    target = tmp_path / "run"
    _archive_protocol(target, tmp_path, identity)
    assert (target / "protocol.json").read_bytes() == content
    source.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        _archive_protocol(tmp_path / "later_run", tmp_path, identity)


def test_final_budget_check_records_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("frem.experiments.time.perf_counter", lambda: 16.0)
    with pytest.raises(TimeoutError, match="final matrix completion"):
        _remaining(5.0, 10.0, "final matrix completion", directory=tmp_path)
    result = json.loads((tmp_path / "budget_failure.json").read_text(encoding="utf-8"))
    assert result["elapsed_seconds"] == 11.0 and result["status"] == "EXCEEDED"


def test_all_missing_training_channel_has_explicit_finite_corruption_scale() -> None:
    train_values = np.zeros((3, 4, 3), dtype=np.float32)
    train_values[..., 0] = np.nan
    train_values[..., 2] = 1.0
    names = ["new_feature", "observed_feature", "new_feature__missing"]
    train = panel_windows(train_values, names)
    evaluation = panel_windows(np.zeros_like(train_values), names)
    original_x = evaluation.x.copy()
    out = perturb_inputs(evaluation, train, noise_std=0.1, missing_probability=0.5, seed=11)
    report = out.metadata["input_perturbation"]
    assert report["all_missing_training_features"] == ["new_feature"]
    assert report["feature_medians"][0] == 0.0
    assert report["feature_noise_widths"][0] == 1.0
    assert np.isfinite(out.x).all()
    assert out.y is evaluation.y and out.future_x is evaluation.future_x
    np.testing.assert_array_equal(evaluation.x, original_x)
    assert set(np.unique(out.x[..., 2])) <= {0.0, 1.0}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_nonfinite_or_negative_corruption_is_rejected(value) -> None:
    data = panel_windows(np.ones((2, 4, 1)), ["feature"])
    with pytest.raises(ValueError, match="severity"):
        perturb_inputs(data, data, noise_std=value)


def test_ood_split_allows_an_unused_empty_complement() -> None:
    data = panel_windows(np.ones((2, 4, 1)), ["feature"])
    retained, heldout = held_out_assets(data, ["new_asset"])
    assert len(retained) == len(data) and len(heldout) == 0
    retained, heldout = held_out_assets(data, ["old_asset"])
    assert len(retained) == 0 and len(heldout) == len(data)


def test_recent_training_fraction_preserves_complete_utc_dates_with_intraday_rows() -> None:
    data = panel_windows(np.ones((6, 4, 1)), ["feature"])
    data.time = np.asarray(
        [DAY_NS + 1, DAY_NS + 2, 2 * DAY_NS + 1, 3 * DAY_NS + 1, 3 * DAY_NS + 2, 3 * DAY_NS + 3],
        dtype=np.int64,
    )
    data.label_time = data.time + DAY_NS
    selected = recent_training_fraction(data, 0.5)
    # ceil(3 UTC dates * 0.5) keeps two full dates, despite uneven intraday counts.
    np.testing.assert_array_equal(selected.time // DAY_NS, np.asarray([2, 3, 3, 3]))
    np.testing.assert_array_equal(selected.time, data.time[2:])
    np.testing.assert_array_equal(selected.y, data.y[2:])
