"""Regression tests for causal APEN evaluation; no external data or credentials."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wharton_lab.backtesting.walk_forward import iter_walk_forward
from wharton_lab.data.causal_tracks import CausalPanel, causal_return_features
from wharton_lab.evaluation.m04_prequential import (
    ARMS, MaturedAPEN, evaluate_fold, run_suite, verify_sources,
)
from wharton_lab.models.m04.config import M04Config


def fixture_panel(n=180):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    y = rng.normal(size=n)
    return CausalPanel(X, y, np.arange(n), "unit_test_only")


def learner(**kwargs):
    p = fixture_panel()
    return MaturedAPEN(p.X[:80], p.y[:80], p.row_t[:80], first_test=100, **kwargs)


def run(p):
    return evaluate_fold(p, np.arange(100), np.arange(100, len(p.y)))


def test_original_source_hashes():
    assert len(verify_sources()) == 8


def test_features_do_not_depend_on_current_or_future_returns():
    rng = np.random.default_rng(23)
    r = rng.normal(size=(300, 5))
    a = causal_return_features(r)
    r[150:] += 1000
    b = causal_return_features(r)
    np.testing.assert_array_equal(a.X[a.row_t <= 150], b.X[b.row_t <= 150])


def test_training_label_maturity_is_enforced():
    p = fixture_panel()
    with pytest.raises(ValueError, match="not mature"):
        MaturedAPEN(p.X[:100], p.y[:100], p.row_t[:100], first_test=100)


def test_fold_purges_exactly_two_immature_training_rows():
    rows, m = run(fixture_panel())
    assert m["n_purged"] == 2
    assert m["train_last_row"] + 3 == m["test_first_row"]
    assert [r["available_test_labels"] for r in rows[:5]] == [0, 0, 0, 1, 2]


def test_no_target_enters_pending_queue_at_prediction_time():
    m = learner()
    m.predict(np.ones(4), 100)
    assert m.model._pending.pending_count() == 0
    assert not m.model._memory.episodes
    assert len(m.issued[100]) == 2  # Feature vector and backbone prediction only.


def test_early_outcome_rejected():
    m = learner(); m.predict(np.ones(4), 100)
    with pytest.raises(ValueError, match="not mature"):
        m.observe(100, 7.0, now_row=102)
    assert not m.model._memory.episodes
    assert m.observed_count == 0


def test_mature_outcome_populates_memory():
    m = learner(); m.predict(np.ones(4), 100)
    m.observe(100, 7.0, now_row=103)
    assert m.observed_count == 1
    assert len(m.model._memory.episodes) == 1
    assert m.model._pending.pending_count() == 0


def test_duplicate_outcome_rejected():
    m = learner(); m.predict(np.ones(4), 100); m.observe(100, 7.0, now_row=103)
    with pytest.raises(ValueError, match="already observed"):
        m.observe(100, 7.0, now_row=104)


def test_outcome_without_prediction_rejected():
    with pytest.raises(ValueError, match="No outstanding"):
        learner().observe(100, 7.0, now_row=104)


def test_duplicate_prediction_rejected():
    m = learner(); m.predict(np.ones(4), 100)
    with pytest.raises(ValueError, match="strictly increasing"):
        m.predict(np.ones(4), 100)


def test_retroactive_prediction_after_later_observation_rejected():
    m = learner(); m.predict(np.ones(4), 100); m.observe(100, 7.0, now_row=105)
    with pytest.raises(ValueError, match="backwards"):
        m.predict(np.ones(4), 104)


def test_future_label_cannot_change_any_prediction_before_maturity():
    p = fixture_panel()
    a, _ = run(p)
    q = CausalPanel(p.X.copy(), p.y.copy(), p.row_t.copy(), "mutated_labels")
    q.y[140:] += 1000
    b, _ = run(q)
    for arm in ARMS:
        np.testing.assert_array_equal([r[arm] for r in a if r["row_t"] < 143],
                                      [r[arm] for r in b if r["row_t"] < 143])


def test_future_feature_cannot_change_earlier_predictions():
    p = fixture_panel(); a, _ = run(p)
    q = CausalPanel(p.X.copy(), p.y.copy(), p.row_t.copy(), "mutated_features")
    q.X[140:] += 1000
    b, _ = run(q)
    for arm in ARMS:
        np.testing.assert_array_equal([r[arm] for r in a if r["row_t"] < 140],
                                      [r[arm] for r in b if r["row_t"] < 140])


def test_scaler_uses_only_mature_training_rows():
    p = fixture_panel()
    m = MaturedAPEN(p.X[:80], p.y[:80], p.row_t[:80], first_test=100)
    np.testing.assert_array_equal(m.mu, p.X[:80].mean(axis=0))
    np.testing.assert_array_equal(m.sd, p.X[:80].std(axis=0))


def test_zero_residual_ablation_matches_frozen_backbone_exactly():
    rows, _ = run(fixture_panel())
    np.testing.assert_array_equal([r["M04_zero_residual"] for r in rows],
                                  [r["ridge_no_memory"] for r in rows])


def test_mechanism_positive_control_is_active_after_maturity():
    p = fixture_panel()
    p.y[:100] = 0; p.y[100:] = 2; p.X[100:] = 1
    rows, m = run(p)
    np.testing.assert_allclose([r["M04_online"] for r in rows[:3]], 0)
    np.testing.assert_allclose([r["M04_online"] for r in rows[3:]], 2, atol=1e-6)
    assert m["correction_count"] == len(rows)-3


def test_folds_reset_memory_and_predictions_are_deterministic():
    p = fixture_panel()
    a, am = run(p); b, bm = run(p)
    assert a == b and am == bm
    assert a[0]["memory_size"] == 0


def test_test_windows_are_disjoint():
    folds = list(iter_walk_forward(1180, n_folds=8, min_train=300, test_size=100))
    assert len(folds) == 8
    idx = np.concatenate([f.test_idx for f in folds])
    assert len(idx) == len(np.unique(idx)) == 800


def test_online_mean_uses_same_matured_labels_as_APEN():
    p = fixture_panel(); rows, _ = run(p)
    for r in rows:
        eligible = p.y[100:r["row_t"]-2] if r["row_t"] >= 103 else np.array([])
        expected = np.concatenate([p.y[:98], eligible]).mean()
        assert r["online_mean"] == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_delay_rejected(value):
    with pytest.raises(ValueError):
        learner(config=M04Config(horizon_steps=1, maturity_delay=value))


@pytest.mark.parametrize("change", ["nan", "unordered", "noninteger", "mismatched"])
def test_invalid_training_inputs_rejected(change):
    p = fixture_panel()
    X, y, t = p.X[:80].copy(), p.y[:80].copy(), p.row_t[:80].copy()
    if change == "nan": X[0, 0] = np.nan
    if change == "unordered": t[[0, 1]] = t[[1, 0]]
    if change == "noninteger": t = t.astype(float)
    if change == "mismatched": y = y[:-1]
    with pytest.raises(ValueError): MaturedAPEN(X, y, t, first_test=100)


def test_unmatched_backbone_rejected():
    with pytest.raises(ValueError, match="ridge only"):
        learner(config=M04Config(horizon_steps=1, use_mlp_backbone=True))


def small_protocol():
    return {"confirmatory": False, "data_scope": "synthetic_only", "arms": list(ARMS),
            "seeds": [11], "worlds": ["null"], "n_steps": 120, "n_assets": 5,
            "n_folds": 2, "min_train": 40, "test_size": 10, "horizon_steps": 1,
            "maturity_delay": 2, "bias_gain": 0.1}


def test_suite_writes_replayable_artifacts_and_refuses_overwrite(tmp_path):
    out = tmp_path/"new_evidence"
    result = run_suite(out, small_protocol())
    assert result["dataset_seed_cases"] == 1 and result["fold_evaluations"] == 2
    assert result["target_rows"] == 20
    assert len(list(out.glob("*_predictions.csv"))) == 1
    manifest = json.loads((out/"RUN_MANIFEST.json").read_text())
    assert manifest["status"] == "COMPLETED" and len(manifest["sources"]) == 8
    assert manifest["real_market_validation"] == "NOT_RUN"
    assert (out/"PROTOCOL.json").stat().st_mtime <= (out/"RESULTS.json").stat().st_mtime
    with pytest.raises(FileExistsError): run_suite(out, small_protocol())


@pytest.mark.parametrize("folder", ["lockbox", "repaired", "factorial", "protocol"])
def test_protected_legacy_paths_rejected(tmp_path, folder):
    with pytest.raises(ValueError, match="protected"):
        run_suite(tmp_path/folder/"new", small_protocol())


def test_unapproved_seed_rejected(tmp_path):
    cfg = small_protocol(); cfg["seeds"] = [999]
    with pytest.raises(ValueError, match="development seeds"):
        run_suite(tmp_path/"badseed", cfg)


def test_confirmatory_claim_rejected(tmp_path):
    cfg = small_protocol(); cfg["confirmatory"] = True
    with pytest.raises(ValueError, match="exploratory"):
        run_suite(tmp_path/"badclaim", cfg)


@pytest.mark.parametrize("horizon", [5, True])
def test_non_one_step_horizon_rejected(horizon):
    with pytest.raises(ValueError):
        learner(config=M04Config(horizon_steps=horizon))


@pytest.mark.parametrize("horizon", [5, True])
def test_protocol_horizon_cannot_mislabel_results(tmp_path, horizon):
    cfg = small_protocol(); cfg["horizon_steps"] = horizon
    with pytest.raises(ValueError):
        run_suite(tmp_path/"badhorizon", cfg)
