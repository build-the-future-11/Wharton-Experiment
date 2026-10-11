import numpy as np

from reng.evaluation import (
    BootstrapResult,
    classify_primary,
    energy_score,
    holm_adjust,
    normalized_quantile_scores,
    paired_stationary_bootstrap,
    representation_diagnostics,
)


def test_primary_score_uses_dates_not_asset_rows() -> None:
    outcomes = np.array([[0.0, 0.0], [1.0, 1.0]])
    scales = np.ones_like(outcomes)
    predictions = np.zeros(outcomes.shape + (7,))
    dates, score = normalized_quantile_scores(outcomes, predictions, scales, np.array(["d1", "d2"]))
    assert list(dates) == ["d1", "d2"]
    assert len(score) == 2


def test_stationary_bootstrap_preserves_declared_folds() -> None:
    candidate = np.linspace(0.8, 1.0, 30)
    control = np.ones(30)
    folds = np.repeat([2018, 2019, 2020], 10)
    result = paired_stationary_bootstrap(
        candidate, control, folds, repetitions=200, expected_block=5
    )
    assert result.point > 0
    assert len(result.draws) == 200


def test_primary_decision_rules() -> None:
    assert classify_primary(BootstrapResult(0.03, 0.01, 0.05, np.array([]), 63)) == "SUPPORTED"
    assert classify_primary(BootstrapResult(0.00, -0.01, 0.01, np.array([]), 63)) == "NEGATIVE"
    assert classify_primary(BootstrapResult(0.03, -0.01, 0.05, np.array([]), 63)) == "INCONCLUSIVE"
    assert (
        classify_primary(BootstrapResult(0.03, 0.01, 0.05, np.array([]), 63), valid=False)
        == "INVALID"
    )


def test_holm_is_step_down() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted["a"]["reject"] is True
    assert adjusted["c"]["reject"] is False


def test_collapse_diagnostic_and_energy_score() -> None:
    collapsed = representation_diagnostics(np.ones((30, 32)))
    assert collapsed["collapsed"] is True
    scenarios = np.stack([np.zeros(3), np.ones(3), np.full(3, 0.5)])
    assert np.isfinite(energy_score(scenarios, np.full(3, 0.4)))
