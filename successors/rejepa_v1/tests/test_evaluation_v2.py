import numpy as np
import pytest

from rejepa.baselines import (
    CausalContinuationRetriever,
    CausalStreamingResidualMemory,
    ResidualAlignment,
    equal_weight,
    fit_linear_baseline,
    inverse_volatility_weight,
)
from rejepa.evaluation import (
    PurgedWalkForwardSplitter,
    distribution_calibration,
    paired_moving_block_interval,
    portfolio_metrics,
)


def test_walk_forward_has_embargoed_nonoverlapping_dates() -> None:
    dates = np.arange("2025-01-01", "2025-02-20", dtype="datetime64[D]")
    folds = PurgedWalkForwardSplitter(20, 5, 5, 2).split(dates)
    assert folds
    first = folds[0]
    assert first.train_dates[-1] < first.validation_dates[0] < first.test_dates[0]


def test_metrics_include_costs_drawdown_and_cvar() -> None:
    result = portfolio_metrics(
        np.array([0.01, -0.02, 0.015, -0.01]),
        turnover=np.array([0.1, 0.2, 0.1, 0.0]),
        cost_rate=0.001,
        periods_per_year=4,
    )
    assert result["transaction_cost"] == pytest.approx(0.0004)
    assert result["maximum_drawdown"] <= 0
    assert result["cvar_loss"] >= 0


def test_calibration_and_paired_block_interval() -> None:
    calibration = distribution_calibration(
        np.array([0.0, 1.0, 2.0]), np.array([-1.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]), 0.9
    )
    assert calibration["empirical_coverage"] == 1.0
    candidate = np.linspace(0.8, 1.0, 40)
    control = candidate + 0.02
    estimate, lower, upper = paired_moving_block_interval(
        candidate, control, block_length=5, repetitions=100, seed=7
    )
    assert estimate == pytest.approx(0.02)
    assert lower <= estimate <= upper


def test_simple_baselines_are_executable() -> None:
    x = np.arange(20.0).reshape(10, 2)
    y = x[:, 0] * 0.1
    model = fit_linear_baseline(x, y)
    assert model.predict(x).shape == y.shape
    assert equal_weight(4).sum() == pytest.approx(1.0)
    assert inverse_volatility_weight(np.diag([1.0, 4.0])).sum() == pytest.approx(1.0)


def test_causal_continuation_retrieval_excludes_unmatured_records() -> None:
    control = CausalContinuationRetriever(
        np.eye(3),
        np.array([0.1, 0.2, 9.0]),
        np.array([5, 10, 50]),
        identifiers=("early-a", "early-b", "future"),
        neighbors=2,
    )
    result = control.predict(np.array([[0.0, 0.0, 1.0]]), np.array([20], dtype=np.int64))
    assert "future" not in result.retrieved_identifiers[0]
    assert result.values[0] < 1.0


def test_continuation_retrieval_returns_empty_when_no_outcome_has_matured() -> None:
    control = CausalContinuationRetriever(np.eye(2), np.array([0.1, 0.2]), np.array([10, 20]))
    result = control.predict(np.array([[1.0, 0.0]]), np.array([5], dtype=np.int64))
    assert result.values[0] == 0.0
    assert result.retrieved_identifiers == ((),)


def test_streaming_residual_control_is_maturity_gated_and_alignment_is_frozen() -> None:
    memory = CausalStreamingResidualMemory(neighbors=1)
    with pytest.raises(ValueError, match="before it is available"):
        memory.add(
            "premature",
            np.array([1.0, 0.0]),
            base_prediction=0.0,
            outcome=1.0,
            outcome_available_ns=10,
            recorded_at_ns=9,
        )
    memory.add(
        "mature",
        np.array([1.0, 0.0]),
        base_prediction=0.2,
        outcome=0.5,
        outcome_available_ns=10,
        recorded_at_ns=10,
    )
    with pytest.raises(ValueError, match="recorded-time order"):
        memory.add(
            "backfilled",
            np.array([1.0, 0.0]),
            base_prediction=0.0,
            outcome=0.0,
            outcome_available_ns=8,
            recorded_at_ns=9,
        )
    with pytest.raises(ValueError, match="same width"):
        memory.add(
            "wrong-width",
            np.array([1.0, 0.0, 0.0]),
            base_prediction=0.0,
            outcome=0.0,
            outcome_available_ns=11,
            recorded_at_ns=11,
        )
    before = memory.raw_correction(np.array([[1.0, 0.0]]), np.array([9], dtype=np.int64))
    after = memory.raw_correction(np.array([[1.0, 0.0]]), np.array([10], dtype=np.int64))
    assert before.values[0] == 0.0 and before.retrieved_identifiers == ((),)
    assert after.values[0] == pytest.approx(0.3)
    alignment = ResidualAlignment.fit(
        np.array([0.0, 0.0]), np.array([0.5, -0.5]), np.array([1.0, -1.0]), ridge=0.0
    )
    np.testing.assert_allclose(
        alignment.apply(np.array([0.1, 0.2]), np.array([0.5, -0.5])),
        np.array([1.1, -0.8]),
    )
