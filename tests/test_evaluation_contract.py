"""Prospective evaluation contracts, now covered by runtime verification."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from frem.evaluation import (
    build_execution_ledger,
    evaluate_portfolio,
    evaluate_predictions,
    score_predictions,
)
from frem.metrics import apply_execution_costs, student_t_nll
from frem.statistics import average_seeds, date_block_bootstrap, holm_adjust, paired_date_test


def predictions():
    day = 86_400_000_000_000
    start = pd.Timestamp("2020-01-01T20:00:00Z").value
    times = np.asarray([start, start + 3 * day], dtype=np.int64)
    return pd.DataFrame(
        {
            "time": times,
            "label_time": times + 2 * day + 1,
            "entry_time": times + day,
            "exit_time": times + 2 * day,
            "asset_id": 0,
            "prediction": [0.01, 0.02],
            "scale": 0.1,
            "student_df": [4.0, 9.0],
            "actual": np.log([1.1, 0.9]),
        }
    )


def test_student_t_nll_matches_configured_distribution_not_gaussian():
    actual = np.asarray([0.01, -0.1])
    location = np.asarray([0.0, 0.02])
    scale = np.asarray([0.03, 0.1])
    degrees = np.asarray([3.0, 11.0])
    np.testing.assert_allclose(
        student_t_nll(actual, location, scale, degrees),
        -t.logpdf(actual, df=degrees, loc=location, scale=scale),
    )


def test_saved_per_row_df_overrides_scalar_fallback():
    frame = predictions()
    scores = score_predictions(frame, student_df=5.0)
    expected = -t.logpdf(frame.actual, df=frame.student_df, loc=frame.prediction, scale=frame.scale)
    np.testing.assert_allclose(scores.nll, expected)


def test_unlabeled_predictions_are_rejected_by_evaluator():
    frame = predictions()
    frame["actual"] = np.nan
    with pytest.raises(ValueError, match="Actual outcomes"):
        evaluate_predictions(frame)


def test_missing_asset_identifier_cannot_disappear_from_grouped_scores():
    frame = predictions()
    frame.loc[0, "asset_id"] = np.nan
    with pytest.raises(ValueError, match="asset_id"):
        score_predictions(frame)


def test_undefined_cross_section_rank_is_reported_not_zero_filled():
    report = evaluate_predictions(predictions())
    rank = report["rank_correlation"]
    assert rank["mean_over_defined_dates"] is None
    assert rank["defined_dates"] == 0 and rank["undefined_dates"] == 2


def test_partial_cash_costs_solve_the_post_fee_weight_budget():
    frame = predictions().iloc[:1]
    weights = frame[["time", "asset_id"]].assign(weight=0.5)
    cash = pd.Series([0.02], index=frame.time)
    ledger = build_execution_ledger(
        frame, weights, entry_cost_bps=100.0, exit_cost_bps=200.0, cash_returns=cash
    )
    alpha = 1.0 / 1.005
    expected = alpha * (0.5 * 1.1 + 0.5 * 1.02) - 0.02 * alpha * 0.5 * 1.1 - 1
    assert ledger.net_return.iloc[0] == pytest.approx(expected)
    assert ledger.entry_cost_fraction.iloc[0] == pytest.approx(0.01 * alpha * 0.5)


def test_no_backtest_from_overlapping_labels():
    frame = predictions()
    day = 86_400_000_000_000
    frame.loc[1, ["time", "entry_time", "exit_time", "label_time"]] -= 2 * day + day // 2
    weights = frame[["time", "asset_id"]].assign(weight=1.0)
    with pytest.raises(ValueError, match="Overlapping"):
        build_execution_ledger(frame, weights)


def test_entry_and_exit_costs_have_explicit_compounding():
    net = apply_execution_costs(np.asarray([0.1]), 0.001, 0.002)
    assert net[0] == pytest.approx(0.999 * 1.1 * 0.998 - 1)
    frame = predictions()
    weights = frame[["time", "asset_id"]].assign(weight=1.0)
    ledger = build_execution_ledger(frame, weights, entry_cost_bps=10.0, exit_cost_bps=20.0)
    alpha = 1.0 / 1.001
    assert ledger.net_return.iloc[0] == pytest.approx(alpha * 1.1 * 0.998 - 1)
    assert ledger.entry_cost_fraction.iloc[0] == pytest.approx(0.001 * alpha)
    assert ledger.turnover.iloc[0] == pytest.approx(alpha + alpha * 1.1)
    report = evaluate_portfolio(ledger)
    assert report["annualized"] is False
    with pytest.raises(ValueError, match="regular_periods"):
        evaluate_portfolio(ledger, periods_per_year=252.0)


def test_bootstrap_groups_opposite_asset_errors_on_same_date():
    dates = np.repeat(pd.date_range("2020-01-01T00:00:00Z", periods=20).asi8, 3)
    dates += np.tile([0, 3_600_000_000_000, 0], 20)
    frame = pd.DataFrame(
        {"time": dates, "asset_id": np.tile([0, 0, 1], 20), "loss_difference": np.tile([1.0, 1.0, -1.0], 20)}
    )
    result = date_block_bootstrap(frame, "loss_difference", block_length=2, n_resamples=199)
    assert result.estimate == 0.0
    assert result.ci_lower == 0.0 and result.ci_upper == 0.0
    assert result.p_two_sided == 1.0


def test_paired_test_never_drops_unmatched_assets():
    frame = predictions().assign(nll=[1.0, 2.0])
    with pytest.raises(ValueError, match="samples differ"):
        paired_date_test(frame, frame.iloc[:1], "nll", block_length=1)


def test_inference_requires_overlap_aware_block_size():
    frame = pd.DataFrame(
        {"time": pd.date_range("2020-01-01T00:00:00Z", periods=20).asi8, "value": np.arange(20.0)}
    )
    with pytest.raises(ValueError, match="block length"):
        date_block_bootstrap(frame, "value", block_length=2, minimum_block_length=5)


def test_seed_aggregation_requires_complete_observation_family():
    frame = pd.DataFrame(
        {"time": [1, 1, 2], "asset_id": [0, 0, 0], "seed": [0, 1, 0], "nll": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(ValueError, match="every seed"):
        average_seeds(frame, ["nll"])


def test_holm_family_adjustment():
    result = holm_adjust({"one": 0.01, "two": 0.02, "three": 0.20})
    assert result == {"one": 0.03, "two": 0.04, "three": 0.20}
