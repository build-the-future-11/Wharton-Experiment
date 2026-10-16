"""Generated fixed-stream, denominator and precision reference evidence."""

import math
from dataclasses import replace

import numpy as np
import pytest
from test_endpoints_audit import fold

from frem_s10.analysis.bootstrap import (
    assess_precision,
    campaign_indices,
    pilot_indices,
    relative_gain,
    resampled_gains,
)
from frem_s10.analysis.endpoints import assess_study
from frem_s10.errors import NumericalError
from frem_s10.math.scores import calibration_bins, discrete_quantiles, loss_probabilities, variogram_score


def test_every_fixed_calibration_edge_sparse_boundary_and_exact_loss_events() -> None:
    """TEST-031: every declared bin edge, empty/missing and exact event conventions."""
    probabilities = np.arange(11, dtype=np.float64) / 10
    bins = calibration_bins(probabilities, np.zeros(11))
    assert [row["count"] for row in bins] == [1] * 9 + [2]
    dense = calibration_bins(np.zeros(10), np.zeros(10))
    assert dense[0]["sparse"] is False and dense[0]["frequency"] == 0
    assert dense[1]["sparse"] is True and dense[1]["frequency"] is None
    # Asset marginal events differ from the joint equal-weight basket event.
    atoms = np.array([[-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    np.testing.assert_array_equal(loss_probabilities(atoms, [0.5, 0.5]), [0.5, 0.5, 0, 0])
    np.testing.assert_array_equal(
        discrete_quantiles([1, 1, 3], [0.25, 0.25, 0.5], [0.5, np.nextafter(0.5, 1)]), [1, 3]
    )
    assert variogram_score(np.array([[1.0, 1, 1]]), [1], np.ones(3)) == 0
    # The three pair discrepancies are sqrt(2),1,1; their squared mean is4/3.
    assert variogram_score(np.zeros((1, 3)), [1], [-1, 1, 0]) == pytest.approx(4 / 3)


def test_all_5000_rng_draws_are_independent_of_operation_order() -> None:
    """TEST-066: local SeedSequences cannot consume another operation's stream."""
    first_pilot = pilot_indices(24)
    first_campaign = campaign_indices(24, 25)
    second_campaign = campaign_indices(24, 25)
    second_pilot = pilot_indices(24)
    for key, array in first_pilot.items():
        assert array.shape == (5000, 24)
        np.testing.assert_array_equal(array, second_pilot[key])
    for key, array in first_campaign.items():
        np.testing.assert_array_equal(array, second_campaign[key])


def test_relative_gain_exact_denominator_threshold_observed_and_resampled() -> None:
    """TEST-065: equality at the declared floor is undefined, never a zero gain."""
    indices = np.tile(np.array([0, 1], dtype=np.int64), (5000, 1))
    for value in (np.nextafter(1e-12, 0), 1e-12):
        control = np.full(2, value)
        with pytest.raises(NumericalError):
            relative_gain(control / 2, control)
        with pytest.raises(NumericalError):
            resampled_gains(control / 2, control, indices)
    control = np.full(2, np.nextafter(1e-12, math.inf))
    assert relative_gain(control / 2, control) == 0.5
    np.testing.assert_array_equal(resampled_gains(control / 2, control, indices), np.full(5000, 0.5))


def test_precision_sample_standard_deviation_matches_scalar_reference() -> None:
    """TEST-070: scalar ratios and ddof1 establish the fixed projection arithmetic."""
    x = np.arange(24, dtype=np.float64)
    candidate = 1 + 0.02 * np.sin(x)
    equal = 1.1 + 0.03 * np.cos(x)
    global_stack = 1.05 + 0.01 * np.sin(x + 1)
    result = assess_precision(candidate, equal, global_stack, 24, 25)
    indices = pilot_indices(24)
    for name, block, sd, projected, passed in result.comparisons:
        control = equal if name == "equal" else global_stack
        ratios = [
            1 - math.fsum(float(candidate[i]) for i in row) / math.fsum(float(control[i]) for i in row)
            for row in indices[block]
        ]
        mean = math.fsum(ratios) / 5000
        reference = math.sqrt(math.fsum((x - mean) ** 2 for x in ratios) / 4999)
        assert sd == pytest.approx(reference, rel=1e-13)
        assert projected == pytest.approx(reference * math.sqrt(24 / 49), rel=1e-13)
        assert passed == (2.8 * projected <= 0.05)


def test_valid_negative_disposition_keeps_both_failed_hypotheses() -> None:
    """TEST-071: a valid bad result remains NEGATIVE; no score or control substitution."""
    folds = []
    for name in ("A", "B"):
        original = fold(name)
        energy = original.energy.copy()
        energy[:, 0] = 1.2
        wealth = original.terminal_wealth.copy()
        wealth[0] = 90000
        increments = np.repeat((np.log(wealth / 100000) / 24)[None, :], 24, axis=0)
        folds.append(replace(original, energy=energy, terminal_wealth=wealth, log_increments=increments))
    result = assess_study((folds[0], folds[1]), integrity_verified=True, complete_matrix=True)
    assert result.disposition == "NEGATIVE"
    assert result.h1 is not None and result.h2 is not None
    assert result.h1.status == result.h2.status == "FAIL"
