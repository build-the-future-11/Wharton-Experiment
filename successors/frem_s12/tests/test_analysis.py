from __future__ import annotations

import numpy as np

from frem_s12.analysis import _aggregate, _hierarchical_draws, _seed_averaged


def test_algorithmic_seeds_are_averaged_before_inference() -> None:
    grouped = {
        "size_bm": {
            "A": [np.asarray([1.0, 3.0]), np.asarray([3.0, 5.0])],
        }
    }
    averaged = _seed_averaged(grouped)
    np.testing.assert_array_equal(averaged["size_bm"]["A"], np.asarray([2.0, 4.0]))


def test_aggregation_weights_families_then_folds_equally() -> None:
    series = {
        "first": {"A": np.asarray([1.0]), "B": np.asarray([3.0])},
        "second": {"A": np.asarray([100.0])},
    }
    assert _aggregate(series) == 51.0


def test_hierarchical_bootstrap_preserves_constant_family_means() -> None:
    series = {
        "first": {"A": np.ones(24), "B": np.ones(24)},
        "second": {"A": np.full(24, 3.0), "B": np.full(24, 3.0)},
    }
    draws = _hierarchical_draws(series, block=12, draws=1_000, seed=11)
    np.testing.assert_array_equal(draws, np.full(1_000, 2.0))


def test_hierarchical_bootstrap_preserves_cross_family_calendar_dependence() -> None:
    series = {
        "first": {"A": np.ones(24), "B": -np.ones(24)},
        "second": {"A": -np.ones(24), "B": np.ones(24)},
    }
    draws = _hierarchical_draws(series, block=12, draws=1_000, seed=29)
    np.testing.assert_array_equal(draws, np.zeros(1_000))
