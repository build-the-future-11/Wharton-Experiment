"""A successful prospective controller must report a finite defined objective."""
from fractions import Fraction
import warnings

import numpy as np
import pytest

from wharton_lab.models.m11.mpc import PortfolioState
from wharton_lab.models.m11.mpc_v2 import empirical_cvar_v2, solve_mpc_v2


def cash_only(reward=0.0, cash=4e307, scenarios=5):
    return solve_mpc_v2(
        state=PortfolioState(np.zeros(1), cash),
        scenario_returns=np.zeros((scenarios, 1, 1)),
        obligations=np.zeros(1), min_cash_buffer=1,
        tcost_bps=0, terminal_wealth_weight=reward,
    )


def test_disabled_terminal_reward_does_not_create_nan_from_unused_mean():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = cash_only()
    assert result.success
    assert result.objective == 0
    np.testing.assert_array_equal(result.ledger.wealth, np.full((5, 1), 4e307))


@pytest.mark.parametrize("reward", [0.5, 1.0, 2.0])
def test_representable_terminal_reward_survives_overflowing_naive_mean(reward):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = cash_only(reward)
    expected = -float(Fraction(4e307) * Fraction(reward))
    assert result.success
    assert result.objective == expected


def test_unrepresentable_terminal_reward_is_not_returned_as_success():
    with pytest.raises(ArithmeticError, match="objective"):
        cash_only(reward=10)


def test_nonzero_terminal_reward_product_underflow_is_not_reported_as_zero():
    with pytest.raises(ArithmeticError, match="objective product underflow"):
        cash_only(reward=1e-200, cash=1e-200)


def test_representable_small_terminal_reward_remains_supported():
    result = cash_only(reward=1e-100, cash=1e-100)
    assert result.objective == -1e-200
    assert result.success


@pytest.mark.parametrize("reward", [0, 0.5, 2])
def test_ordinary_and_zero_wealth_cash_objectives_keep_the_declared_law(reward):
    for cash in (0, 1, 100):
        result = cash_only(reward, cash)
        assert result.objective == -reward * cash
        assert result.success


@pytest.mark.parametrize("small", [1.0, float.fromhex("0x0.0000000000003p-1022")])
def test_cvar_retains_residual_after_large_opposite_losses(small):
    values = [1e308, small, -1e308]
    expected = float(sum((Fraction(v) for v in values), Fraction()) / 3)
    assert expected > 0
    assert empirical_cvar_v2(values, alpha=0) == expected


@pytest.mark.parametrize("alpha", [0, 0.125, 0.625, 0.875])
def test_cvar_matches_exact_fractional_tail_mass(alpha):
    values = [1e308, 1e-200, -2.0, -1e308]
    mass = Fraction((1 - alpha) * len(values))
    whole = int(mass)
    boundary = mass - whole
    ordered = sorted(map(Fraction, values), reverse=True)
    total = sum(ordered[:whole], Fraction())
    if boundary:
        total += boundary * ordered[whole]
    expected = float(total / mass)
    assert empirical_cvar_v2(values, alpha) == expected


def test_identical_large_losses_have_a_finite_cvar():
    assert empirical_cvar_v2([np.finfo(float).max] * 5, 0) == np.finfo(float).max


def test_numpy_float32_alpha_retains_its_existing_tail_mass():
    alpha = np.float32(0.625)
    assert empirical_cvar_v2([0, 1, 2, 3], alpha) == float(Fraction(8, 3))


def test_unrepresentable_nonzero_cvar_is_not_reported_as_zero():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    with pytest.raises(ArithmeticError, match="underflow"):
        empirical_cvar_v2([tiny, 0], 0)
    assert empirical_cvar_v2([tiny, -tiny], 0) == 0
    assert empirical_cvar_v2([tiny, 0], 0.5) == tiny


def test_actual_bounded_optimizer_still_improves_constructed_funding_gap():
    result = solve_mpc_v2(
        state=PortfolioState(np.zeros(2), 1),
        scenario_returns=np.ones((3, 1, 2)), obligations=np.array([1.5]),
        max_weight=0.5, tcost_bps=0, maxiter=20,
    )
    assert result.success
    assert np.isfinite(result.objective)
    assert result.objective == pytest.approx(0, abs=1e-8)
    np.testing.assert_allclose(result.ledger.payments, [[1.5], [1.5], [1.5]])
