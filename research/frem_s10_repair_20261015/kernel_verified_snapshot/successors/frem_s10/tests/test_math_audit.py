"""Constructed engineering checks; no market data or scientific evidence."""

import math
from collections.abc import Callable, Sequence
from decimal import Decimal, localcontext
from itertools import product
from typing import cast

import numpy as np
import pytest
from frem_s10.arrays import FloatArray
from frem_s10.errors import ContractError, NumericalError
from frem_s10.math.distances import extrapolation_threshold
from frem_s10.math.fees import keep_holdings_solution, solve_rebalance, solve_rebalances
from frem_s10.math.probabilities import recency_weights, validate_probabilities
from frem_s10.math.scores import (
    calibration_bins,
    discrete_quantiles,
    energy_score,
    mean_squared_error,
    score_coefficients,
    variogram_score,
)
from frem_s10.math.simplex import SolverTolerances, solve_simplex
from frem_s10.math.utility import crra_utility, robust_objective
from numpy.typing import ArrayLike, NDArray


def scalar_es(
    atoms: Sequence[Sequence[float]], probabilities: Sequence[float], outcome: Sequence[float]
) -> float:
    """Independent Python scalar norm and double-sum reference."""
    attraction = math.fsum(p * math.dist(z, outcome) for p, z in zip(probabilities, atoms, strict=True))
    spread = math.fsum(
        p * q * math.dist(x, y)
        for p, x in zip(probabilities, atoms, strict=True)
        for q, y in zip(probabilities, atoms, strict=True)
    )
    return attraction - spread / 2


def test_energy_score_cross_terms_and_law_invariance() -> None:
    """TEST-025/027/029: complementarity is lost if cross terms are omitted."""
    atoms = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    outcome = np.zeros(3)
    experts = np.array([[1.0, 0], [0.0, 1], [0.5, 0.5]])
    alpha = np.array([0.5, 0.5, 0.0])
    coefficients = score_coefficients(atoms, experts, outcome)
    assert coefficients.score(alpha) == pytest.approx(0.5)
    assert coefficients.score([1, 0, 0]) == pytest.approx(1)
    assert energy_score(atoms, alpha @ experts, outcome) == pytest.approx(
        scalar_es(atoms.tolist(), [0.5, 0.5], outcome.tolist())
    )
    # Permuting, duplicating an atom and translating preserve the same law/score.
    assert energy_score(atoms[::-1], [0.5, 0.5], outcome) == pytest.approx(0.5)
    assert energy_score(atoms[[0, 1, 1]], [0.5, 0.2, 0.3], outcome) == pytest.approx(0.5)
    assert energy_score(atoms + 17, [0.5, 0.5], outcome + 17) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "a,expected",
    [
        ([0, 0.01, 0.02], [13 / 30, 1 / 3, 7 / 30]),
        ([0, 0.02, 0.2], [0.6, 0.4, 0]),
        ([0, 0.2, 0.4], [1, 0, 0]),
    ],
)
def test_simplex_hand_interior_face_and_vertex(a: list[float], expected: list[float]) -> None:
    """TEST-033: analytic projection for B=0, lambda=.05, uniform anchor."""
    result = solve_simplex(a, np.zeros((3, 3)), [1 / 3] * 3)
    np.testing.assert_allclose(result.alpha, expected, atol=1e-12, rtol=0)
    assert len(result.faces) == 7
    gradient = np.asarray(a) + 0.1 * (result.alpha - 1 / 3)
    active = result.alpha > 1e-10
    multiplier = -float(gradient[active][0])
    assert np.max(np.abs(gradient[active] + multiplier)) < 1e-12
    assert np.all(gradient[~active] + multiplier >= -1e-12)
    # Independently enumerate a dense rational grid; no point can beat the solve.
    for i, j in product(range(41), repeat=2):
        if i + j <= 40:
            trial = np.array([i, j, 40 - i - j]) / 40
            objective = float(trial @ a + 0.05 * np.sum((trial - 1 / 3) ** 2))
            assert result.objective <= objective + 1e-12


def test_simplex_rejects_bad_curvature_and_preserves_scale_certificate() -> None:
    """TEST-033/036: numerical invalidity cannot be repaired by clipping B."""
    with pytest.raises(NumericalError, match="curvature"):
        solve_simplex([0, 0, 0], np.eye(3), [1 / 3] * 3)
    result = solve_simplex([1e6, 1e6 + 100, 1e6 + 200], np.zeros((3, 3)), [1 / 3] * 3)
    np.testing.assert_allclose(result.alpha, [1, 0, 0], rtol=0, atol=1e-12)
    assert result.coefficient_scale > 1e6
    assert result.residual <= 1e-9 * result.coefficient_scale


@pytest.mark.parametrize(
    "kwargs",
    [{"kkt": float("nan")}, {"condition_limit": float("inf")}, {"feasibility": -1}, {"symmetry": True}],
)
def test_solver_tolerances_fail_closed(kwargs: dict[str, float]) -> None:
    """TEST-003: invalid numerical certificates cannot silently loosen checking."""
    with pytest.raises(ContractError):
        SolverTolerances(**kwargs)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1, True])
def test_probability_tolerance_fail_closed(tolerance: float) -> None:
    """TEST-003: NaN tolerance must not admit a vector with twice unit mass."""
    with pytest.raises(ContractError):
        validate_probabilities([1, 1], tolerance=tolerance)


def test_exact_fee_roots_and_batch_agree_with_decimal() -> None:
    """TEST-042/052: analytic sign regions provide independent fee roots."""
    holdings = np.array([40000.0, 20000.0, 0])
    target = np.array([0.2, 0.3, 0.1])
    result = solve_rebalance(holdings, 40000.0, target, one_way_cost=0.01)
    with localcontext() as context:
        context.prec = 50
        reference = (Decimal(100000) - Decimal(".01") * Decimal(20000)) / Decimal("1.002")
    assert result.post_fee_nav == pytest.approx(float(reference), abs=1e-5, rel=0)
    assert result.cash + float(result.risky_dollars.sum()) + result.fee == pytest.approx(100000, abs=1e-5)
    batch = solve_rebalances(holdings, 40000.0, np.stack((target, np.zeros(3))), one_way_cost=0.01)
    np.testing.assert_allclose(batch.solution(0).risky_dollars, result.risky_dollars, atol=1e-5)
    assert batch.solution(1).post_fee_nav == pytest.approx(99400, abs=1e-5)
    scaled = solve_rebalance(holdings * 7, 280000.0, target, one_way_cost=0.01)
    assert scaled.post_fee_nav == pytest.approx(result.post_fee_nav * 7, rel=1e-10)
    hold = keep_holdings_solution(holdings, 40000.0)
    np.testing.assert_array_equal(hold.risky_dollars, holdings)
    assert hold.fee == hold.turnover == hold.residual == 0


def test_all_cash_rebalance_preserves_exact_known_root() -> None:
    """TEST-042/051: numeric root tolerance cannot create cumulative cash loss."""
    cash = 100.0
    for _ in range(10):
        solution = solve_rebalance(np.zeros(3), cash, np.zeros(3), one_way_cost=0.004)
        cash = solution.cash
        assert cash == 100 and solution.fee == solution.turnover == solution.residual == 0


def test_crra_matches_decimal_and_zero_nominal_atoms_affect_robustness() -> None:
    """TEST-048/049: contamination includes zero-weight support and keeps cash neutral."""
    factors = np.array([0.75, 1, 1.25])
    utility = crra_utility(factors)
    with localcontext() as context:
        context.prec = 50
        expected = [float((Decimal(str(x)) ** -3 - 1) / -3) for x in factors]
    np.testing.assert_allclose(utility, expected, rtol=1e-14, atol=1e-15)
    assert utility[1] == 0
    objective = robust_objective(utility, np.array([0.0, 0.0, 1.0]), rho=0.05)
    assert objective[0] == pytest.approx(0.95 * utility[2] + 0.05 * utility[0])


@pytest.mark.parametrize("invalid", [np.array([True]), np.array([1], dtype=np.float32)])
def test_utility_rejects_implicit_dtype_changes(invalid: NDArray[np.generic]) -> None:
    """TEST-002: float64/bool contract must also hold at utility boundaries."""
    with pytest.raises(ContractError):
        crra_utility(cast(FloatArray, invalid))
    with pytest.raises(ContractError):
        robust_objective(cast(FloatArray, invalid), np.array([1.0]))


def test_utility_outputs_are_immutable() -> None:
    """TEST-005: immutable scientific arrays cannot be altered through return values."""
    for output in (crra_utility(np.array([1.0])), robust_objective(np.array([0.0]), np.array([1.0]))):
        with pytest.raises(ValueError):
            output.setflags(write=True)


def test_discrete_quantiles_and_fixed_calibration_edges() -> None:
    """TEST-031: zero mass is excluded and probability one occupies the last bin."""
    np.testing.assert_array_equal(
        discrete_quantiles([-9, 2, 1, 2], [0, 0.25, 0.5, 0.25], [0, 0.5, 0.75, 1]),
        [1, 1, 2, 2],
    )
    bins = calibration_bins([0, 0.1, 0.9, 1], [0, 1, 0, 1])
    assert len(bins) == 10
    assert [bins[i]["count"] for i in (0, 1, 9)] == [1, 1, 2]
    assert bins[9]["brier"] == pytest.approx(0.405)


@pytest.mark.parametrize("scorer", [mean_squared_error, variogram_score])
def test_overflowing_scores_fail_explicitly(
    scorer: Callable[[ArrayLike, ArrayLike, ArrayLike], float],
) -> None:
    """TEST-003: finite inputs with unrepresentable arithmetic cannot yield inf."""
    with pytest.raises(NumericalError):
        scorer(np.array([[1e308, -1e308, 0]]), [1], [-1e308, 1e308, 0])


def test_reject_bool_neighbor_count_and_noninteger_clock() -> None:
    """TEST-002/012: bool-as-count and fractional clocks are not valid metadata."""
    with pytest.raises(ContractError):
        extrapolation_threshold(np.zeros((3, 2)), neighbors=True)
    with pytest.raises(ContractError):
        recency_weights(np.array([0, 1], dtype=np.int64), decision_us=cast(int, 2.5))
