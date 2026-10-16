"""Independent small references and boundary cases for catalog obligations."""

import math
from dataclasses import replace
from decimal import Decimal, localcontext
from itertools import permutations

import numpy as np
import pytest
from numpy.typing import NDArray
from test_model_audit import BASE, WEEK, feedback_records, record, request, transform

from frem_s10.arrays import QuantityArray, frozen_f64
from frem_s10.domain import ForecastDistribution
from frem_s10.errors import ChronologyError, ContractError, NumericalError
from frem_s10.math.distances import extrapolation_threshold, neighbor_order, pairwise_distances
from frem_s10.math.fees import solve_rebalance
from frem_s10.math.scores import energy_score, marginal_crps, mean_squared_error, score_coefficients
from frem_s10.math.simplex import solve_simplex
from frem_s10.math.utility import crra_utility, robust_objective
from frem_s10.metadata import Unit, parse_utc_timestamp
from frem_s10.model.config import ForecastConfig
from frem_s10.model.experts import issue_experts
from frem_s10.model.library import empty_library, update_library
from frem_s10.model.stacking import fit_mixtures
from frem_s10.portfolio.benchmarks import unconditional_covariance


def test_all_declared_units_reject_every_wrong_unit_without_conversion() -> None:
    """TEST-001: identical numeric shapes cannot make incompatible units interchangeable."""
    for actual in Unit:
        quantity = QuantityArray(np.array([0.5, 0.75]), actual)
        np.testing.assert_array_equal(quantity.require(actual, shape=(2,)), [0.5, 0.75])
        for requested in Unit:
            if requested != actual:
                with pytest.raises(ContractError, match="unit mismatch"):
                    quantity.require(requested, shape=(2,))


@pytest.mark.parametrize(
    "invalid",
    (
        np.array([], dtype=np.float64),
        np.zeros((1, 3)),
        np.zeros(2),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=object),
        np.zeros(3, dtype=np.complex128),
        np.array([True, False, True]),
        np.array([0.0, math.nan, 0.0]),
        np.array([0.0, math.inf, 0.0]),
    ),
)
def test_typed_numeric_boundary_rejects_shape_dtype_and_nonfinite(invalid: NDArray[np.generic]) -> None:
    """TEST-002/003: shape and numeric type failures occur before arithmetic."""
    with pytest.raises(ContractError):
        frozen_f64(invalid, shape=(3,), name="declared-three-assets")


@pytest.mark.parametrize("clock", ("2026-10-15T09:30:00", "2026-10-15", "not-a-date"))
def test_timestamp_without_timezone_is_not_assumed_utc(clock: str) -> None:
    """TEST-004: a plausible local timestamp must not silently become UTC."""
    with pytest.raises(ContractError):
        parse_utc_timestamp(clock)


@pytest.mark.parametrize("offset", (0, 1, WEEK))
def test_feature_equal_or_after_decision_is_unavailable(offset: int) -> None:
    """TEST-012: strict availability includes equality and future timestamps."""
    original = request(70)
    with pytest.raises(ChronologyError):
        replace(original, feature_available_us=original.decision_us + offset)


def test_exact_expert_probabilities_and_label_independent_neighbor_ties() -> None:
    """TEST-023/039/040: derive U/R/C by scalar ages and deterministic tie keys."""
    records = tuple(replace(record(i), x=np.zeros(12)) for i in range(64))
    config = ForecastConfig()
    decision = BASE + 80 * WEEK
    library = update_library(empty_library(), records, decision_us=decision, config=config)
    experts = issue_experts(library, np.zeros(12), decision_us=decision, transform=transform(), config=config)
    recent = [2.0 ** (-(decision - item.origin_us) / WEEK / 52.0) for item in records]
    total = math.fsum(recent)
    np.testing.assert_allclose(experts.probabilities[0], np.full(64, 1 / 64), rtol=0, atol=0)
    np.testing.assert_allclose(experts.probabilities[1], [value / total for value in recent], rtol=1e-14)
    assert experts.neighbor_indices == tuple(range(32))
    np.testing.assert_array_equal(experts.probabilities[2], np.r_[np.full(32, 1 / 32), np.zeros(32)])
    changed = replace(library, g=library.g * 7, d=library.d + 0.1)
    second = issue_experts(changed, np.zeros(12), decision_us=decision, transform=transform(), config=config)
    np.testing.assert_array_equal(second.probabilities, experts.probabilities)
    assert neighbor_order([1, 1, 1], np.array([9, 8, 8]), ("c", "b", "a")) == (2, 1, 0)
    # Threshold equality is admitted; any positive squared distance exceeds zero.
    assert experts.neighbor_distance == experts.extrapolation_threshold == 0
    outside = issue_experts(
        library, np.ones(12) * 1e-8, decision_us=decision, transform=transform(), config=config
    )
    assert outside.fallback_reason == "CONTEXT_EXTRAPOLATION"


def test_leave_one_out_threshold_31_32_33_matches_scalar_order_statistic() -> None:
    """TEST-024: no self-neighbor and explicit linear 95th percentile."""
    for count in (31, 32):
        with pytest.raises(ContractError):
            extrapolation_threshold(np.zeros((count, 1)), neighbors=32)
    x = np.arange(33, dtype=np.float64)[:, None]
    values = sorted(sorted(float(i - j) ** 2 for j in range(33) if i != j)[31] for i in range(33))
    position = 0.95 * 32
    lower = math.floor(position)
    expected = values[lower] + (position - lower) * (values[lower + 1] - values[lower])
    assert extrapolation_threshold(x, neighbors=32) == expected


def test_one_atom_energy_crps_mse_and_cross_mixture_reference() -> None:
    """TEST-026/028/031: deterministic distances and arbitrary interior mixtures."""
    z = np.array([[2.0, 3.0, 6.0]])
    assert energy_score(z, [1], z[0]) == 0
    assert energy_score(z, [1], np.zeros(3)) == 7
    np.testing.assert_array_equal(marginal_crps(z, [1], np.zeros(3)), [2, 3, 6])
    assert mean_squared_error(z, [1], np.zeros(3)) == pytest.approx(49 / 3)
    atoms = np.array([[-2.0, 1, 0], [1.0, 3, -1], [4.0, -1, 2]])
    laws = np.array([[0.5, 0.25, 0.25], [0.1, 0.7, 0.2], [0.3, 0.2, 0.5]])
    target = np.array([0.2, 0.7, -0.1])
    coefficients = score_coefficients(atoms, laws, target)
    np.testing.assert_allclose(coefficients.b, coefficients.b.T, atol=1e-15)
    distances = np.array([[math.dist(a, b) for b in atoms] for a in atoms])
    np.testing.assert_array_equal(np.diag(distances), np.zeros(3))
    for alpha in ([1, 0, 0], [0.2, 0.8, 0], [0.17, 0.31, 0.52]):
        mass = np.asarray(alpha) @ laws
        attraction = math.fsum(float(p) * math.dist(row, target) for p, row in zip(mass, atoms, strict=True))
        spread = math.fsum(
            float(mass[i] * mass[j]) * math.dist(atoms[i], atoms[j]) for i in range(3) for j in range(3)
        )
        assert coefficients.score(alpha) == pytest.approx(attraction - spread / 2, rel=1e-14)


def test_score_rotations_rescaling_and_stable_large_coordinate_distance() -> None:
    """TEST-029/032: exact orthogonal permutation/signs and Decimal distance."""
    atoms = np.array([[-1.0, 2, 0.5], [3.0, -1, 2]])
    y = np.array([0.2, -0.3, 1.0])
    rotation = np.array([[0.0, -1, 0], [0, 0, 1], [1, 0, 0]])
    score = energy_score(atoms, [0.4, 0.6], y)
    assert energy_score(atoms @ rotation, [0.4, 0.6], y @ rotation) == pytest.approx(score)
    assert energy_score(atoms * 7, [0.4, 0.6], y * 7) == pytest.approx(score * 7)
    left = np.array([[1e12, -1e12, 1e12]])
    right = left + np.array([[0.001, 0.002, -0.003]])
    with localcontext() as context:
        context.prec = 70
        squared = sum(
            (
                (Decimal.from_float(float(a)) - Decimal.from_float(float(b))) ** 2
                for a, b in zip(left[0], right[0], strict=True)
            ),
            Decimal(0),
        )
        expected = float(squared.sqrt())
    assert pairwise_distances(left, right)[0, 0] == pytest.approx(expected, rel=1e-15)
    assert pairwise_distances([[0.0]], [[1e-150]])[0, 0] == pytest.approx(1e-150, rel=1e-15, abs=0)


def test_simplex_clipping_is_not_projection_and_identical_experts_keep_anchor() -> None:
    """TEST-034/035/036: boundary optimum beats clip-renormalize; tangent curvature suffices."""
    a = np.array([0.0, 0.02, 0.2])
    uniform = np.full(3, 1 / 3)
    result = solve_simplex(a, np.zeros((3, 3)), uniform)
    unconstrained = uniform - (a - a.mean()) / 0.1
    clipped = np.maximum(unconstrained, 0)
    clipped /= clipped.sum()
    clipped_objective = float(clipped @ a + 0.05 * np.square(clipped - uniform).sum())
    assert result.objective < clipped_objective - 1e-4
    for permutation in permutations(range(3)):
        order = np.asarray(permutation)
        permuted = solve_simplex(a[order], np.zeros((3, 3)), uniform)
        np.testing.assert_allclose(permuted.alpha, result.alpha[order], atol=1e-12)
    anchor = np.array([0.2, 0.3, 0.5])
    identical = solve_simplex([1, 1, 1], np.ones((3, 3)), anchor)
    np.testing.assert_allclose(identical.alpha, anchor, atol=1e-12)
    assert np.linalg.eigvalsh(-np.ones((3, 3)) + 0.1 * np.eye(3))[0] < 0
    with pytest.raises(NumericalError, match="asymmetric"):
        solve_simplex([0, 0, 0], [[0, 1, 1], [0, 0, 1], [1, 1, 0]], anchor)


def test_global_25_26_local_32_33_and_recency_individual_reference() -> None:
    """TEST-038/039/040: exact fit eligibility and nonconstant individual-score weights."""
    config = ForecastConfig()
    decision = BASE + 100 * WEEK
    results = {
        n: fit_mixtures(feedback_records(n), np.zeros(12), decision_us=decision, config=config)
        for n in (25, 26, 32, 33)
    }
    assert results[25].global_certificate is None
    assert results[26].global_certificate is not None
    assert results[32].local_certificate is None
    assert results[33].local_certificate is not None
    rows = tuple(
        replace(item, a=np.array([i / 100, (50 - i) / 100, 0.3]))
        for i, item in enumerate(feedback_records(40))
    )
    result = fit_mixtures(rows, np.zeros(12), decision_us=decision, config=config)
    selected = rows[-26:]
    weights = [2 ** (-(decision - item.origin_us) / WEEK / 52) for item in selected]
    total = math.fsum(weights)
    means = [
        math.fsum(w / total * float(item.a[j]) for w, item in zip(weights, selected, strict=True))
        for j in range(3)
    ]
    expected = np.exp(-np.array(means))
    expected /= expected.sum()
    np.testing.assert_allclose(result.individual_alpha, expected, rtol=1e-14)


def test_sale_root_near_one_crra_and_all_contamination_vertices() -> None:
    """TEST-042/048/049: closed form sale, Decimal utility, complete adversarial simplex."""
    sale = solve_rebalance(np.array([40.0, 0, 0]), 60.0, np.zeros(3), one_way_cost=0.01)
    assert sale.cash == pytest.approx(99.6, abs=1e-8) and sale.fee == pytest.approx(0.4)
    factors = np.array([np.nextafter(1.0, 2.0), np.nextafter(1.0, 0.0), 1e-5])
    with localcontext() as context:
        context.prec = 80
        expected = [float((Decimal.from_float(float(x)) ** -3 - 1) / -3) for x in factors]
    np.testing.assert_allclose(crra_utility(factors), expected, rtol=1e-14, atol=1e-29)
    for bad in (0.0, -1.0):
        with pytest.raises(ContractError):
            crra_utility(np.array([bad]))
    utilities = np.array([-3.0, 0.5, 2.0])
    p = np.array([0.0, 0.4, 0.6])
    values = [float(((1 - 0.05) * p + 0.05 * vertex) @ utilities) for vertex in np.eye(3)]
    assert robust_objective(utilities, p)[0] == pytest.approx(min(values))
    assert robust_objective(utilities, p, rho=0)[0] == pytest.approx(float(p @ utilities))


def test_joint_law_moments_psd_and_atom_split_invariance() -> None:
    """TEST-025/030: covariance preserves aligned and offsetting joint dependence."""
    for atoms in (np.array([[-1.0, -1, -1], [1.0, 1, 1]]), np.array([[-1.0, 1, 0], [1.0, -1, 0]])):
        mean = np.mean(atoms, axis=0)
        covariance = sum(np.outer(row - mean, row - mean) / 2 for row in atoms)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-14
        for direction in (np.ones(3), np.array([1.0, -1, 2])):
            expected = math.fsum(float(np.dot(row - mean, direction)) ** 2 / 2 for row in atoms)
            assert float(direction @ covariance @ direction) == pytest.approx(expected)
        original = ForecastDistribution("law", atoms / 10 + 1, np.zeros_like(atoms), np.array([0.5, 0.5]))
        np.testing.assert_allclose(
            unconditional_covariance(original), covariance / 100, rtol=1e-14, atol=1e-15
        )
        split = ForecastDistribution(
            "split", original.g[[0, 1, 1]], original.d[[0, 1, 1]], np.array([0.5, 0.2, 0.3])
        )
        target = np.zeros(3)
        assert energy_score(original.r, original.probabilities, target) == pytest.approx(
            energy_score(split.r, split.probabilities, target)
        )


@pytest.mark.parametrize("epsilon,feasible", ((1e-13, True), (1e-10, False)))
def test_tiny_negative_simplex_face_is_corrected_only_within_tolerance(
    epsilon: float, feasible: bool
) -> None:
    """TEST-037: substantive infeasibility cannot be relabeled as floating roundoff."""
    anchor = np.full(3, 1 / 3)
    unconstrained = np.array([-epsilon, 0.4, 0.6 + epsilon])
    a = 0.1 * (anchor - unconstrained)
    result = solve_simplex(a, np.zeros((3, 3)), anchor)
    full = next(face for face in result.faces if face.face == (0, 1, 2))
    assert full.feasible is feasible
    assert np.all(result.alpha >= 0) and abs(result.alpha.sum() - 1) < 1e-12
    assert result.objective == pytest.approx(
        float(result.alpha @ a + 0.05 * np.square(result.alpha - anchor).sum()), abs=1e-15
    )
    assert result.residual <= 1e-9


@pytest.mark.parametrize("recent_count,expected", ((16, None), (15, "LOCAL_CONCENTRATION")))
def test_local_effective_count_at_and_below_floor(recent_count: int, expected: str | None) -> None:
    """TEST-038: zero-underflow old weights yield exact effective counts 16 and 15."""
    rows = feedback_records(33)
    old = BASE - 2000 * 52 * WEEK
    # The last context is excluded by distance; 32 tied candidates remain.
    changed = tuple(
        replace(
            item,
            origin_us=old if i < 32 - recent_count else BASE,
            available_us=BASE + i + 1,
            context=np.zeros(12) if i < 32 else np.ones(12),
        )
        for i, item in enumerate(rows)
    )
    result = fit_mixtures(changed, np.zeros(12), decision_us=BASE + WEEK, config=ForecastConfig())
    assert result.effective_count == recent_count
    assert result.fallback_reason == expected
