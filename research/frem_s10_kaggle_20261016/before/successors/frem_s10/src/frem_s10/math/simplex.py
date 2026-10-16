"""Certified three-expert regularized energy-score minimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from frem_s10.arrays import frozen_f64
from frem_s10.errors import ContractError, NumericalError
from frem_s10.math.probabilities import validate_probabilities
from frem_s10.metadata import finite_float


@dataclass(frozen=True, slots=True)
class SolverTolerances:
    """Finite nonnegative numerical thresholds for explicit solver certification."""

    feasibility: float = 1e-12
    symmetry: float = 1e-12
    kkt: float = 1e-9
    curvature: float = 1e-10
    condition_limit: float = 1e12

    def __post_init__(self) -> None:
        """Reject invalid tolerances before any constrained solve can begin."""
        for name in ("feasibility", "symmetry", "kkt", "curvature"):
            finite_float(getattr(self, name), name=name, minimum=0)
        finite_float(self.condition_limit, name="condition_limit", minimum=1)


DEFAULT_TOLERANCES = SolverTolerances()


@dataclass(frozen=True, slots=True)
class FaceCertificate:
    """Retain feasibility, conditioning and original-scale KKT evidence for one face."""

    face: tuple[int, ...]
    condition: float
    feasible: bool
    certified: bool
    residual: float
    multiplier: float


@dataclass(frozen=True, slots=True)
class CertifiedMixture:
    """Preserve the selected mixture and all seven face certificates."""

    alpha: NDArray[np.float64]
    objective: float
    active_face: tuple[int, ...]
    residual: float
    coefficient_scale: float
    faces: tuple[FaceCertificate, ...]

    def __post_init__(self) -> None:
        """Own immutable simplex weights at the public result boundary."""
        object.__setattr__(self, "alpha", validate_probabilities(self.alpha, size=3))


def _validate_curvature(b: NDArray[np.float64], tolerance: SolverTolerances) -> None:
    scale = max(1.0, float(np.abs(b).max()))
    if float(np.abs(b - b.T).max()) > tolerance.symmetry * scale:
        raise NumericalError("energy-score B is asymmetric")
    basis = np.array(
        [[1 / np.sqrt(2), 1 / np.sqrt(6)], [-1 / np.sqrt(2), 1 / np.sqrt(6)], [0, -2 / np.sqrt(6)]],
        dtype=np.float64,
    )
    curvature = basis.T @ (-b) @ basis
    minimum = float(np.linalg.eigvalsh(curvature).min())
    if minimum < -tolerance.curvature * scale:
        raise NumericalError("negative energy-score curvature on simplex tangent")


def _face_candidate(
    h: NDArray[np.float64],
    a: NDArray[np.float64],
    face: tuple[int, ...],
    *,
    scale: float,
    tolerances: SolverTolerances,
) -> tuple[NDArray[np.float64], FaceCertificate]:
    indices = np.asarray(face, dtype=np.int64)
    size = len(face)
    system = np.empty((size + 1, size + 1), dtype=np.float64)
    system[:size, :size] = h[np.ix_(indices, indices)] / scale
    system[:size, size] = 1.0
    system[size, :size] = 1.0
    system[size, size] = 0.0
    rhs = np.concatenate((-a[indices] / scale, [1.0]))
    condition = float(np.linalg.cond(system, p=2))
    if not np.isfinite(condition) or condition > tolerances.condition_limit:
        raise NumericalError(f"uncertified bordered face condition: {face}")
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise NumericalError(f"singular bordered face: {face}") from exc
    alpha = np.zeros(3, dtype=np.float64)
    alpha[indices] = solution[:size]
    multiplier = float(solution[size] * scale)
    feasible = bool(
        np.all(alpha >= -tolerances.feasibility) and abs(float(alpha.sum()) - 1) <= tolerances.feasibility
    )
    if not feasible:
        return alpha, FaceCertificate(face, condition, False, False, float(np.abs(alpha).max()), multiplier)
    if np.any(alpha < 0):
        alpha = np.maximum(alpha, 0)
        alpha /= float(alpha.sum())
    gradient = h @ alpha + a + multiplier
    inactive = [i for i in range(3) if i not in face]
    stationarity = float(np.abs(gradient[indices]).max())
    dual = max(0.0, -float(gradient[inactive].min())) if inactive else 0.0
    complementary = float(np.abs(alpha * gradient).max())
    residual = max(stationarity, dual, complementary)
    certified = residual <= tolerances.kkt * scale
    return alpha, FaceCertificate(face, condition, True, certified, residual, multiplier)


def solve_simplex(
    mean_a: ArrayLike,
    mean_b: ArrayLike,
    anchor: ArrayLike,
    *,
    l2_coefficient: float = 0.05,
    tolerances: SolverTolerances = DEFAULT_TOLERANCES,
) -> CertifiedMixture:
    """Enumerate all seven faces, preserving original-scale KKT certificates.

    Scaling H/a changes the multiplier: nu_original=s*nu_scaled. No failing
    bordered numerical system is silently skipped, even on a nonoptimal face.
    """
    coefficient_a = frozen_f64(mean_a, shape=(3,), name="mean A")
    b = frozen_f64(mean_b, shape=(3, 3), name="mean B")
    center = validate_probabilities(anchor, size=3)
    l2_coefficient = finite_float(l2_coefficient, name="l2_coefficient", strictly_positive=True)
    if not isinstance(tolerances, SolverTolerances):
        raise ContractError("declared solver tolerances required")
    _validate_curvature(b, tolerances)
    h = -b + 2.0 * l2_coefficient * np.eye(3, dtype=np.float64)
    a = coefficient_a - 2.0 * l2_coefficient * center
    scale = max(1.0, float(np.abs(h).max()), float(np.abs(a).max()))
    certificates: list[FaceCertificate] = []
    candidates: list[tuple[float, tuple[float, ...], NDArray[np.float64], FaceCertificate]] = []
    for mask in range(1, 8):
        face = tuple(i for i in range(3) if mask & (1 << i))
        alpha, certificate = _face_candidate(h, a, face, scale=scale, tolerances=tolerances)
        certificates.append(certificate)
        if certificate.certified:
            objective = float(
                alpha @ coefficient_a
                - 0.5 * alpha @ b @ alpha
                + l2_coefficient * np.sum((alpha - center) ** 2)
            )
            if not np.isfinite(objective):
                raise NumericalError("nonfinite certified mixture objective")
            candidates.append((objective, tuple(float(x) for x in alpha), alpha, certificate))
    if not candidates:
        raise NumericalError("no active-face candidate satisfies the KKT certificate")
    objective, _, alpha, chosen = min(candidates, key=lambda item: (item[0], item[1]))
    return CertifiedMixture(alpha, objective, chosen.face, chosen.residual, scale, tuple(certificates))
