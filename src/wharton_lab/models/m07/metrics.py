"""Covariance / projector metrics for Eigen-JEPA."""

from __future__ import annotations

import numpy as np


def chordal_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """|| sin Θ ||_F between subspaces via projectors."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    diff = P - Q
    return float(np.linalg.norm(diff, ord="fro"))


def projector_from_eigenvectors(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    return U @ U.T


def projector_distance(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.linalg.norm(P - Q, ord="fro"))


def projector_sign_invariant_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Distance between P=UU^T and Q=VV^T (invariant to column sign flips)."""
    P = projector_from_eigenvectors(U)
    Q = projector_from_eigenvectors(V)
    # also check flipped V
    Vflip = -V
    Q2 = projector_from_eigenvectors(Vflip)
    return min(projector_distance(P, Q), projector_distance(P, Q2))


def flag_small_eigengaps(eigenvalues: np.ndarray, threshold: float) -> np.ndarray:
    ev = np.sort(np.asarray(eigenvalues, dtype=float).ravel())
    if ev.size < 2:
        return np.array([False])
    gaps = np.diff(ev)
    return gaps < threshold
