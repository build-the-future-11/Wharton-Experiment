"""Conditional flow matching utilities."""

from __future__ import annotations

import numpy as np


def bridge_sample(
    z: np.ndarray,
    y: np.ndarray,
    tau: float,
) -> np.ndarray:
    """X_tau = (1 - tau) Z + tau Y."""
    return (1.0 - tau) * z + tau * y


def bridge_velocity(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Target velocity field v = Y - Z."""
    return y - z


def euler_integrate(
    v_fn,
    z0: np.ndarray,
    *,
    n_steps: int = 10,
) -> np.ndarray:
    """Integrate dz/dtau = v_fn(z, tau) from tau=0 to 1."""
    z = np.asarray(z0, dtype=float).copy()
    dt = 1.0 / n_steps
    tau = 0.0
    for _ in range(n_steps):
        v = v_fn(z, tau)
        z = z + dt * v
        tau += dt
    return z


def sample_base_noise(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    *,
    student_t_df: float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    if student_t_df is not None and student_t_df > 2:
        x = rng.standard_t(student_t_df, size=shape)
        x *= np.sqrt((student_t_df - 2) / student_t_df)
    else:
        x = rng.normal(size=shape)
    return x * scale
