"""Deterministic direct-distance kernels with bounded temporary storage."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from frem_s10.arrays import frozen_f64
from frem_s10.errors import ContractError, NumericalError
from frem_s10.metadata import strict_int


def pairwise_distances(
    left: ArrayLike, right: ArrayLike | None = None, *, squared: bool = False
) -> NDArray[np.float64]:
    """Compute rowwise direct norms, avoiding cancellation of dot-product identities."""
    x = frozen_f64(left, shape=(None, None), name="left coordinates")
    y = x if right is None else frozen_f64(right, shape=(None, x.shape[1]), name="right coordinates")
    if x.shape[1] == 0:
        raise ContractError("coordinate dimension cannot be zero")
    result = np.empty((len(x), len(y)), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for i, row in enumerate(x):
                difference = y - row
                norm2 = np.sum(difference * difference, axis=1, dtype=np.float64)
                result[i] = norm2 if squared else np.sqrt(norm2)
    except FloatingPointError as exc:
        raise NumericalError("distance arithmetic overflow") from exc
    return frozen_f64(result, shape=result.shape, name="pair distances")


def neighbor_order(
    distances: ArrayLike, origins_us: ArrayLike, record_ids: tuple[str, ...]
) -> tuple[int, ...]:
    """Distance, origin and stable record identity define every neighbor tie."""
    d = frozen_f64(distances, shape=(len(record_ids),), name="neighbor distances")
    origins = np.asarray(origins_us)
    if origins.shape != d.shape or origins.dtype.kind not in "iu":
        raise ContractError("neighbor origin identity mismatch")
    if len(set(record_ids)) != len(record_ids) or np.any(d < 0):
        raise ContractError("duplicate neighbors or negative distance")
    return tuple(sorted(range(len(d)), key=lambda i: (float(d[i]), int(origins[i]), record_ids[i])))


def extrapolation_threshold(contexts: ArrayLike, *, neighbors: int) -> float:
    """Linear 95th percentile of leave-one-out kth squared distances."""
    strict_int(neighbors, name="neighbors", minimum=1)
    x = frozen_f64(contexts, shape=(None, None), name="reference contexts")
    if neighbors < 1 or len(x) <= neighbors:
        raise ContractError("leave-one-out lookup needs at least k+1 contexts")
    distances = np.array(pairwise_distances(x, squared=True), copy=True)
    np.fill_diagonal(distances, np.inf)
    kth = np.sort(distances, axis=1, kind="stable")[:, neighbors - 1]
    return float(np.quantile(kth, 0.95, method="linear"))
