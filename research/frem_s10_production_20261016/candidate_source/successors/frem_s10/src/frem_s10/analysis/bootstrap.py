"""Fixed paired circular-block streams and the prospective S10 precision rule."""

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np

from frem_s10.arrays import FloatArray, IntArray, frozen_f64, frozen_i64
from frem_s10.errors import ContractError, NumericalError
from frem_s10.metadata import strict_int

DRAWS = 5000


def circular_indices(count: int, block: int, generator: np.random.Generator) -> IntArray:
    """Generate CL-08's 5000 draws, with consecutive modular blocks per draw."""
    strict_int(count, name="date_count", minimum=1)
    strict_int(block, name="block_length", minimum=1)
    if count > 1000 or block not in (2, 4):
        raise ContractError("bootstrap size exceeds fixed S10 schedule")
    starts = generator.integers(0, count, size=(DRAWS, ceil(count / block)), dtype=np.int64)
    indices = ((starts[:, :, None] + np.arange(block)) % count).reshape(DRAWS, -1)[:, :count]
    return frozen_i64(indices, shape=(DRAWS, count), name="bootstrap_indices")


def campaign_indices(count_a: int, count_b: int) -> dict[tuple[str, int], IntArray]:
    """Use four isolated child streams in exactly A2, B2, A4, B4 order."""
    children = np.random.SeedSequence(1901).spawn(4)
    schedule = (("A", 2, count_a), ("B", 2, count_b), ("A", 4, count_a), ("B", 4, count_b))
    return {
        (fold, block): circular_indices(count, block, np.random.Generator(np.random.PCG64(child)))
        for (fold, block, count), child in zip(schedule, children, strict=True)
    }


def pilot_indices(count: int) -> dict[int, IntArray]:
    """Keep pilot seed 1902 separate from every campaign RNG stream."""
    children = np.random.SeedSequence(1902).spawn(2)
    return {
        block: circular_indices(count, block, np.random.Generator(np.random.PCG64(child)))
        for block, child in zip((2, 4), children, strict=True)
    }


def relative_gain(candidate: FloatArray, control: FloatArray) -> float:
    """Return a ratio of date means, never a mean of datewise ratios."""
    candidate = frozen_f64(candidate, shape=(None,), name="candidate_scores")
    control = frozen_f64(control, shape=candidate.shape, name="control_scores")
    if not len(candidate) or np.any(candidate < -1e-12) or np.any(control < -1e-12):
        raise ContractError("finite nonnegative complete date scores required")
    denominator = float(np.mean(control))
    if denominator <= 1e-12:
        raise NumericalError("relative ES denominator is undefined")
    value = 1.0 - float(np.mean(candidate)) / denominator
    if not np.isfinite(value):
        raise NumericalError("nonfinite relative ES improvement")
    return value


def resampled_gains(candidate: FloatArray, control: FloatArray, indices: IntArray) -> FloatArray:
    """Preserve complete paired dates and reject every undefined draw."""
    candidate = frozen_f64(candidate, shape=(None,), name="candidate_scores")
    control = frozen_f64(control, shape=candidate.shape, name="control_scores")
    relative_gain(candidate, control)
    indices = frozen_i64(indices, shape=(DRAWS, len(candidate)), name="paired_indices")
    if np.any(indices < 0) or np.any(indices >= len(candidate)):
        raise ContractError("bootstrap index outside date array")
    means_c = np.mean(control[indices], axis=1)
    means_p = np.mean(candidate[indices], axis=1)
    if np.any(means_c <= 1e-12):
        raise NumericalError("undefined bootstrap denominator; draws cannot be dropped")
    return frozen_f64(1 - means_p / means_c, shape=(DRAWS,), name="relative_bootstrap_gains")


@dataclass(frozen=True)
class PrecisionResult:
    """Planning heuristic only: neither achieved signal nor validated joint power."""

    status: str
    reason: str | None
    pilot_blocks: int
    projected_blocks: int
    comparisons: tuple[tuple[str, int, float, float, bool], ...]


def assess_precision(
    candidate: FloatArray, equal: FloatArray, global_stack: FloatArray, projected_a: int, projected_b: int
) -> PrecisionResult:
    """Evaluate all four fixed CL-09 pilot conditions without selecting parameters."""
    strict_int(projected_a, name="projected_A", minimum=24)
    strict_int(projected_b, name="projected_B", minimum=24)
    candidate = frozen_f64(candidate, shape=(None,), name="pilot_candidate")
    equal = frozen_f64(equal, shape=candidate.shape, name="pilot_equal")
    global_stack = frozen_f64(global_stack, shape=candidate.shape, name="pilot_global")
    count = len(candidate)
    total = projected_a + projected_b
    if count < 24 or np.array_equal(candidate, global_stack):
        return PrecisionResult(
            "PRECISION_UNIDENTIFIABLE", "too few blocks or no contextual difference", count, total, ()
        )
    indices = pilot_indices(count)
    comparisons: list[tuple[str, int, float, float, bool]] = []
    for name, control in (("equal", equal), ("global", global_stack)):
        for block in (2, 4):
            if np.ptp(candidate - control) == 0:
                return PrecisionResult(
                    "PRECISION_UNIDENTIFIABLE",
                    "zero score-difference variability",
                    count,
                    total,
                    tuple(comparisons),
                )
            try:
                gains = resampled_gains(candidate, control, indices[block])
                sd = float(np.std(gains, ddof=1))
            except NumericalError:
                return PrecisionResult(
                    "PRECISION_UNIDENTIFIABLE",
                    "undefined required score ratio",
                    count,
                    total,
                    tuple(comparisons),
                )
            if not np.isfinite(sd) or sd == 0:
                return PrecisionResult(
                    "PRECISION_UNIDENTIFIABLE",
                    "zero or nonfinite dispersion",
                    count,
                    total,
                    tuple(comparisons),
                )
            projected = sd * sqrt(count / total)
            passed = (1.96 + 0.84) * projected <= 0.10 - 0.05
            comparisons.append((name, block, sd, projected, passed))
    status = "PRECISION_PASS" if all(row[4] for row in comparisons) else "PRECISION_INADEQUATE"
    return PrecisionResult(status, None, count, total, tuple(comparisons))


def holm16(p_values: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    """Adjust the complete eight-ablation by two-block family, retaining zeros."""
    expected = {(f"A{i}", b) for i in range(1, 9) for b in (2, 4)}
    if set(p_values) != expected:
        raise ContractError("complete 16-comparison ablation family required")
    if any(isinstance(p, bool) or not np.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ContractError("p-values must be finite in [0,1]")
    if any(p_values[(a, b)] != 1 for a in ("A4", "A5") for b in (2, 4)):
        raise ContractError("policy-only ablations have structural ES p-value one")
    adjusted = {}
    previous = 0.0
    for index, key in enumerate(sorted(p_values, key=lambda k: (p_values[k], k))):
        previous = max(previous, min(1.0, (16 - index) * p_values[key]))
        adjusted[key] = previous
    return adjusted
