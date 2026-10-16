"""Fixed centered paired-fold secondary ablation inference, without model selection."""

from dataclasses import dataclass
from datetime import date

import numpy as np

from frem_s10.arrays import FloatArray, IntArray, frozen_f64, frozen_i64
from frem_s10.errors import ContractError, IntegrityError, NumericalError
from frem_s10.metadata import midnight_us

from .bootstrap import DRAWS, campaign_indices, holm16

ABLATION_IDS = tuple(f"A{number}" for number in range(1, 9))


@dataclass(frozen=True, slots=True)
class AblationFold:
    """Complete paired dates, primary ES and the exact eight ordered ablation scores."""

    fold_id: str
    decision_us: IntArray
    origin_ids: tuple[str, ...]
    primary: FloatArray
    scores: tuple[tuple[str, FloatArray], ...]

    def __post_init__(self) -> None:
        """Keep every fixed comparison; reject invented policy-only score variation."""
        if self.fold_id not in ("A", "B"):
            raise ContractError("fixed A/B ablation fold required")
        count = len(self.origin_ids)
        if (
            not isinstance(self.origin_ids, tuple)
            or not 24 <= count <= 1000
            or any(not isinstance(item, str) or not item for item in self.origin_ids)
            or len(set(self.origin_ids)) != count
        ):
            raise ContractError("24..1000 unique complete paired ablation origins required")
        dates = frozen_i64(self.decision_us, shape=(count,), name="ablation_dates")
        start, end = (
            (date(2010, 1, 1), date(2013, 1, 1))
            if self.fold_id == "A"
            else (date(2013, 1, 1), date(2017, 1, 1))
        )
        if np.any(np.diff(dates) <= 0) or dates[0] < midnight_us(start) or dates[-1] >= midnight_us(end):
            raise ContractError("ablation dates violate fixed ordered fold scope")
        object.__setattr__(self, "decision_us", dates)
        primary = frozen_f64(self.primary, shape=(count,), name="primary_ES")
        object.__setattr__(self, "primary", primary)
        if not isinstance(self.scores, tuple) or tuple(name for name, _ in self.scores) != ABLATION_IDS:
            raise ContractError("complete ordered A1..A8 ablation identity family required")
        scores = tuple(
            (name, frozen_f64(values, shape=(count,), name=f"{name}_ES")) for name, values in self.scores
        )
        if np.any(primary < -1e-12) or any(np.any(values < -1e-12) for _, values in scores):
            raise ContractError("finite nonnegative ES scores required")
        if any(not np.array_equal(values, primary) for name, values in scores if name in ("A4", "A5")):
            raise IntegrityError("A4/A5 policy-only ablations must retain exact zero ES differences")
        object.__setattr__(self, "scores", scores)


@dataclass(frozen=True, slots=True)
class AblationComparison:
    """One retained block-family test; positive delta favors the fixed primary."""

    ablation_id: str
    block: int
    fold_means: tuple[float, float]
    pooled_mean: float
    exceedances: int
    raw_p: float
    holm_p: float
    rejects_zero_advantage: bool
    structural_zero: bool


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Secondary computational evidence; no promotion of an ablation to primary."""

    comparisons: tuple[AblationComparison, ...]
    bootstrap: tuple[tuple[str, int, IntArray], ...]
    evidence_scope: str = "COMPUTATIONAL_STATISTICS_SOURCE_AUTHORITY_UNVERIFIED"
    confirmation_claim: bool = False
    changes_primary: bool = False


def _centered_test(
    first: FloatArray,
    second: FloatArray,
    first_indices: IntArray,
    second_indices: IntArray,
) -> tuple[tuple[float, float], float, int, float]:
    """Compare null-centered date-pooled bootstrap means to the observed mean."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            observed = float((np.sum(first) + np.sum(second)) / (len(first) + len(second)))
            first_centered, second_centered = first - observed, second - observed
            null_means = (
                np.sum(first_centered[first_indices], axis=1)
                + np.sum(second_centered[second_indices], axis=1)
            ) / (len(first) + len(second))
            fold_means = (float(np.mean(first)), float(np.mean(second)))
        except FloatingPointError as exc:
            raise NumericalError("nonfinite centered ablation bootstrap arithmetic") from exc
    if not np.isfinite(observed) or not np.all(np.isfinite(null_means)):
        raise NumericalError("nonfinite centered ablation endpoint")
    exceedances = int(np.count_nonzero(null_means >= observed))
    return fold_means, observed, exceedances, (1 + exceedances) / (DRAWS + 1)


def assess_ablations(folds: tuple[AblationFold, AblationFold]) -> AblationResult:
    """Retain all eight by two centered tests and apply one fixed Holm16 family."""
    if not isinstance(folds, tuple) or len(folds) != 2 or tuple(f.fold_id for f in folds) != ("A", "B"):
        raise ContractError("complete ordered A/B ablation evidence required")
    first, second = folds
    if set(first.origin_ids) & set(second.origin_ids):
        raise IntegrityError("ablation fold origin identities must be disjoint")
    indices = campaign_indices(len(first.origin_ids), len(second.origin_ids))
    provisional: list[AblationComparison] = []
    for (name, values_a), (_, values_b) in zip(first.scores, second.scores, strict=True):
        with np.errstate(over="raise", invalid="raise"):
            try:
                delta_a, delta_b = values_a - first.primary, values_b - second.primary
            except FloatingPointError as exc:
                raise NumericalError("nonfinite paired ablation differences") from exc
        for block in (2, 4):
            means, observed, count, p_value = _centered_test(
                delta_a, delta_b, indices[("A", block)], indices[("B", block)]
            )
            structural = name in ("A4", "A5")
            if structural and (observed != 0 or p_value != 1):
                raise IntegrityError("policy-only structural zero test cannot invent score variation")
            provisional.append(
                AblationComparison(name, block, means, observed, count, p_value, 1.0, False, structural)
            )
    adjusted = holm16({(row.ablation_id, row.block): row.raw_p for row in provisional})
    comparisons = tuple(
        AblationComparison(
            row.ablation_id,
            row.block,
            row.fold_means,
            row.pooled_mean,
            row.exceedances,
            row.raw_p,
            adjusted[(row.ablation_id, row.block)],
            adjusted[(row.ablation_id, row.block)] <= 0.05,
            row.structural_zero,
        )
        for row in provisional
    )
    return AblationResult(
        comparisons, tuple((fold, block, array) for (fold, block), array in indices.items())
    )
