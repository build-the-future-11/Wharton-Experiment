"""Fixed paired-fold S10 endpoints; every block family and control is retained.

Inputs are already saved, aligned ten-basis-point evidence. This module imports
no fitting, policy-selection or experiment execution path. Bootstrap summaries
are descriptive diagnostics of a retrospective path, never executable replays.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np

from frem_s10.arrays import FloatArray, IntArray, frozen_f64, frozen_i64
from frem_s10.errors import ContractError, NumericalError
from frem_s10.math.utility import crra_utility
from frem_s10.metadata import midnight_us

from .bootstrap import DRAWS, campaign_indices, relative_gain

SCORE_COLUMNS = ("contextual_stack", "global_stack", "equal")
WEALTH_COLUMNS = ("contextual_robust", "equal100", "buy_hold100", "equal80", "contextual_nominal")
INITIAL_WEALTH = 100_000.0


@dataclass(frozen=True, slots=True)
class FoldEvidence:
    """Complete aligned decision dates and fixed-column score/ledger evidence."""

    fold_id: str
    decision_us: IntArray
    origin_ids: tuple[str, ...]
    energy: FloatArray
    log_increments: FloatArray
    terminal_wealth: FloatArray
    maximum_drawdown: FloatArray

    def __post_init__(self) -> None:
        """Validate source-independent axes, chronology, scope and ledger telescoping."""
        if self.fold_id not in ("A", "B"):
            raise ContractError("fixed A/B fold required")
        count = len(self.origin_ids)
        if not 24 <= count <= 1000:
            raise ContractError("each fold requires 24..1000 complete dated blocks")
        if (
            not isinstance(self.origin_ids, tuple)
            or any(not isinstance(item, str) or not item for item in self.origin_ids)
            or len(set(self.origin_ids)) != count
        ):
            raise ContractError("unique complete origin identities required")
        object.__setattr__(self, "decision_us", frozen_i64(self.decision_us, shape=(count,), name="dates"))
        for name, shape in (
            ("energy", (count, 3)),
            ("log_increments", (count, 5)),
            ("terminal_wealth", (5,)),
            ("maximum_drawdown", (5,)),
        ):
            object.__setattr__(self, name, frozen_f64(getattr(self, name), shape=shape, name=name))
        start, end = (
            (date(2010, 1, 1), date(2013, 1, 1))
            if self.fold_id == "A"
            else (date(2013, 1, 1), date(2017, 1, 1))
        )
        if (
            np.any(np.diff(self.decision_us) <= 0)
            or self.decision_us[0] < midnight_us(start)
            or self.decision_us[-1] >= midnight_us(end)
        ):
            raise ContractError("dates violate fixed ordered fold scope")
        if np.any(self.energy < -1e-12) or np.any(self.terminal_wealth <= 0):
            raise ContractError("nonnegative scores and positive terminal wealth required")
        if np.any(self.maximum_drawdown < 0) or np.any(self.maximum_drawdown > 1):
            raise ContractError("drawdown must be in [0,1]")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            try:
                logged = np.sum(self.log_increments, axis=0)
                expected = np.log(self.terminal_wealth / INITIAL_WEALTH)
            except FloatingPointError as exc:
                raise NumericalError("nonfinite ledger endpoint arithmetic") from exc
        if not np.allclose(logged, expected, atol=1e-10, rtol=1e-10):
            raise ContractError("ledger boundary log increments do not telescope to final wealth")


@dataclass(frozen=True, slots=True)
class Comparison:
    """One required control/block comparison, including undefined endpoints."""

    control: str
    block: int
    fold_points: tuple[float, float] | None
    pooled_point: float | None
    lower: float | None
    upper: float | None
    reason: str | None = None
    lower_clears_margin: bool = False
    upper_at_or_below_margin: bool = False


@dataclass(frozen=True, slots=True)
class HypothesisResult:
    """A fixed H1/H2 gate result, not an unrestricted performance claim."""

    status: str
    comparisons: tuple[Comparison, ...]
    point_gates: tuple[tuple[str, bool], ...]


@dataclass(frozen=True, slots=True)
class UtilitySummary:
    """Descriptive CL-12 dated and terminal utility, robust and nominal separated."""

    fold_id: str
    observations: int
    robust_mean_block_utility: float
    nominal_mean_block_utility: float
    robust_terminal_utility: float | None
    nominal_terminal_utility: float | None


@dataclass(frozen=True, slots=True)
class StudyResult:
    """Conditional numeric disposition without source authority or scientific admission."""

    disposition: str
    h1: HypothesisResult | None
    h2: HypothesisResult | None
    h3: tuple[UtilitySummary, ...]
    bootstrap: tuple[tuple[str, int, IntArray], ...]
    reason: str | None = None
    evidence_scope: str = "COMPUTATIONAL_STATISTICS_SOURCE_AUTHORITY_UNVERIFIED"
    original_nll_hypothesis_satisfied: bool = False
    confirmation_claim: bool = False
    profitability_guarantee: bool = False


def _pooled_draws(
    first: FloatArray,
    second: FloatArray,
    first_indices: IntArray,
    second_indices: IntArray,
) -> FloatArray:
    """Weight every original date equally, retaining fold-specific paired draws."""
    with np.errstate(over="raise", invalid="raise"):
        try:
            values = np.sum(first[first_indices], axis=1) + np.sum(second[second_indices], axis=1)
            values = values / (len(first) + len(second))
        except FloatingPointError as exc:
            raise NumericalError("nonfinite pooled bootstrap mean") from exc
    return frozen_f64(values, shape=(DRAWS,), name="pooled_bootstrap_means")


def _interval(values: FloatArray) -> tuple[float, float]:
    quantiles = np.quantile(values, [0.025, 0.975], method="linear")
    return float(quantiles[0]), float(quantiles[1])


def _h1_comparison(
    folds: tuple[FoldEvidence, FoldEvidence],
    indices: dict[tuple[str, int], IntArray],
    column: int,
    block: int,
) -> Comparison:
    first, second = folds
    control = SCORE_COLUMNS[column]
    try:
        points = (
            relative_gain(first.energy[:, 0], first.energy[:, column]),
            relative_gain(second.energy[:, 0], second.energy[:, column]),
        )
        point = relative_gain(
            np.concatenate((first.energy[:, 0], second.energy[:, 0])),
            np.concatenate((first.energy[:, column], second.energy[:, column])),
        )
        numerator = _pooled_draws(
            first.energy[:, 0], second.energy[:, 0], indices[("A", block)], indices[("B", block)]
        )
        denominator = _pooled_draws(
            first.energy[:, column], second.energy[:, column], indices[("A", block)], indices[("B", block)]
        )
        if np.any(denominator <= 1e-12):
            raise NumericalError("required bootstrap score denominator is undefined; no draw dropped")
        ratios = frozen_f64(numerator / denominator, shape=(DRAWS,), name="score_ratios")
        ratio_lower, ratio_upper = _interval(ratios)
        # Compare the equivalent ratio boundary before subtracting from one:
        # a represented 19/20 ratio must not pass due to cancellation in 1-ratio.
        return Comparison(
            control,
            block,
            points,
            point,
            1 - ratio_upper,
            1 - ratio_lower,
            lower_clears_margin=ratio_upper < 0.95,
            upper_at_or_below_margin=ratio_lower >= 0.95,
        )
    except NumericalError as exc:
        return Comparison(control, block, None, None, None, None, str(exc))


def _status(comparisons: tuple[Comparison, ...], point_pass: bool) -> str:
    if any(row.reason is not None for row in comparisons):
        return "INCONCLUSIVE"
    if point_pass and all(row.lower_clears_margin for row in comparisons):
        return "PASS"
    if not point_pass or any(row.upper_at_or_below_margin for row in comparisons):
        return "FAIL"
    return "INCONCLUSIVE"


def _assess_h1(
    folds: tuple[FoldEvidence, FoldEvidence],
    indices: dict[tuple[str, int], IntArray],
) -> HypothesisResult:
    """Require both score controls, both positive folds and both strict 5% bounds."""
    comparisons = tuple(
        _h1_comparison(folds, indices, column, block) for column in (1, 2) for block in (2, 4)
    )
    gates = tuple(
        (
            f"{row.control}/both_folds_positive",
            row.fold_points is not None and all(point > 0 for point in row.fold_points),
        )
        for row in comparisons
        if row.block == 2
    )
    return HypothesisResult(_status(comparisons, all(passed for _, passed in gates)), comparisons, gates)


def _h2_comparison(
    folds: tuple[FoldEvidence, FoldEvidence],
    indices: dict[tuple[str, int], IntArray],
    column: int,
    block: int,
) -> Comparison:
    first = folds[0].log_increments[:, 0] - folds[0].log_increments[:, column]
    second = folds[1].log_increments[:, 0] - folds[1].log_increments[:, column]
    try:
        lower, upper = _interval(_pooled_draws(first, second, indices[("A", block)], indices[("B", block)]))
        point = float((np.sum(first) + np.sum(second)) / (len(first) + len(second)))
        return Comparison(
            WEALTH_COLUMNS[column],
            block,
            (float(np.mean(first)), float(np.mean(second))),
            point,
            lower,
            upper,
            lower_clears_margin=lower > 0,
            upper_at_or_below_margin=upper <= 0,
        )
    except NumericalError as exc:
        return Comparison(WEALTH_COLUMNS[column], block, None, None, None, None, str(exc))


def _assess_h2(
    folds: tuple[FoldEvidence, FoldEvidence],
    indices: dict[tuple[str, int], IntArray],
) -> HypothesisResult:
    """Require both broad 6% wealth gates, cash, risk and paired positive log bounds."""
    comparisons = tuple(
        _h2_comparison(folds, indices, column, block) for column in (1, 2) for block in (2, 4)
    )
    gates: list[tuple[str, bool]] = []
    for fold in folds:
        primary = fold.terminal_wealth[0]
        for column in (1, 2):
            gates.append(
                (
                    f"{fold.fold_id}/{WEALTH_COLUMNS[column]}/at_least_6pct",
                    bool(primary >= 1.06 * fold.terminal_wealth[column]),
                )
            )
        gates.append((f"{fold.fold_id}/beats_cash", bool(primary > INITIAL_WEALTH)))
        gates.append(
            (
                f"{fold.fold_id}/drawdown_no_worse_than_equal80",
                bool(fold.maximum_drawdown[0] <= fold.maximum_drawdown[3]),
            )
        )
    return HypothesisResult(
        _status(comparisons, all(passed for _, passed in gates)), comparisons, tuple(gates)
    )


def utility_summaries(folds: tuple[FoldEvidence, FoldEvidence]) -> tuple[UtilitySummary, ...]:
    """Report equal-date block CRRA and separate monotone terminal CRRA."""
    rows: list[UtilitySummary] = []
    for fold in folds:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            try:
                block = crra_utility(np.exp(fold.log_increments[:, [0, 4]]))
                terminal = crra_utility(fold.terminal_wealth[[0, 4]] / INITIAL_WEALTH)
            except FloatingPointError as exc:
                raise NumericalError("nonfinite H3 utility factors") from exc
        rows.append(
            UtilitySummary(
                fold.fold_id,
                len(fold.origin_ids),
                float(np.mean(block[:, 0])),
                float(np.mean(block[:, 1])),
                float(terminal[0]),
                float(terminal[1]),
            )
        )
    count = rows[0].observations + rows[1].observations
    rows.append(
        UtilitySummary(
            "POOLED",
            count,
            sum(row.robust_mean_block_utility * row.observations for row in rows) / count,
            sum(row.nominal_mean_block_utility * row.observations for row in rows) / count,
            None,
            None,
        )
    )
    return tuple(rows)


def assess_study(
    folds: tuple[FoldEvidence, FoldEvidence],
    *,
    integrity_verified: bool,
    complete_matrix: bool,
) -> StudyResult:
    """Assess fixed H1/H2 and descriptive H3 without a favorable-control/block choice."""
    if type(integrity_verified) is not bool or type(complete_matrix) is not bool:
        raise ContractError("explicit boolean integrity and completeness attestations required")
    if not integrity_verified or not complete_matrix:
        return StudyResult("INVALID", None, None, (), (), "integrity or complete matrix not established")
    if not isinstance(folds, tuple) or len(folds) != 2 or tuple(f.fold_id for f in folds) != ("A", "B"):
        raise ContractError("complete ordered A/B evidence required")
    if set(folds[0].origin_ids) & set(folds[1].origin_ids):
        raise ContractError("fold origin identities must be disjoint")
    indices = campaign_indices(len(folds[0].origin_ids), len(folds[1].origin_ids))
    h1, h2 = _assess_h1(folds, indices), _assess_h2(folds, indices)
    status_pair = (h1.status, h2.status)
    if "INCONCLUSIVE" in status_pair:
        disposition = "INCONCLUSIVE"
    elif status_pair == ("PASS", "PASS"):
        disposition = "SUPPORTED"
    elif status_pair == ("FAIL", "FAIL"):
        disposition = "NEGATIVE"
    else:
        disposition = "MIXED"
    return StudyResult(
        disposition,
        h1,
        h2,
        utility_summaries(folds),
        tuple((fold, block, array) for (fold, block), array in indices.items()),
    )
