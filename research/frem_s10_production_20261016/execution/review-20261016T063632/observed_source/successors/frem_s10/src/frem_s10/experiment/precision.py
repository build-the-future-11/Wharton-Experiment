"""Fixed primary-only pre-2010 precision preparation on validated pilot inputs."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from frem_s10.analysis.bootstrap import DRAWS, PrecisionResult, assess_precision, pilot_indices
from frem_s10.arrays import FloatArray, IntArray, frozen_f64, frozen_i64
from frem_s10.data.precision import PrecisionInputs
from frem_s10.domain import ForecastDistribution, OriginRecord, Transform
from frem_s10.errors import ContractError, IntegrityError, ResourceLimitError
from frem_s10.math.scores import energy_score
from frem_s10.model.engine import advance, initial_state
from frem_s10.model.records import ForecastTransition, IssueRequest


@dataclass(frozen=True, slots=True)
class PrecisionScores:
    """Dated paired pilot scores and their original forecast/state provenance."""

    inputs_id: str
    origin_ids: tuple[str, ...]
    forecast_ids: tuple[tuple[str, str, str], ...]
    candidate: FloatArray
    equal: FloatArray
    global_stack: FloatArray
    weekly_issue_count: int
    final_state_id: str
    result: PrecisionResult
    evidence_scope: str
    bootstrap: tuple[tuple[int, IntArray], ...]
    cold_start_blocks: tuple[bool, ...]
    local_fallback_reasons: tuple[str | None, ...]
    authorizes_campaign: bool = False

    def __post_init__(self) -> None:
        """Keep paired scores complete and immutable; never drop inconvenient dates."""
        if (
            self.evidence_scope not in ("GENERATED_ENGINEERING", "SOURCE_VIEW_WITHOUT_SCIENTIFIC_ADMISSION")
            or self.authorizes_campaign is not False
        ):
            raise IntegrityError("pure precision calculation cannot authorize a scientific campaign")
        count = len(self.origin_ids)
        if len(set(self.origin_ids)) != count or len(self.forecast_ids) != count or count < 24:
            raise IntegrityError("complete unique paired pilot dates required")
        for name in ("candidate", "equal", "global_stack"):
            object.__setattr__(self, name, frozen_f64(getattr(self, name), shape=(count,), name=name))
        if not 1 <= self.weekly_issue_count <= 1000:
            raise ResourceLimitError("pilot weekly issue count exceeds the declared bound")
        if not self.inputs_id or not self.final_state_id or self.result.pilot_blocks != count:
            raise IntegrityError("pilot result provenance or count mismatch")
        self._validate_diagnostics(count)

    def _validate_diagnostics(self, count: int) -> None:
        if len(self.cold_start_blocks) != count or any(
            type(flag) is not bool for flag in self.cold_start_blocks
        ):
            raise IntegrityError("paired boolean cold-start flags required")
        allowed = (None, "INSUFFICIENT_LOCAL_RECORDS", "LOCAL_CONCENTRATION", "LOCAL_EXTRAPOLATION")
        if len(self.local_fallback_reasons) != count or any(
            reason not in allowed for reason in self.local_fallback_reasons
        ):
            raise IntegrityError("paired recognized local fallback reasons required")
        if tuple(block for block, _ in self.bootstrap) != (2, 4):
            raise IntegrityError("both fixed pilot bootstrap arrays required")
        expected = pilot_indices(count)
        arrays = tuple(
            (block, frozen_i64(array, shape=(DRAWS, count), name="pilot_indices"))
            for block, array in self.bootstrap
        )
        if any(not np.array_equal(array, expected[block]) for block, array in arrays):
            raise IntegrityError("pilot indices differ from fixed seed 1902 streams")
        object.__setattr__(self, "bootstrap", arrays)

    @property
    def cold_start_count(self) -> int:
        """Count retained pilot decisions issued before the first global fit."""
        return sum(self.cold_start_blocks)

    @property
    def local_fallback_counts(self) -> tuple[tuple[str, int], ...]:
        """Separate local fallback causes from global cold-start diagnostics."""
        return tuple(
            (reason, self.local_fallback_reasons.count(reason))
            for reason in ("INSUFFICIENT_LOCAL_RECORDS", "LOCAL_CONCENTRATION", "LOCAL_EXTRAPOLATION")
        )

    @property
    def exact_zero_difference_counts(self) -> tuple[tuple[str, int], ...]:
        """Count exact paired ES equalities; equality does not imply identical laws."""
        return (
            ("equal", int(np.count_nonzero(self.candidate == self.equal))),
            ("global_stack", int(np.count_nonzero(self.candidate == self.global_stack))),
        )


def _score(law: ForecastDistribution, record: OriginRecord, transform: Transform) -> float:
    return energy_score(transform.scale_return(law.r), law.probabilities, transform.scale_return(record.r))


def run_precision(
    inputs: PrecisionInputs,
    *,
    transition_sink: Callable[[str, ForecastTransition], None] | None = None,
) -> PrecisionScores:
    """Run the fixed candidate once; caller must provide the consumed guarded operation.

    This pure function performs no IO, source acquisition, parameter selection,
    plotting or campaign execution. The fixed kernel issues all forecast arms
    weekly, but only the predeclared contextual/equal/global pilot scores enter
    precision. Projected A/B counts come exclusively from input calendar plans.
    """
    if not isinstance(inputs, PrecisionInputs):
        raise ContractError("validated pilot-only inputs required")
    selected = set(inputs.pilot.origin_indices)
    records = tuple(
        record
        for record in inputs.records
        if record.origin_us >= 1199145600000000 and min(selected) <= record.anchor_index <= max(selected)
    )
    if not records or len(records) > 1000:
        raise ResourceLimitError("pilot issue schedule is empty or exceeds its bound")
    state = initial_state(inputs.transform)
    scores: list[tuple[float, float, float]] = []
    origins: list[str] = []
    identities: list[tuple[str, str, str]] = []
    cold_start: list[bool] = []
    local_fallback: list[str | None] = []
    for record in records:
        request = IssueRequest(
            record.record_id,
            record.origin_us,
            record.endpoint_us,
            record.available_us,
            record.feature_available_us,
            record.x,
            record.anchor_index,
        )
        matured = tuple(
            item
            for item in inputs.records
            if item.available_us < request.decision_us and item.record_id not in state.consumed_event_ids
        )
        step = advance(state, request, matured)
        if transition_sink is not None:
            transition_sink("primary", step)
        state = step.state
        if record.anchor_index in selected:
            cold_start.append(step.batch.mixtures.global_certificate is None)
            local_fallback.append(step.batch.mixtures.fallback_reason)
            laws = tuple(
                step.batch.get(name).distribution for name in ("contextual_stack", "equal", "global_stack")
            )
            scores.append(
                (
                    _score(laws[0], record, inputs.transform),
                    _score(laws[1], record, inputs.transform),
                    _score(laws[2], record, inputs.transform),
                )
            )
            origins.append(record.record_id)
            identities.append((laws[0].record_id, laws[1].record_id, laws[2].record_id))
    if len(scores) != len(selected):
        raise IntegrityError("pilot did not produce every frozen block score")
    values = np.asarray(scores, dtype=np.float64)
    result = assess_precision(
        values[:, 0],
        values[:, 1],
        values[:, 2],
        len(inputs.projected_a.origin_indices),
        len(inputs.projected_b.origin_indices),
    )
    return PrecisionScores(
        inputs.record_id,
        tuple(origins),
        tuple(identities),
        values[:, 0],
        values[:, 1],
        values[:, 2],
        len(records),
        state.record_id,
        result,
        inputs.evidence_scope,
        tuple(pilot_indices(len(selected)).items()),
        tuple(cold_start),
        tuple(local_fallback),
    )
