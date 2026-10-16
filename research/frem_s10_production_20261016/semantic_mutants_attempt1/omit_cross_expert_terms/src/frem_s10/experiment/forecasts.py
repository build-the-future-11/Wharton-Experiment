"""Causal weekly kernel orchestration and immutable shared evaluation laws."""

from collections.abc import Callable
from dataclasses import dataclass

from frem_s10.domain import ForecastDistribution, OriginRecord, Transform
from frem_s10.errors import ChronologyError, ContractError, DataAdmissionError, ResourceLimitError
from frem_s10.identities import scientific_digest
from frem_s10.metadata import strict_int
from frem_s10.model.config import ForecastConfig
from frem_s10.model.engine import advance, initial_state
from frem_s10.model.records import ForecastState, ForecastTransition, IssueRequest, scientific_id

from .matrix import FORECAST_ARMS, MODEL_VARIANTS

PREQUENTIAL_START_US = 1199145600000000  # 2008-01-01 00:00 UTC, frozen design cutoff.


@dataclass(frozen=True, slots=True)
class ForecastFrame:
    """Exact laws issued at one origin by one model configuration."""

    anchor_index: int
    origin_id: str
    decision_us: int
    variant: str
    laws: tuple[tuple[str, ForecastDistribution], ...]
    state_before_id: str
    state_after_id: str

    def __post_init__(self) -> None:
        """Validate complete arm ordering and immutable record membership."""
        strict_int(self.anchor_index, name="frame anchor", minimum=0)
        strict_int(self.decision_us, name="frame decision")
        if self.variant not in MODEL_VARIANTS:
            raise ContractError("unknown forecast frame variant")
        expected = FORECAST_ARMS if self.variant == "primary" else ("contextual_stack",)
        if not isinstance(self.laws, tuple) or tuple(name for name, _ in self.laws) != expected:
            raise ContractError("forecast frame does not contain its exact ordered arms")
        if any(not isinstance(law, ForecastDistribution) for _, law in self.laws):
            raise ContractError("forecast frame requires immutable finite laws")
        if any(
            not isinstance(value, str) or not value
            for value in (self.origin_id, self.state_before_id, self.state_after_id)
        ):
            raise ContractError("forecast frame provenance identities required")

    def get(self, arm: str) -> ForecastDistribution:
        """Return the original saved law; do not recompute from current state."""
        for name, law in self.laws:
            if name == arm:
                return law
        raise ContractError("arm absent from forecast frame")

    @property
    def record_id(self) -> str:
        """Hash each validated binary64 law before the small ordered frame envelope."""
        return scientific_digest(
            {
                "schema": "s10-forecast-frame-v2",
                "anchor": self.anchor_index,
                "origin": self.origin_id,
                "decision": self.decision_us,
                "variant": self.variant,
                "before": self.state_before_id,
                "after": self.state_after_id,
                "laws": [
                    [
                        name,
                        scientific_id(
                            "s10-law-component-v1",
                            (law.record_id, "g:<f8", "d:<f8", "p:<f8"),
                            law.g,
                            law.d,
                            law.probabilities,
                        ),
                    ]
                    for name, law in self.laws
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class ForecastTape:
    """Retained decision laws shared across policies, costs and unchanged ablations."""

    frames: tuple[ForecastFrame, ...]
    transform_id: str
    evidence_scope: str
    weekly_origins: int
    final_state_ids: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        """Require unique ordered origins and all six independent causal model states."""
        strict_int(self.weekly_origins, name="weekly origins", minimum=1)
        if self.weekly_origins > 1000:
            raise ResourceLimitError("weekly issue budget exceeds 1000 origins per arm")
        if self.evidence_scope not in ("KERNEL_REPLAY", "GENERATED_ENGINEERING"):
            raise ContractError("explicit forecast evidence scope required")
        if not isinstance(self.frames, tuple) or not self.frames or not self.transform_id:
            raise ContractError("nonempty immutable forecast tape required")
        keys = tuple((frame.anchor_index, MODEL_VARIANTS.index(frame.variant)) for frame in self.frames)
        if keys != tuple(sorted(set(keys))):
            raise ChronologyError("forecast tape keys are duplicate or out of order")
        for index in dict.fromkeys(frame.anchor_index for frame in self.frames):
            selected = tuple(frame for frame in self.frames if frame.anchor_index == index)
            if tuple(frame.variant for frame in selected) != MODEL_VARIANTS:
                raise ContractError("every retained origin needs all six forecast model variants")
            if len({(frame.origin_id, frame.decision_us) for frame in selected}) != 1:
                raise ChronologyError("model variants disagree about their common origin")
        if tuple(name for name, _ in self.final_state_ids) != MODEL_VARIANTS:
            raise ContractError("final model state registry is incomplete")
        if not isinstance(self.final_state_ids, tuple) or any(
            not isinstance(identity, str) or not identity for _, identity in self.final_state_ids
        ):
            raise ContractError("final model state identities must be immutable nonempty strings")

    @property
    def record_id(self) -> str:
        """Hash numerical laws as well as their provenance; exclude filesystem state."""
        return scientific_digest(
            {
                "schema": "s10-forecast-tape-v2",
                "transform": self.transform_id,
                "scope": self.evidence_scope,
                "weekly_origins": self.weekly_origins,
                "final_states": [[name, identity] for name, identity in self.final_state_ids],
                "frames": [frame.record_id for frame in self.frames],
            }
        )

    def frame(self, anchor_index: int, variant: str = "primary") -> ForecastFrame:
        """Resolve one immutable frame or fail on incomplete forecast evidence."""
        for frame in self.frames:
            if frame.anchor_index == anchor_index and frame.variant == variant:
                return frame
        raise ContractError("required forecast frame absent")


def _requests(records: tuple[OriginRecord, ...], retained: tuple[int, ...]) -> tuple[OriginRecord, ...]:
    if not records or not retained or retained != tuple(sorted(set(retained))):
        raise ContractError("ordered nonempty records and retained anchors required")
    indices = tuple(record.anchor_index for record in records)
    if indices != tuple(sorted(set(indices))) or len({record.record_id for record in records}) != len(
        records
    ):
        raise ChronologyError("origin records have duplicate or unordered identities")
    if not set(retained).issubset(indices):
        raise ContractError("a requested evaluation anchor has no origin record")
    selected = tuple(
        record
        for record in records
        if record.origin_us >= PREQUENTIAL_START_US
        and record.anchor_index <= retained[-1]
        and sum(item.available_us < record.origin_us + 1 for item in records) >= 64
    )
    if not selected or not set(retained).issubset(item.anchor_index for item in selected):
        raise DataAdmissionError("evaluation origins lack prequential history")
    if len(selected) > 1000:
        raise ResourceLimitError("weekly issue budget exceeds 1000 origins per arm")
    return selected


def _advance_variant(
    state: ForecastState,
    record: OriginRecord,
    records: tuple[OriginRecord, ...],
    variant: str,
    transition_sink: Callable[[str, ForecastTransition], None] | None = None,
) -> tuple[ForecastState, ForecastFrame]:
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
        for item in records
        if item.available_us < request.decision_us and item.record_id not in state.consumed_event_ids
    )
    step = advance(state, request, matured)
    if transition_sink is not None:
        transition_sink(variant, step)
    arms = FORECAST_ARMS if variant == "primary" else ("contextual_stack",)
    frame = ForecastFrame(
        record.anchor_index,
        record.record_id,
        request.decision_us,
        variant,
        tuple((arm, step.batch.get(arm).distribution) for arm in arms),
        step.batch.get("contextual_stack").state_before_id,
        step.state.record_id,
    )
    return step.state, frame


def build_forecast_tape(
    records: tuple[OriginRecord, ...],
    transform: Transform,
    *,
    retained_anchor_indices: tuple[int, ...],
    transition_sink: Callable[[str, ForecastTransition], None] | None = None,
) -> ForecastTape:
    """Issue every eligible weekly origin from 2008, retaining only needed laws.

    Models continue across the A/B boundary. Each forecast-changing ablation
    owns its delayed feedback. No holdings, policy, cost or target for the
    current issue is passed to the kernel. Source/resource admission belongs
    to the caller; this pure function performs no IO or process launch.
    """
    selected = _requests(records, retained_anchor_indices)
    states = {name: initial_state(transform, ForecastConfig().variant(name)) for name in MODEL_VARIANTS}
    frames: list[ForecastFrame] = []
    for record in selected:
        for name in MODEL_VARIANTS:
            states[name], frame = _advance_variant(states[name], record, records, name, transition_sink)
            if record.anchor_index in retained_anchor_indices:
                frames.append(frame)
    return ForecastTape(
        tuple(frames),
        transform.record_id,
        "KERNEL_REPLAY",
        len(selected),
        tuple((name, states[name].record_id) for name in MODEL_VARIANTS),
    )
