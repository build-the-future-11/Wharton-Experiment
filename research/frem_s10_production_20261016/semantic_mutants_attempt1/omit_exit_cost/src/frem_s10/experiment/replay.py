"""Pure provided-data campaign replay, separated from admission and supervision."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import numpy as np

from frem_s10.arrays import FloatArray, frozen_f64
from frem_s10.data.calendar import CalendarSchedule, build_calendar_schedule, plan_fold
from frem_s10.data.features import build_origin_records, fit_burnin_transform
from frem_s10.domain import ForecastDistribution, OriginRecord, SessionPanel, Transform
from frem_s10.errors import ChronologyError, ContractError, IntegrityError, S10Error
from frem_s10.math.scores import (
    discrete_quantiles,
    energy_score,
    loss_probabilities,
    marginal_crps,
    mean_squared_error,
    variogram_score,
)
from frem_s10.metadata import from_utc_us, scheduled_open_us, strict_int
from frem_s10.portfolio.actions import Valuation
from frem_s10.portfolio.benchmarks import select_benchmark
from frem_s10.portfolio.ledger import (
    LedgerEvent,
    LedgerState,
    LedgerTransition,
    NavMark,
    boundary_log_increments,
    close_session,
    forced_liquidation_open,
    initial_capital_mark,
    initial_ledger,
    liquidatable_boundary_mark,
    maximum_daily_close_drawdown,
    open_session,
    rebalance,
)
from frem_s10.portfolio.policy import PolicyDecision, select_policy

from .forecasts import ForecastTape
from .matrix import CellSpec, campaign_matrix

type DividendGroups = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class SessionTiming:
    """Caller-admitted close timestamps and exact asset-aligned source action IDs."""

    open_us: int
    close_us: int
    dividend_event_ids: DividendGroups

    def __post_init__(self) -> None:
        """Do not invent a close timestamp or compress source-event identities."""
        strict_int(self.open_us, name="session open")
        strict_int(self.close_us, name="session close")
        if (
            self.close_us <= self.open_us + 2
            or from_utc_us(self.open_us).date() != from_utc_us(self.close_us).date()
        ):
            raise ChronologyError("close must follow same-session opening phases")
        if not isinstance(self.dividend_event_ids, tuple) or len(self.dividend_event_ids) != 3:
            raise ContractError("asset-aligned source dividend identity groups required")
        if any(not isinstance(group, tuple) for group in self.dividend_event_ids):
            raise ContractError("immutable source dividend identity groups required")
        flat = tuple(item for group in self.dividend_event_ids for item in group)
        if any(not isinstance(item, str) or not item for item in flat) or len(flat) != len(set(flat)):
            raise ContractError("source dividend identities must be nonempty and unique")


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    """One contiguous i0+4k ledger window, with its own initial capital."""

    fold_id: str
    origin_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Require exact four-anchor progression, without outcome-based skips."""
        if (
            self.fold_id not in ("A", "B")
            or not isinstance(self.origin_indices, tuple)
            or not self.origin_indices
        ):
            raise ContractError("a declared nonempty immutable fold window is required")
        for index in self.origin_indices:
            strict_int(index, name="window anchor", minimum=0)
        if self.origin_indices != tuple(range(self.origin_indices[0], self.origin_indices[-1] + 1, 4)):
            raise ChronologyError("replay window must progress exactly four weekly anchors")


@dataclass(frozen=True, slots=True)
class ReplayInputs:
    """Provided immutable inputs; generated fixtures remain explicitly engineering evidence."""

    panel: SessionPanel
    calendar: CalendarSchedule
    timing: tuple[SessionTiming, ...]
    origins: tuple[OriginRecord, ...]
    transform: Transform
    windows: tuple[ReplayWindow, ...]
    evidence_scope: str

    def __post_init__(self) -> None:
        """Reject contradictory calendars, unlinked sources and incomplete block metadata."""
        if any(not isinstance(value, tuple) for value in (self.timing, self.origins, self.windows)):
            raise ContractError("replay timing, records and windows must be immutable tuples")
        if self.evidence_scope not in ("KERNEL_REPLAY", "GENERATED_ENGINEERING"):
            raise ContractError("explicit replay evidence scope required")
        if self.calendar != build_calendar_schedule(self.panel.dates) or len(self.timing) != len(
            self.panel.dates
        ):
            raise IntegrityError("replay calendar or session timing differs from provided panel")
        if tuple(window.fold_id for window in self.windows) != ("A", "B"):
            raise ContractError("both ordered fold windows required")
        self._validate_sources()
        self._validate_origins()
        if self.evidence_scope == "KERNEL_REPLAY":
            expected = tuple(
                ReplayWindow(fold, plan_fold(self.calendar, fold).origin_indices) for fold in ("A", "B")
            )
            if self.windows != expected:
                raise ContractError("kernel campaign requires the complete frozen fold plans")
            self._validate_derived_payloads()

    def _validate_derived_payloads(self) -> None:
        expected = build_origin_records(self.panel, calendar=self.calendar)
        if len(expected) != len(self.origins) or any(
            not _same_origin(left, right) for left, right in zip(expected, self.origins, strict=True)
        ):
            raise IntegrityError(
                "kernel origin metadata or numeric payload differs from admitted panel construction"
            )
        transform = fit_burnin_transform(expected)
        if (transform.record_id, transform.record_ids) != (
            self.transform.record_id,
            self.transform.record_ids,
        ) or any(
            not np.array_equal(getattr(transform, name), getattr(self.transform, name))
            for name in ("feature_mean", "feature_scale", "score_scale")
        ):
            raise IntegrityError("kernel transform differs from exact admitted burn-in reconstruction")

    def _validate_sources(self) -> None:
        if self.panel.dates[-1] >= date(2017, 1, 1):
            raise ContractError("S10 replay excludes all source observations from 2017 onward")
        aligned = self.panel.require_asset_event_ids()
        for index, (day, timing) in enumerate(zip(self.panel.dates, self.timing, strict=True)):
            if timing.open_us != scheduled_open_us(day):
                raise ChronologyError("session quote differs from CL-01's exact opening clock")
            flat = tuple(item for group in timing.dividend_event_ids for item in group)
            if flat != self.panel.event_ids[index]:
                raise IntegrityError("aligned source actions differ from retained panel event IDs")
            expected = tuple(() if item is None else (item,) for item in aligned[index])
            if timing.dividend_event_ids != expected:
                raise IntegrityError("source dividend ownership differs from admitted asset alignment")
            if any(
                value > 0 and not ids
                for value, ids in zip(self.panel.dividends[index], timing.dividend_event_ids, strict=True)
            ):
                raise IntegrityError("nonzero cash dividend lacks source-event provenance")

    def _validate_origins(self) -> None:
        by_index = {record.anchor_index: record for record in self.origins}
        if len(by_index) != len(self.origins) or len({record.record_id for record in self.origins}) != len(
            self.origins
        ):
            raise IntegrityError("duplicate replay origin index or stable identity")
        for window in self.windows:
            for index in window.origin_indices:
                if index not in by_index or index + 4 >= len(self.calendar.anchors):
                    raise ContractError("replay block lacks its complete origin or endpoint")
                item = by_index[index]
                if (item.origin_us, item.endpoint_us) != (
                    self.calendar.anchors[index].quote_us,
                    self.calendar.anchors[index + 4].quote_us,
                ):
                    raise ChronologyError("replay target metadata differs from exact H4 anchor clock")

    def window(self, fold_id: str) -> ReplayWindow:
        """Resolve a fold without merging or carrying its portfolio state."""
        for window in self.windows:
            if window.fold_id == fold_id:
                return window
        raise ContractError("unknown replay fold")


def _same_origin(left: OriginRecord, right: OriginRecord) -> bool:
    metadata = (
        "record_id",
        "anchor_index",
        "origin_us",
        "endpoint_us",
        "available_us",
        "feature_available_us",
    )
    if any(getattr(left, name) != getattr(right, name) for name in metadata):
        return False
    for name in ("x", "g", "d", "h1_g", "h1_d"):
        first, second = getattr(left, name), getattr(right, name)
        if (first is None) != (second is None) or (first is not None and not np.array_equal(first, second)):
            return False
    return True


def prepare_campaign(panel: SessionPanel, timing: tuple[SessionTiming, ...]) -> ReplayInputs:
    """Build exact data-derived records and full frozen A/B windows from admitted inputs.

    This function does not grant source rights, protected access, resource
    admission or scientific execution permission. Those remain caller gates.
    """
    calendar = build_calendar_schedule(panel.dates)
    origins = build_origin_records(panel, calendar=calendar)
    transform = fit_burnin_transform(origins)
    windows = tuple(ReplayWindow(fold, plan_fold(calendar, fold).origin_indices) for fold in ("A", "B"))
    return ReplayInputs(panel, calendar, timing, origins, transform, windows, "KERNEL_REPLAY")


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """An executed selection tied to the original forecast and post-opening state."""

    anchor_index: int
    origin_id: str
    forecast_id: str
    forecast_decision_us: int
    decision_us: int
    valuation_state_id: str
    selection: PolicyDecision
    fill: LedgerEvent


@dataclass(frozen=True, slots=True)
class ForecastScore:
    """Dated out-of-sample diagnostics computed from the law actually used."""

    origin_id: str
    forecast_id: str
    anchor_index: int
    energy: float
    variogram: float
    marginal_crps: FloatArray
    mean_squared_error: float
    loss_probabilities: FloatArray
    loss_outcomes: FloatArray
    raw_marginal_crps: FloatArray
    raw_mean_squared_errors: FloatArray
    interval_lower_raw: FloatArray
    interval_upper_raw: FloatArray
    interval_coverage: FloatArray
    interval_outcome_boundary_ties: FloatArray

    def __post_init__(self) -> None:
        """Keep score arrays immutable when shared between reports and controls."""
        for name, length in (
            ("marginal_crps", 3),
            ("loss_probabilities", 4),
            ("loss_outcomes", 4),
            ("raw_marginal_crps", 3),
            ("raw_mean_squared_errors", 3),
        ):
            object.__setattr__(self, name, frozen_f64(getattr(self, name), shape=(length,), name=name))
        for name in (
            "interval_lower_raw",
            "interval_upper_raw",
            "interval_coverage",
            "interval_outcome_boundary_ties",
        ):
            object.__setattr__(self, name, frozen_f64(getattr(self, name), shape=(3, 3), name=name))


type ScoreCache = dict[tuple[int, int], ForecastScore]


@dataclass(frozen=True, slots=True)
class CellResult:
    """Retained cell events and marks; failure prefixes explicitly retain a nonfinal phase."""

    spec: CellSpec
    final_state: LedgerState
    events: tuple[LedgerEvent, ...]
    decisions: tuple[DecisionRecord, ...]
    scores: tuple[ForecastScore, ...]
    daily_marks: tuple[NavMark, ...]
    boundary_marks: tuple[NavMark, ...]

    @property
    def log_increments(self) -> FloatArray:
        """Return one telescoping liquidatable increment per complete H4 block."""
        return boundary_log_increments(self.boundary_marks)

    @property
    def maximum_drawdown(self) -> float:
        """Use only actual daily-close marks and separately labeled end anchors."""
        return maximum_daily_close_drawdown(self.daily_marks)


@dataclass(frozen=True, slots=True)
class CellFailure:
    """A retained failure identity, without manufacturing a successful result."""

    cell_id: str
    code: str
    message: str
    partial: CellResult | None = None


class CellReplayError(S10Error):
    """Retain every completed event in a failed cell without calling it terminal."""

    def __init__(self, cause: S10Error | ArithmeticError | ValueError, partial: CellResult) -> None:
        """Own the immutable failure prefix and its original diagnostic code."""
        super().__init__(str(cause), record_id=partial.spec.cell_id)
        self.code = cause.code if isinstance(cause, S10Error) else type(cause).__name__
        self.partial = partial


@dataclass(frozen=True, slots=True)
class CampaignResult:
    """An ordered complete or failed-prefix matrix; no failed cell is omitted silently."""

    results: tuple[CellResult, ...]
    failure: CellFailure | None
    planned_cell_ids: tuple[str, ...]
    forecast_tape_id: str
    evidence_scope: str

    @property
    def complete(self) -> bool:
        """Require the exact full matrix, in order, and no recorded failure."""
        return (
            self.failure is None
            and self.planned_cell_ids == tuple(cell.cell_id for cell in campaign_matrix())
            and tuple(item.spec.cell_id for item in self.results) == self.planned_cell_ids
            and all(item.final_state.phase == "FINAL" for item in self.results)
        )


def _law(
    tape: ForecastTape, cell: CellSpec, window: ReplayWindow, index: int
) -> tuple[ForecastDistribution, int]:
    selected_index = (
        window.origin_indices[0] if cell.kind == "ablation" and cell.arm == "frozen_state" else index
    )
    frame = tape.frame(selected_index, cell.variant)
    return frame.get(cell.forecast_arm), frame.decision_us


def _selection(
    cell: CellSpec, state: LedgerState, law: ForecastDistribution, initial: bool
) -> PolicyDecision:
    valuation = Valuation(state.risky_dollars, state.cash)
    if cell.kind == "benchmark":
        return select_benchmark(
            cell.arm, valuation, law, one_way_cost=cell.one_way_cost, initial_entry=initial
        )
    return select_policy(
        valuation,
        law,
        one_way_cost=cell.one_way_cost,
        robust=cell.policy == "robust",
        include_hold=not (cell.kind == "ablation" and cell.arm == "no_exact_hold"),
    )


def _interval_diagnostics(
    law: ForecastDistribution, outcome: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    lower, upper = np.empty((3, 3)), np.empty((3, 3))
    for asset in range(3):
        quantiles = discrete_quantiles(
            law.r[:, asset], law.probabilities, np.array([0.25, 0.75, 0.10, 0.90, 0.025, 0.975])
        )
        lower[:, asset], upper[:, asset] = quantiles[::2], quantiles[1::2]
    coverage = ((lower <= outcome) & (outcome <= upper)).astype(np.float64)
    ties = ((lower == outcome) | (upper == outcome)).astype(np.float64)
    return lower, upper, coverage, ties


def _score(record: OriginRecord, law: ForecastDistribution, transform: Transform) -> ForecastScore:
    atoms, outcome = transform.scale_return(law.r), transform.scale_return(record.r)
    losses = np.concatenate(((record.r < 0).astype(np.float64), [float(record.r.mean() < 0)]))
    return ForecastScore(
        record.record_id,
        law.record_id,
        record.anchor_index,
        energy_score(atoms, law.probabilities, outcome),
        variogram_score(atoms, law.probabilities, outcome),
        marginal_crps(atoms, law.probabilities, outcome),
        mean_squared_error(atoms, law.probabilities, outcome),
        loss_probabilities(law.r, law.probabilities),
        losses,
        marginal_crps(law.r, law.probabilities, record.r),
        (law.probabilities @ law.r - record.r) ** 2,
        *_interval_diagnostics(law, record.r),
    )


def _entry(
    state: LedgerState,
    cell: CellSpec,
    index: int,
    record: OriginRecord,
    law: ForecastDistribution,
    issued_us: int,
    initial: bool,
) -> tuple[LedgerState, tuple[LedgerEvent, ...], DecisionRecord]:
    selection = _selection(cell, state, law, initial)
    fill = rebalance(
        state,
        event_id=f"{cell.cell_id}/{index}/fill",
        decision_us=record.origin_us + 1,
        fill_us=record.origin_us + 2,
        target_weights=selection.action.weights,
        one_way_cost=cell.one_way_cost,
        is_hold=selection.action.is_hold,
    )
    decision = DecisionRecord(
        index,
        record.record_id,
        law.record_id,
        issued_us,
        record.origin_us + 1,
        state.record_id,
        selection,
        fill.event,
    )
    return fill.state, (fill.event,), decision


def _require_mark(transition: LedgerTransition) -> NavMark:
    if transition.mark is None:
        raise IntegrityError("mark-producing transition returned no valuation mark")
    return transition.mark


class _CellBuilder:
    def __init__(self, spec: CellSpec, state: LedgerState) -> None:
        self.spec = spec
        self.state = state
        initial = initial_capital_mark(state)
        self.daily = [initial]
        self.boundaries = [initial]
        self.events: list[LedgerEvent] = []
        self.decisions: list[DecisionRecord] = []
        self.scores: list[ForecastScore] = []

    def record(self, transition: LedgerTransition) -> None:
        self.state = transition.state
        self.events.append(transition.event)

    def snapshot(self) -> CellResult:
        return CellResult(
            self.spec,
            self.state,
            tuple(self.events),
            tuple(self.decisions),
            tuple(self.scores),
            tuple(self.daily),
            tuple(self.boundaries),
        )


def _record_decision(
    inputs: ReplayInputs,
    tape: ForecastTape,
    window: ReplayWindow,
    builder: _CellBuilder,
    record: OriginRecord,
    score_cache: ScoreCache,
) -> None:
    cell, index = builder.spec, record.anchor_index
    if builder.decisions:
        builder.boundaries.append(liquidatable_boundary_mark(builder.state, one_way_cost=cell.one_way_cost))
    if cell.kind == "ablation" and cell.arm == "force_liquidate":
        builder.record(
            forced_liquidation_open(
                builder.state, event_id=f"{cell.cell_id}/{index}/forced-sale", one_way_cost=cell.one_way_cost
            )
        )
    law, issued_us = _law(tape, cell, window, index)
    state, events, decision = _entry(
        builder.state, cell, index, record, law, issued_us, not builder.decisions
    )
    builder.state = state
    builder.events.extend(events)
    builder.decisions.append(decision)
    key = (id(law), record.anchor_index)
    if key not in score_cache:
        score_cache[key] = _score(record, law, inputs.transform)
    builder.scores.append(score_cache[key])


def _replay_sessions(
    inputs: ReplayInputs,
    tape: ForecastTape,
    window: ReplayWindow,
    builder: _CellBuilder,
    score_cache: ScoreCache,
) -> None:
    cell, anchors = builder.spec, inputs.calendar.anchors
    first, final = anchors[window.origin_indices[0]], anchors[window.origin_indices[-1] + 4]
    origins = {item.anchor_index: item for item in inputs.origins}
    decision_sessions = {anchors[index].session_index: index for index in window.origin_indices}
    for session in range(first.session_index, final.session_index + 1):
        timing = inputs.timing[session]
        opened = open_session(
            builder.state,
            event_id=f"{cell.cell_id}/{session}/open",
            timestamp_us=timing.open_us,
            prices=inputs.panel.open[session],
            dividends_per_share=inputs.panel.dividends[session],
            dividend_event_ids=timing.dividend_event_ids,
        )
        builder.record(opened)
        if session == final.session_index:
            exited = rebalance(
                builder.state,
                event_id=f"{cell.cell_id}/{session}/final",
                decision_us=timing.open_us + 1,
                fill_us=timing.open_us + 2,
                target_weights=np.zeros(3),
                one_way_cost=cell.one_way_cost,
                final_liquidation=True,
            )
            builder.record(exited)
            builder.daily.append(_require_mark(exited))
            builder.boundaries.append(_require_mark(exited))
            break
        if session in decision_sessions:
            _record_decision(inputs, tape, window, builder, origins[decision_sessions[session]], score_cache)
        closed = close_session(
            builder.state,
            event_id=f"{cell.cell_id}/{session}/close",
            timestamp_us=timing.close_us,
            prices=inputs.panel.close[session],
        )
        builder.record(closed)
        builder.daily.append(_require_mark(closed))


def replay_cell(
    inputs: ReplayInputs, tape: ForecastTape, cell: CellSpec, *, _score_cache: ScoreCache | None = None
) -> CellResult:
    """Replay one complete window, preserving its event prefix on any numerical failure."""
    _validate_tape(inputs, tape)
    window = inputs.window(cell.fold_id)
    first = inputs.calendar.anchors[window.origin_indices[0]]
    state = initial_ledger(
        cell_id=cell.cell_id,
        start_us=first.quote_us - 1,
        forced_liquidation_enabled=cell.kind == "ablation" and cell.arm == "force_liquidate",
    )
    builder = _CellBuilder(cell, state)
    try:
        _replay_sessions(inputs, tape, window, builder, {} if _score_cache is None else _score_cache)
    except (S10Error, ArithmeticError, ValueError) as exc:
        raise CellReplayError(exc, builder.snapshot()) from exc
    return builder.snapshot()


def _validate_tape(inputs: ReplayInputs, tape: ForecastTape) -> None:
    if tape.transform_id != inputs.transform.record_id or tape.evidence_scope != inputs.evidence_scope:
        raise IntegrityError("forecast tape transform or evidence scope mismatch")
    indices = tuple(sorted(index for window in inputs.windows for index in window.origin_indices))
    if tuple(dict.fromkeys(frame.anchor_index for frame in tape.frames)) != indices:
        raise IntegrityError("forecast tape differs from the exact retained evaluation origins")
    records = {record.anchor_index: record for record in inputs.origins}
    for frame in tape.frames:
        if (frame.origin_id, frame.decision_us) != (
            records[frame.anchor_index].record_id,
            records[frame.anchor_index].origin_us + 1,
        ):
            raise ChronologyError("forecast frame metadata does not match its evaluation origin")


def campaign_steps(inputs: ReplayInputs, tape: ForecastTape) -> Iterator[CellResult | CellFailure]:
    """Yield each completed ledger for durable publication before running the next."""
    _validate_tape(inputs, tape)
    score_cache: ScoreCache = {}
    for cell in campaign_matrix():
        try:
            yield replay_cell(inputs, tape, cell, _score_cache=score_cache)
        except (S10Error, ArithmeticError, ValueError) as exc:
            yield CellFailure(
                cell.cell_id,
                exc.code if isinstance(exc, S10Error) else type(exc).__name__,
                str(exc),
                exc.partial if isinstance(exc, CellReplayError) else None,
            )
            return


def replay_campaign(inputs: ReplayInputs, tape: ForecastTape) -> CampaignResult:
    """Execute exactly 192 ledgers or return the complete successful prefix and failure.

    This pure engine does not read source files, persist artifacts, resume a
    consumed attempt, inspect protected outcomes or launch workers. A runtime
    supervisor must publish each returned result and the failure receipt.
    """
    matrix = campaign_matrix()
    completed: list[CellResult] = []
    failure: CellFailure | None = None
    for step in campaign_steps(inputs, tape):
        if isinstance(step, CellFailure):
            failure = step
            break
        completed.append(step)
    return CampaignResult(
        tuple(completed),
        failure,
        tuple(cell.cell_id for cell in matrix),
        tape.record_id,
        inputs.evidence_scope,
    )
