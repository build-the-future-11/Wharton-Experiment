"""Pure share/cash accounting under the declared ideal opening-fill contract.

Opening quotes and ex-date cash credits are one atomic event. Its dividend
entitlement uses entering shares; a trade can follow only at open plus two
microseconds. Hypothetical liquidation marks never change ledger cash.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise

import numpy as np

from frem_s10.arrays import FloatArray, frozen_f64
from frem_s10.errors import ChronologyError, ContractError, NumericalError
from frem_s10.identities import JSONValue, scientific_digest
from frem_s10.math.fees import keep_holdings_solution, solve_rebalance, validate_cost
from frem_s10.metadata import finite_float, from_utc_us, strict_int


class MarkKind(StrEnum):
    """Keep actual NAV observations distinct from hypothetical liquidation values."""

    INITIAL_CAPITAL = "INITIAL_CAPITAL"
    DAILY_CLOSE = "DAILY_CLOSE"
    LIQUIDATABLE_BOUNDARY = "LIQUIDATABLE_BOUNDARY"
    FINAL_LIQUIDATION = "FINAL_LIQUIDATION"


@dataclass(frozen=True, slots=True)
class NavMark:
    """A dated valuation with an explicit actual or hypothetical convention."""

    timestamp_us: int
    kind: MarkKind
    value: float
    state_id: str

    def __post_init__(self) -> None:
        """Reject ambiguous time, mark type or nonpositive wealth."""
        strict_int(self.timestamp_us, name="mark_timestamp_us")
        finite_float(self.value, name="mark_value", strictly_positive=True)
        if not isinstance(self.kind, MarkKind) or not isinstance(self.state_id, str) or not self.state_id:
            raise ContractError("declared mark kind and source state identity required")


@dataclass(frozen=True, slots=True)
class LedgerState:
    """Immutable holdings and cumulative attribution, independent of forecasts."""

    cell_id: str
    initial_cash: float
    start_us: int
    shares: FloatArray
    cash: float
    marks: FloatArray | None
    price_pnl: FloatArray
    dividends: FloatArray
    fees: float
    turnover: float
    cumulative_residual: float
    cursor_us: int
    open_us: int | None
    phase: str
    consumed_event_ids: frozenset[str]
    forced_liquidation_enabled: bool = False
    forced_sale_us: int | None = None

    def __post_init__(self) -> None:
        """Own immutable arrays and validate state/accounting invariants."""
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ContractError("ledger cell identity required")
        _verify_clock(self)
        for name in ("shares", "price_pnl", "dividends"):
            object.__setattr__(self, name, frozen_f64(getattr(self, name), shape=(3,), name=name))
        if np.any(self.shares < 0) or np.any(self.dividends < 0):
            raise ContractError("shares and cumulative cash dividends must be nonnegative")
        if self.marks is not None:
            object.__setattr__(self, "marks", frozen_f64(self.marks, shape=(3,), name="marks"))
            if np.any(self.marks <= 0):
                raise ContractError("marks must be strictly positive")
        elif np.any(self.shares != 0):
            raise ContractError("nonzero shares require visible marks")
        finite_float(self.initial_cash, name="initial_cash", strictly_positive=True)
        for name in ("cash", "fees", "turnover"):
            finite_float(getattr(self, name), name=name, minimum=0)
        finite_float(self.cumulative_residual, name="cumulative_residual")
        _verify_phase(self)
        _verify_cumulative_accounting(self)

    @property
    def risky_dollars(self) -> FloatArray:
        """Value actual shares at the latest visible marks."""
        values = np.zeros(3) if self.marks is None else self.shares * self.marks
        return frozen_f64(values, shape=(3,), name="risky_dollars")

    @property
    def nav(self) -> float:
        """Return actual marked wealth without hypothetical exit deductions."""
        return finite_float(self.cash + float(self.risky_dollars.sum()), name="NAV", strictly_positive=True)

    @property
    def record_id(self) -> str:
        """Hash all scientific state, excluding machine paths and wall-clock timing."""
        event_ids = list[JSONValue](sorted(self.consumed_event_ids))
        return scientific_digest(
            {
                "schema": "s10-ledger-v2-cl15",
                "cell_id": self.cell_id,
                "initial_cash": self.initial_cash,
                "start_us": self.start_us,
                "shares": self.shares.tolist(),
                "cash": self.cash,
                "marks": None if self.marks is None else self.marks.tolist(),
                "price_pnl": self.price_pnl.tolist(),
                "dividends": self.dividends.tolist(),
                "fees": self.fees,
                "turnover": self.turnover,
                "cumulative_residual": self.cumulative_residual,
                "cursor_us": self.cursor_us,
                "open_us": self.open_us,
                "phase": self.phase,
                "consumed_event_ids": event_ids,
                "forced_liquidation_enabled": self.forced_liquidation_enabled,
                "forced_sale_us": self.forced_sale_us,
            }
        )

    @property
    def cursor(self) -> tuple[int, int]:
        """Return ordered event time/subphase, including CL-15's opening sale."""
        return self.cursor_us, int(self.forced_sale_us == self.cursor_us)


def _verify_clock(state: LedgerState) -> None:
    for name in ("start_us", "cursor_us"):
        strict_int(getattr(state, name), name=name)
    if state.cursor_us < state.start_us:
        raise ChronologyError("ledger cursor precedes initial capital")
    if state.open_us is not None:
        strict_int(state.open_us, name="open_us")
        if not state.start_us < state.open_us <= state.cursor_us:
            raise ChronologyError("ledger opening cursor is inconsistent")
    if state.forced_sale_us is not None:
        strict_int(state.forced_sale_us, name="forced_sale_us")
        if not state.forced_liquidation_enabled or state.forced_sale_us != state.open_us:
            raise ChronologyError("forced-sale subphase must belong to the enabled current opening")


def _verify_phase(state: LedgerState) -> None:
    if not isinstance(state.forced_liquidation_enabled, bool):
        raise ContractError("forced_liquidation_enabled must be boolean")
    if state.phase not in ("INITIAL", "OPEN", "FILLED", "CLOSE", "FINAL"):
        raise ContractError("unsupported ledger phase")
    if not isinstance(state.consumed_event_ids, frozenset) or any(
        not isinstance(item, str) or not item for item in state.consumed_event_ids
    ):
        raise ContractError("immutable nonempty event identities required")
    if state.phase == "FINAL" and np.any(state.shares != 0):
        raise ContractError("final liquidation cannot retain risky shares")
    if state.phase == "OPEN" and state.forced_sale_us is not None and np.any(state.shares != 0):
        raise ContractError("a completed forced-sale opening must hold zero shares")
    if state.phase == "INITIAL" and (
        state.cursor_us != state.start_us
        or state.open_us is not None
        or state.marks is not None
        or state.consumed_event_ids
        or state.cash != state.initial_cash
    ):
        raise ContractError("initial state must precede every market event")


def _verify_cumulative_accounting(state: LedgerState) -> None:
    explained = float(state.price_pnl.sum() + state.dividends.sum()) - state.fees
    difference = state.nav - state.initial_cash - explained - state.cumulative_residual
    if abs(difference) > 1e-10 * max(1.0, state.initial_cash, state.nav):
        raise NumericalError("cumulative ledger attribution does not reconcile")


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """Retain an event's dollar conservation certificate and exact parent identities."""

    event_id: str
    timestamp_us: int
    kind: str
    before_id: str
    after_id: str
    price_pnl: FloatArray
    dividends: FloatArray
    fee: float
    turnover: float
    nav_before: float
    nav_after: float
    residual: float
    shares_before: FloatArray
    shares_after: FloatArray
    cash_before: float
    cash_after: float
    phase_order: int = 0

    def __post_init__(self) -> None:
        """Own immutable event attribution arrays."""
        for name in ("price_pnl", "dividends", "shares_before", "shares_after"):
            object.__setattr__(self, name, frozen_f64(getattr(self, name), shape=(3,), name=name))
        for name in ("cash_before", "cash_after"):
            finite_float(getattr(self, name), name=name, minimum=0)
        strict_int(self.phase_order, name="phase_order", minimum=0)


@dataclass(frozen=True, slots=True)
class LedgerTransition:
    """A pure next state, its accounting event and an optional valuation mark."""

    state: LedgerState
    event: LedgerEvent
    mark: NavMark | None = None


def initial_ledger(
    *, cell_id: str, start_us: int, initial_cash: float = 100000.0, forced_liquidation_enabled: bool = False
) -> LedgerState:
    """Initialize cash only, without inventing an observed security price."""
    return LedgerState(
        cell_id,
        initial_cash,
        start_us,
        np.zeros(3),
        initial_cash,
        None,
        np.zeros(3),
        np.zeros(3),
        0.0,
        0.0,
        0.0,
        start_us,
        None,
        "INITIAL",
        frozenset(),
        forced_liquidation_enabled,
    )


def initial_capital_mark(state: LedgerState) -> NavMark:
    """Label initial capital separately from all observed daily closes."""
    if state.phase != "INITIAL":
        raise ContractError("initial capital anchor requires the initial state")
    return NavMark(state.start_us, MarkKind.INITIAL_CAPITAL, state.initial_cash, state.record_id)


def _check_event(
    state: LedgerState, event_id: str, timestamp_us: int, additional_ids: tuple[str, ...] = ()
) -> frozenset[str]:
    strict_int(timestamp_us, name="event_timestamp_us")
    identities = (event_id, *additional_ids)
    if any(not isinstance(item, str) or not item for item in identities) or len(set(identities)) != len(
        identities
    ):
        raise ContractError("unique nonempty event identities required")
    if state.phase == "FINAL":
        raise ChronologyError("final liquidated ledger cannot be advanced")
    if state.consumed_event_ids.intersection(identities):
        raise ChronologyError("already applied event identity")
    if timestamp_us <= state.cursor_us:
        raise ChronologyError("ledger event cursor must increase strictly")
    return state.consumed_event_ids.union(identities)


def _finish(
    state: LedgerState,
    candidate: LedgerState,
    *,
    event_id: str,
    kind: str,
    price_pnl: FloatArray,
    dividends: FloatArray,
    fee: float = 0.0,
    turnover: float = 0.0,
    mark_kind: MarkKind | None = None,
    phase_order: int = 0,
) -> LedgerTransition:
    residual = candidate.nav - state.nav - float(price_pnl.sum() + dividends.sum()) + fee
    if abs(residual) > 1e-10 * max(1.0, state.nav, candidate.nav):
        raise NumericalError("ledger event violates NAV conservation")
    candidate = replace(candidate, cumulative_residual=state.cumulative_residual + residual)
    event = LedgerEvent(
        event_id,
        candidate.cursor_us,
        kind,
        state.record_id,
        candidate.record_id,
        price_pnl,
        dividends,
        fee,
        turnover,
        state.nav,
        candidate.nav,
        residual,
        state.shares,
        candidate.shares,
        state.cash,
        candidate.cash,
        phase_order,
    )
    mark = (
        None
        if mark_kind is None
        else NavMark(candidate.cursor_us, mark_kind, candidate.nav, candidate.record_id)
    )
    return LedgerTransition(candidate, event, mark)


def open_session(
    state: LedgerState,
    *,
    event_id: str,
    timestamp_us: int,
    prices: FloatArray,
    dividends_per_share: FloatArray,
    dividend_event_ids: tuple[
        str | tuple[str, ...] | None, str | tuple[str, ...] | None, str | tuple[str, ...] | None
    ] = (None, None, None),
) -> LedgerTransition:
    """Mark opening prices and credit cash to entering shares before any fill."""
    if state.phase not in ("INITIAL", "CLOSE"):
        raise ChronologyError("next opening requires the preceding daily close")
    price = frozen_f64(prices, shape=(3,), name="opening_prices")
    dividend = frozen_f64(dividends_per_share, shape=(3,), name="dividends_per_share")
    if np.any(price <= 0) or np.any(dividend < 0):
        raise ContractError("positive opening prices and nonnegative dividends required")
    if not isinstance(dividend_event_ids, tuple) or len(dividend_event_ids) != 3:
        raise ContractError("dividend identities must align with three assets")
    groups = tuple(
        () if item is None else (item,) if isinstance(item, str) else item for item in dividend_event_ids
    )
    if any(not isinstance(group, tuple) for group in groups):
        raise ContractError("each asset requires an immutable source dividend identity group")
    if any(amount > 0 and not group for amount, group in zip(dividend, groups, strict=True)):
        raise ContractError("positive dividend requires a source event identity")
    consumed = _check_event(state, event_id, timestamp_us, tuple(item for group in groups for item in group))
    if state.open_us is not None and from_utc_us(timestamp_us).date() <= from_utc_us(state.open_us).date():
        raise ChronologyError("a new opening must belong to a later session date")
    pnl = np.zeros(3) if state.marks is None else state.shares * (price - state.marks)
    credited = state.shares * dividend
    candidate = replace(
        state,
        cash=state.cash + float(credited.sum()),
        marks=price,
        price_pnl=state.price_pnl + pnl,
        dividends=state.dividends + credited,
        cursor_us=timestamp_us,
        open_us=timestamp_us,
        phase="OPEN",
        consumed_event_ids=consumed,
        forced_sale_us=None,
    )
    return _finish(
        state, candidate, event_id=event_id, kind="OPEN_AND_DIVIDENDS", price_pnl=pnl, dividends=credited
    )


def forced_liquidation_open(state: LedgerState, *, event_id: str, one_way_cost: float) -> LedgerTransition:
    """Apply CL-15's precommitted A4 sale after dividends and before the decision.

    This is opening subphase 1 at the unchanged opening timestamp. Only an
    explicitly enabled A4 ledger may use it, exactly once per opening. The next
    ordinary target decision/fill remains at opening plus one/two microseconds.
    """
    if not state.forced_liquidation_enabled:
        raise ContractError("forced sale is available only to an explicitly enabled A4 ledger")
    if state.phase != "OPEN" or state.open_us is None or state.cursor_us != state.open_us:
        raise ChronologyError("forced sale requires the opening dividend phase")
    if state.forced_sale_us is not None:
        raise ChronologyError("forced sale already applied at this opening")
    if not isinstance(event_id, str) or not event_id:
        raise ContractError("nonempty forced-sale event identity required")
    if event_id in state.consumed_event_ids:
        raise ChronologyError("already applied event identity")
    solution = solve_rebalance(state.risky_dollars, state.cash, np.zeros(3), one_way_cost=one_way_cost)
    residual = solution.cash - state.nav + solution.fee
    candidate = replace(
        state,
        shares=np.zeros(3),
        cash=solution.cash,
        fees=state.fees + solution.fee,
        turnover=state.turnover + solution.turnover,
        cumulative_residual=state.cumulative_residual + residual,
        forced_sale_us=state.open_us,
        consumed_event_ids=state.consumed_event_ids.union((event_id,)),
    )
    return _finish(
        state,
        candidate,
        event_id=event_id,
        kind="FORCED_SALE",
        price_pnl=np.zeros(3),
        dividends=np.zeros(3),
        fee=solution.fee,
        turnover=solution.turnover,
        phase_order=1,
    )


def close_session(
    state: LedgerState, *, event_id: str, timestamp_us: int, prices: FloatArray
) -> LedgerTransition:
    """Record actual daily-close NAV and its price contribution exactly once."""
    consumed = _check_event(state, event_id, timestamp_us)
    if state.phase not in ("OPEN", "FILLED") or state.open_us is None or state.marks is None:
        raise ChronologyError("daily close requires an opening state")
    if from_utc_us(timestamp_us).date() != from_utc_us(state.open_us).date():
        raise ChronologyError("daily close must belong to its opening session")
    if timestamp_us <= state.open_us + 2:
        raise ChronologyError("daily close must follow the ideal opening-fill interval")
    price = frozen_f64(prices, shape=(3,), name="closing_prices")
    if np.any(price <= 0):
        raise ContractError("positive closing prices required")
    pnl = state.shares * (price - state.marks)
    candidate = replace(
        state,
        marks=price,
        price_pnl=state.price_pnl + pnl,
        cursor_us=timestamp_us,
        phase="CLOSE",
        consumed_event_ids=consumed,
    )
    return _finish(
        state,
        candidate,
        event_id=event_id,
        kind="DAILY_CLOSE",
        price_pnl=pnl,
        dividends=np.zeros(3),
        mark_kind=MarkKind.DAILY_CLOSE,
    )


def rebalance(
    state: LedgerState,
    *,
    event_id: str,
    decision_us: int,
    fill_us: int,
    target_weights: FloatArray,
    one_way_cost: float,
    is_hold: bool = False,
    final_liquidation: bool = False,
) -> LedgerTransition:
    """Execute a declared target at the same opening quote and certify actual fees."""
    consumed = _check_event(state, event_id, fill_us)
    strict_int(decision_us, name="decision_us")
    if state.phase != "OPEN" or state.open_us is None or state.marks is None:
        raise ChronologyError("a fill requires the current pre-trade opening state")
    if decision_us != state.open_us + 1 or fill_us != state.open_us + 2:
        raise ChronologyError("decision/fill must equal opening plus one/two microseconds")
    if not isinstance(is_hold, bool) or not isinstance(final_liquidation, bool):
        raise ContractError("hold and liquidation switches must be boolean")
    if state.forced_liquidation_enabled and not final_liquidation and state.forced_sale_us is None:
        raise ChronologyError("A4 allocation requires its precommitted opening sale first")
    target = frozen_f64(target_weights, shape=(3,), name="target_weights")
    validate_cost(one_way_cost)
    if final_liquidation and (is_hold or np.any(target != 0)):
        raise ContractError("final liquidation must sell all risky shares")
    if is_hold:
        if not np.array_equal(target, state.risky_dollars / state.nav):
            raise ContractError("hold target must equal actual holdings")
        solution = keep_holdings_solution(state.risky_dollars, state.cash)
        shares = state.shares
    else:
        solution = solve_rebalance(state.risky_dollars, state.cash, target, one_way_cost=one_way_cost)
        shares = solution.risky_dollars / state.marks
    # Record solver roundoff before constructing the next validated state.
    residual = solution.cash + float((shares * state.marks).sum()) - state.nav + solution.fee
    candidate = replace(
        state,
        shares=shares,
        cash=solution.cash,
        fees=state.fees + solution.fee,
        turnover=state.turnover + solution.turnover,
        cumulative_residual=state.cumulative_residual + residual,
        cursor_us=fill_us,
        phase="FINAL" if final_liquidation else "FILLED",
        consumed_event_ids=consumed,
    )
    return _finish(
        state,
        candidate,
        event_id=event_id,
        kind="FINAL_LIQUIDATION" if final_liquidation else "REBALANCE",
        price_pnl=np.zeros(3),
        dividends=np.zeros(3),
        fee=solution.fee,
        turnover=solution.turnover,
        mark_kind=MarkKind.FINAL_LIQUIDATION if final_liquidation else None,
    )


def liquidatable_boundary_mark(state: LedgerState, *, one_way_cost: float) -> NavMark:
    """Value a pre-rebalance boundary without charging cash or changing state."""
    if state.phase != "OPEN":
        raise ChronologyError("boundary mark requires entering-share dividends before rebalance")
    validate_cost(one_way_cost)
    value = state.cash + (1 - one_way_cost) * float(state.risky_dollars.sum())
    return NavMark(state.cursor_us, MarkKind.LIQUIDATABLE_BOUNDARY, value, state.record_id)


def _check_mark_sequence(marks: tuple[NavMark, ...], intermediate: MarkKind) -> None:
    if (
        len(marks) < 2
        or marks[0].kind != MarkKind.INITIAL_CAPITAL
        or marks[-1].kind != MarkKind.FINAL_LIQUIDATION
    ):
        raise ContractError("mark series requires distinct initial and final anchors")
    if any(mark.kind != intermediate for mark in marks[1:-1]):
        raise ContractError("mark convention cannot be mixed in an analysis series")
    if any(right.timestamp_us <= left.timestamp_us for left, right in pairwise(marks)):
        raise ChronologyError("mark sequence must increase strictly")


def boundary_log_increments(marks: tuple[NavMark, ...]) -> FloatArray:
    """Return liquidatable boundary log ratios that telescope to terminal wealth."""
    _check_mark_sequence(marks, MarkKind.LIQUIDATABLE_BOUNDARY)
    values = np.array([mark.value for mark in marks], dtype=np.float64)
    return frozen_f64(np.diff(np.log(values)), shape=(len(marks) - 1,), name="boundary_log_increments")


def maximum_daily_close_drawdown(marks: tuple[NavMark, ...]) -> float:
    """Measure close-to-close drawdown with initial and actual final-liquidation anchors."""
    _check_mark_sequence(marks, MarkKind.DAILY_CLOSE)
    values = np.array([mark.value for mark in marks], dtype=np.float64)
    return float(np.max(1 - values / np.maximum.accumulate(values)))
