"""Independent generated ledger path and saved-event arithmetic references."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_ledger_audit import DAY, OPEN, close, fill, opening

from frem_s10.experiment.artifacts import _encode
from frem_s10.experiment.readback import _decode
from frem_s10.portfolio.ledger import (
    LedgerEvent,
    LedgerState,
    NavMark,
    boundary_log_increments,
    initial_capital_mark,
    initial_ledger,
    liquidatable_boundary_mark,
    maximum_daily_close_drawdown,
)


def fixed_path(
    capital: float, cost: float
) -> tuple[LedgerState, tuple[LedgerEvent, ...], tuple[NavMark, ...]]:
    state = initial_ledger(cell_id="generated-fixed-path", start_us=OPEN - 1, initial_cash=capital)
    events: list[LedgerEvent] = []
    marks: list[NavMark] = [initial_capital_mark(state)]
    for index, (price, dividend, close_price) in enumerate(
        ((10.0, 0.0, 8.0), (9.0, 0.5, 7.0), (11.0, 0.0, 11.0))
    ):
        step = opening(
            state,
            event_id=f"o{index}",
            timestamp_us=OPEN + index * DAY,
            price=price,
            dividend=dividend,
            dividend_id=f"d{index}" if dividend else None,
        )
        state = step.state
        events.append(step.event)
        trade = (
            fill(
                state,
                event_id=f"t{index}",
                target=(0.4, 0.0, 0.0) if index == 0 else (0.0, 0.0, 0.0),
                cost=cost,
                final=index == 2,
            )
            if index in (0, 2)
            else None
        )
        if trade is not None:
            state = trade.state
            events.append(trade.event)
            if trade.mark is not None:
                marks.append(trade.mark)
        if index < 2:
            closed = close(state, event_id=f"c{index}", price=close_price)
            assert closed.mark is not None
            state = closed.state
            events.append(closed.event)
            marks.append(closed.mark)
    return state, tuple(events), tuple(marks)


def test_fixed_action_full_ledger_scales_and_zero_cost_is_not_reoptimized() -> None:
    """TEST-052: independently fixed trades, same events and actual dividend ownership."""
    original, events, marks = fixed_path(100.0, 0.01)
    scaled, scaled_events, scaled_marks = fixed_path(700.0, 0.01)
    assert scaled.nav == pytest.approx(7 * original.nav, rel=1e-10)
    assert scaled.fees == pytest.approx(7 * original.fees, rel=1e-10)
    for left, right in zip(events, scaled_events, strict=True):
        np.testing.assert_allclose(right.shares_after, left.shares_after * 7, rtol=1e-10)
        assert right.cash_after == pytest.approx(left.cash_after * 7, rel=1e-10)
    assert [mark.value / 7 for mark in scaled_marks] == pytest.approx(
        [mark.value for mark in marks], rel=1e-10
    )
    zero, zero_events, _ = fixed_path(100.0, 0.0)
    assert [event.kind for event in events] == [event.kind for event in zero_events]
    # Four fixed shares earn $2 dividend and $4 price profit. No policy is run.
    assert zero.nav == pytest.approx(106.0) and zero.fees == 0


def test_daily_trough_between_decisions_recovers_and_entry_fee_is_drawdown() -> None:
    """TEST-053: the 7-dollar intervening close must not disappear from drawdown."""
    final, _, marks = fixed_path(100.0, 0.0)
    assert final.nav == pytest.approx(106.0)
    assert maximum_daily_close_drawdown(marks) == pytest.approx(0.10)
    _, _, charged_marks = fixed_path(100.0, 0.01)
    assert maximum_daily_close_drawdown(charged_marks) > 0.10


def test_saved_event_dollars_match_independent_decimal_buy_dividend_exit_reference() -> None:
    """TEST-051/081: verify decoded saved outputs with separate scalar accounting."""
    final, events, _ = fixed_path(100.0, 0.01)
    decoded: list[LedgerEvent] = []
    for event in events:
        restored = _decode(_encode(event))
        assert isinstance(restored, LedgerEvent)
        decoded.append(restored)
    with localcontext() as context:
        context.prec = 60
        nav_after_entry = Decimal(100) / Decimal("1.004")
        shares = nav_after_entry * Decimal(".4") / Decimal(10)
        cash = nav_after_entry * Decimal(".6")
        dividend = shares * Decimal(".5")
        cash += dividend
        sale_dollars = shares * Decimal(11)
        exit_fee = sale_dollars * Decimal(".01")
        terminal = cash + sale_dollars - exit_fee
    assert final.nav == pytest.approx(float(terminal), abs=2e-8)
    assert sum(float(event.dividends.sum()) for event in decoded) == pytest.approx(float(dividend), abs=1e-8)
    assert decoded[-1].fee == pytest.approx(float(exit_fee), abs=1e-8)
    for event in decoded:
        delta = math.fsum(
            [
                event.nav_after,
                -event.nav_before,
                -float(event.price_pnl.sum()),
                -float(event.dividends.sum()),
                event.fee,
            ]
        )
        assert abs(delta) <= 1e-10 * max(1, event.nav_before)
    restored_state = _decode(_encode(final))
    assert isinstance(restored_state, LedgerState)
    assert restored_state.record_id == final.record_id


def test_noncash_liquidatable_increments_telescope_without_charging_marks() -> None:
    """TEST-069: hypothetical exit reserve cannot become a booked fee or dividend loss."""
    initial = initial_ledger(cell_id="telescope", start_us=OPEN - 1, initial_cash=100.0)
    first = fill(opening(initial).state, cost=0.01).state
    closed = close(first).state
    endpoint = opening(
        closed, event_id="o2", timestamp_us=OPEN + DAY, price=11.0, dividend=0.5, dividend_id="d2"
    ).state
    mark = liquidatable_boundary_mark(endpoint, one_way_cost=0.01)
    before = endpoint.record_id
    final = fill(endpoint, event_id="exit", target=(0.0, 0.0, 0.0), cost=0.01, final=True)
    assert final.mark is not None and endpoint.record_id == before
    increments = boundary_log_increments((initial_capital_mark(initial), mark, final.mark))
    assert increments.sum() == pytest.approx(math.log(final.state.nav / 100), abs=1e-12)
    assert mark.value == pytest.approx(final.state.nav, abs=1e-8)
