"""Independent constructed share/cash accounting and chronology checks."""

import math
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from frem_s10.errors import ChronologyError, ContractError
from frem_s10.metadata import utc_us
from frem_s10.portfolio.ledger import (
    MarkKind,
    boundary_log_increments,
    close_session,
    initial_capital_mark,
    initial_ledger,
    liquidatable_boundary_mark,
    maximum_daily_close_drawdown,
    open_session,
    rebalance,
)

OPEN = utc_us(datetime(2006, 1, 3, 14, 30, tzinfo=UTC))
DAY = 86_400_000_000
CLOSE_OFFSET = 23_400_000_000


def opening(state, *, event_id="open1", timestamp_us=OPEN, price=10.0, dividend=0.0, dividend_id=None):
    return open_session(
        state,
        event_id=event_id,
        timestamp_us=timestamp_us,
        prices=np.array([price, 10.0, 10.0]),
        dividends_per_share=np.array([dividend, 0.0, 0.0]),
        dividend_event_ids=(dividend_id, None, None),
    )


def fill(state, *, event_id="fill1", target=(0.4, 0.0, 0.0), cost=0.0, final=False, hold=False):
    return rebalance(
        state,
        event_id=event_id,
        decision_us=state.open_us + 1,
        fill_us=state.open_us + 2,
        target_weights=np.array(target),
        one_way_cost=cost,
        final_liquidation=final,
        is_hold=hold,
    )


def close(state, *, event_id="close1", price=10.0):
    return close_session(
        state,
        event_id=event_id,
        timestamp_us=state.open_us + CLOSE_OFFSET,
        prices=np.array([price, 10.0, 10.0]),
    )


def test_entering_share_dividend_before_endpoint_sale_and_negative_path():
    """TEST-015/051/053: four entering shares receive dividend before all are sold."""
    initial = initial_ledger(cell_id="hand_case", start_us=OPEN - 1, initial_cash=100.0)
    anchor = initial_capital_mark(initial)
    first = opening(initial).state
    bought = fill(first).state
    assert bought.shares[0] == 4 and bought.cash == 60
    closed = close(bought, price=9.0)
    assert closed.mark.kind == MarkKind.DAILY_CLOSE and closed.mark.value == 96
    endpoint = opening(
        closed.state,
        event_id="open2",
        timestamp_us=OPEN + DAY,
        price=8.0,
        dividend=1.0,
        dividend_id="dividend_A",
    )
    np.testing.assert_array_equal(endpoint.event.dividends, [4, 0, 0])
    assert endpoint.state.cash == 64 and endpoint.state.nav == 96
    np.testing.assert_array_equal(endpoint.state.price_pnl, [-8, 0, 0])
    boundary = liquidatable_boundary_mark(endpoint.state, one_way_cost=0.1)
    assert boundary.value == pytest.approx(92.8)
    assert endpoint.state.cash == 64 and endpoint.state.fees == 0
    final = fill(endpoint.state, event_id="exit", target=(0.0, 0.0, 0.0), cost=0.1, final=True)
    assert final.mark.kind == MarkKind.FINAL_LIQUIDATION
    assert final.state.cash == pytest.approx(92.8, abs=1e-8)
    assert final.state.fees == pytest.approx(3.2)
    np.testing.assert_array_equal(final.state.shares, np.zeros(3))
    assert maximum_daily_close_drawdown((anchor, closed.mark, final.mark)) == pytest.approx(0.072, abs=1e-10)
    assert final.state.nav - 100 == pytest.approx(-8 + 4 - 3.2, abs=1e-8)
    with pytest.raises(ChronologyError):
        fill(final.state, event_id="duplicate_exit", target=(0.0, 0.0, 0.0), cost=0.1, final=True)


def test_new_buyers_do_not_receive_opening_dividend_and_repeated_marks_charge_nothing():
    """TEST-015/041/050: cash-on-ex-date belongs solely to entering shares."""
    initial = initial_ledger(cell_id="new_buyer", start_us=OPEN - 1, initial_cash=100.0)
    opened = opening(initial, dividend=2.0, dividend_id="dividend_A").state
    assert opened.cash == 100 and not opened.dividends.any()
    assert liquidatable_boundary_mark(opened, one_way_cost=0.01).value == 100
    identity = opened.record_id
    for _ in range(3):
        liquidatable_boundary_mark(opened, one_way_cost=0.01)
    assert opened.record_id == identity
    bought = fill(opened, cost=0.01).state
    assert bought.dividends.sum() == 0
    # q=100/(1+.01*.4) follows independently from the all-cash sign region.
    assert bought.nav == pytest.approx(100 / 1.004, abs=1e-8)
    assert bought.fees == pytest.approx(100 - 100 / 1.004, abs=1e-8)


def test_cash_path_hypothetical_boundaries_and_log_increments_telescope():
    """TEST-051/069: zero holding incurs no cost; boundary ratios telescope."""
    initial = initial_ledger(cell_id="cash", start_us=OPEN - 1, initial_cash=100.0)
    anchor = initial_capital_mark(initial)
    cash = fill(opening(initial).state, target=(0.0, 0.0, 0.0), cost=0.004).state
    day1 = close(cash, price=5.0)
    day2 = opening(
        day1.state, event_id="open2", timestamp_us=OPEN + DAY, price=2.0, dividend=7.0, dividend_id="div2"
    ).state
    middle = liquidatable_boundary_mark(day2, one_way_cost=0.004)
    day2_close = close(day2, event_id="close2", price=1.0)
    day3 = opening(day2_close.state, event_id="open3", timestamp_us=OPEN + 2 * DAY, price=0.5).state
    final = fill(day3, event_id="exit", target=(0.0, 0.0, 0.0), cost=0.004, final=True)
    assert final.state.nav == pytest.approx(100, abs=1e-8)
    assert final.state.fees == final.state.dividends.sum() == 0
    increments = boundary_log_increments((anchor, middle, final.mark))
    assert math.fsum(increments) == pytest.approx(math.log(final.state.nav / 100), abs=1e-14)
    assert maximum_daily_close_drawdown((anchor, day1.mark, day2_close.mark, final.mark)) < 1e-9
    with pytest.raises(ContractError, match="convention"):
        maximum_daily_close_drawdown((anchor, middle, final.mark))


def test_duplicate_events_clock_and_missing_daily_marks_fail_closed():
    """TEST-018/053: stream identity and strict order protect ownership/accounting."""
    initial = initial_ledger(cell_id="clock", start_us=OPEN - 1)
    first = opening(initial, dividend=1.0, dividend_id="same_dividend").state
    with pytest.raises(ChronologyError):
        opening(first, event_id="unclosed_next", timestamp_us=OPEN + DAY)
    with pytest.raises(ChronologyError):
        rebalance(
            first,
            event_id="bad_time",
            decision_us=OPEN,
            fill_us=OPEN + 2,
            target_weights=np.zeros(3),
            one_way_cost=0,
        )
    closed = close(first).state
    with pytest.raises(ChronologyError, match="already applied"):
        opening(closed, event_id="open2", timestamp_us=OPEN + DAY, dividend=1.0, dividend_id="same_dividend")
    with pytest.raises(ChronologyError, match="already applied"):
        opening(closed, event_id="open1", timestamp_us=OPEN + DAY)
    with pytest.raises(ChronologyError, match="increase"):
        opening(closed, event_id="regression", timestamp_us=OPEN)
    with pytest.raises(ContractError, match="identity"):
        opening(closed, event_id="open2", timestamp_us=OPEN + DAY, dividend=1.0)


def test_immutable_state_can_be_independently_reconstructed_then_continued():
    """TEST-005/022: audit reconstruction gives identical future scientific state."""
    initial = initial_ledger(cell_id="reconstruct", start_us=OPEN - 1, initial_cash=100.0)
    filled = fill(opening(initial).state, cost=0.01).state
    external = np.array(filled.shares)
    reconstructed = replace(
        filled,
        shares=external,
        marks=np.array(filled.marks),
        price_pnl=np.array(filled.price_pnl),
        dividends=np.array(filled.dividends),
    )
    external[:] = 999
    assert reconstructed.record_id == filled.record_id
    with pytest.raises(ValueError):
        reconstructed.shares.setflags(write=True)
    uninterrupted = close(filled, price=8.0)
    replayed = close(reconstructed, price=8.0)
    assert uninterrupted.state.record_id == replayed.state.record_id
    assert uninterrupted.event.before_id == replayed.event.before_id
    assert filled.cursor_us == OPEN + 2  # no mutation through either continuation


def test_exact_hold_preserves_shares_and_cannot_hide_a_trade():
    """TEST-044: a hold is zero-trade, even when floating multiplication differs."""
    initial = initial_ledger(cell_id="hold", start_us=OPEN - 1, initial_cash=100.0)
    held = fill(opening(initial).state).state
    next_open = opening(close(held).state, event_id="open2", timestamp_us=OPEN + DAY, price=9.0).state
    exact = next_open.risky_dollars / next_open.nav
    result = fill(next_open, event_id="hold", target=exact, cost=0.004, hold=True)
    np.testing.assert_array_equal(result.state.shares, next_open.shares)
    assert result.event.fee == result.event.turnover == 0
    with pytest.raises(ContractError, match="actual holdings"):
        fill(next_open, event_id="fake_hold", target=(0.0, 0.0, 0.0), hold=True)
