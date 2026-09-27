import numpy as np
import pytest

from wharton_lab.competition.stress import rebalance, simulate


def test_rebalance_conserves_wealth_including_fees():
    h, c, fee, turnover = rebalance([0.7, 0.1], 0.2, [0.2, 0.4], 50)
    assert h.sum() + c + fee == pytest.approx(1)
    assert fee == pytest.approx(turnover * 0.005)
    assert h / (1 - fee) == pytest.approx([0.2, 0.4])


def test_withdrawal_deducted_exactly_once():
    r = simulate(np.zeros((3, 2)), [0.1, 0.1, 0.1], cost_bps=0)
    assert r["terminal_wealth"] == pytest.approx(0.7)
    assert r["total_paid"] == pytest.approx(0.3)
    assert r["max_drawdown_ex_withdrawals"] == pytest.approx(0)


def test_insolvency_is_retained_not_clipped_into_success():
    r = simulate(np.zeros((3, 2)), [0.6, 0.6, 0.6], cost_bps=0)
    assert r["first_shortfall_month"] == 2
    assert r["unpaid_at_end"] == pytest.approx(0.8)
    assert r["terminal_wealth"] == pytest.approx(0)
    assert r["total_paid"] + r["unpaid_at_end"] == pytest.approx(r["total_obligations"])


def test_fee_accounting_and_no_hidden_injection():
    r = simulate(np.zeros((12, 2)), np.full(12, .01), policy="fully_invested", cost_bps=50)
    assert r["terminal_wealth"] + r["total_fees"] + r["total_paid"] == pytest.approx(1)
    assert r["total_fees"] > 0
    assert r["terminal_wealth"] < .88


def test_future_return_cannot_change_past_actions():
    a = np.zeros((12, 2)); b = a.copy(); b[8:] = -.4
    ra = simulate(a, np.full(12, .01)); rb = simulate(b, np.full(12, .01))
    assert ra["trajectory"][:8] == rb["trajectory"][:8]


def test_crash_drawdown_not_counting_spending():
    r = simulate(np.array([[-.5, -.5]]), [0.1], policy="fully_invested", cost_bps=0)
    assert r["terminal_wealth"] == pytest.approx(.4)
    assert r["max_drawdown_ex_withdrawals"] == pytest.approx(.5)


@pytest.mark.parametrize("kwargs", [{"cost_bps": -1}, {"cost_bps": 10000}, {"initial_wealth": -1}, {"reserve_months": 0}])
def test_invalid_finance_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        simulate(np.zeros((2, 2)), [0, 0], **kwargs)
