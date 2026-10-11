import numpy as np

from rejepa.portfolio import (
    AllocationPolicy,
    InternationalInstrument,
    allocate_dual_objective,
    objective_weights,
)


def _instruments() -> list[InternationalInstrument]:
    return [
        InternationalInstrument("US1", "US", "USD", True, True, 0.2),
        InternationalInstrument("TW1", "TW", "TWD", True, True, 0.15),
        InternationalInstrument("JP1", "JP", "JPY", True, True, 0.15),
    ]


def test_protection_weight_rises_near_reserve_date_or_shortfall() -> None:
    _, far = objective_weights(2027, 1.2)
    _, near = objective_weights(2032, 0.7)
    assert near > far


def test_dual_objective_allocation_obeys_cash_single_and_fx_caps() -> None:
    policy = AllocationPolicy(min_cash=0.1, max_single_security=0.2, max_non_base_currency=0.25)
    result = allocate_dual_objective(
        np.array([0.08, 0.10, 0.07]),
        np.array([0.10, 0.25, 0.15]),
        np.diag([0.03, 0.05, 0.04]),
        _instruments(),
        policy,
        year=2031,
        reserve_funded_ratio=0.9,
    )
    assert result["status"] == "optimized_research_only"
    weights = result["weights"]
    assert result["cash"] >= 0.1 - 1e-7
    assert weights.max() <= 0.2 + 1e-7
    assert weights[1:].sum() <= 0.25 + 1e-7


def test_unverified_instrument_blocks_allocation() -> None:
    instruments = _instruments()
    instruments[1] = InternationalInstrument("TW1", "TW", "TWD", True, False, 0.15)
    result = allocate_dual_objective(
        np.zeros(3), np.zeros(3), np.eye(3), instruments, AllocationPolicy(), 2027, 1.0
    )
    assert result["status"] == "blocked"
