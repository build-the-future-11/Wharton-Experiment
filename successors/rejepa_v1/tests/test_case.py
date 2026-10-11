import numpy as np
import pytest

from rejepa.case import LauraMandate


def test_case_cashflows_are_exact() -> None:
    mandate = LauraMandate()
    assert mandate.contributions() == ((2027, 300_000.0), (2028, 150_000.0))
    assert mandate.operating_cashflows()[0] == (2033, 50_000.0)
    assert mandate.operating_cashflows()[-1] == (2042, 50_000.0)
    assert len(mandate.operating_cashflows()) == 10
    assert mandate.annuity_due_reserve(0.0) == 500_000.0


def test_facility_interval_cannot_spend_reserve_or_flexibility() -> None:
    mandate = LauraMandate()
    values = np.array([600_000.0, 700_000.0, 800_000.0, 900_000.0])
    low, high = mandate.facility_contribution_interval(values, 500_000.0, 100_000.0, 0.5)
    assert 0.0 <= low <= high
    assert high <= 300_000.0


def test_invalid_discount_rate_rejected() -> None:
    with pytest.raises(ValueError):
        LauraMandate().annuity_due_reserve(-1.0)
