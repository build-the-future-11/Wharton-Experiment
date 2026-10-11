import numpy as np
import pytest

from rejepa.mandate import (
    ClientMandate,
    LiabilitySchedule,
    dynamic_sleeve_weights,
    evaluate_funding_frontier,
)


def test_liability_present_values_match_annuity_due_cases() -> None:
    schedule = LiabilitySchedule()
    assert schedule.present_value(0.0) == 500_000.0
    assert schedule.present_value(0.02) == pytest.approx(458_111.83, abs=0.1)
    assert schedule.present_value(0.05) == pytest.approx(405_391.08, abs=0.1)


def test_funding_frontier_reports_shortfall_and_facility_bounds() -> None:
    values = np.array([450_000.0, 500_000.0, 600_000.0, 700_000.0])
    result = evaluate_funding_frontier(ClientMandate(), values, 0.02)
    assert 0 <= result.funding_shortfall_probability <= 1
    assert result.funding_reliability == 1 - result.funding_shortfall_probability
    assert result.facility_contribution_bounds[0] <= result.facility_contribution_bounds[1]


def test_dynamic_sleeves_sum_to_one_and_protect_near_liability() -> None:
    mandate = ClientMandate()
    far = dynamic_sleeve_weights(
        mandate, year=2027, current_value=600_000, reserve_discount_rate=0.02, forecast_uncertainty=0.1
    )
    near = dynamic_sleeve_weights(
        mandate, year=2032, current_value=600_000, reserve_discount_rate=0.02, forecast_uncertainty=0.1
    )
    assert sum(far.values()) == pytest.approx(1.0)
    assert sum(near.values()) == pytest.approx(1.0)
    assert near["FUND"] >= far["FUND"]


def test_client_mandate_keeps_competition_separate() -> None:
    mandate = ClientMandate()
    assert mandate.separate_competition_performance
    assert mandate.no_outside_funding
    assert mandate.liabilities.final_year == 2042
