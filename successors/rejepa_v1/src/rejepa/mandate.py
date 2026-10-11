"""Configurable liability-aware client mandate and three-sleeve funding engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiabilitySchedule:
    first_year: int = 2033
    annual_payment: float = 50_000.0
    payment_count: int = 10
    beginning_of_year: bool = True

    def __post_init__(self) -> None:
        if self.first_year < 0 or self.annual_payment <= 0 or self.payment_count < 1:
            raise ValueError("Liability schedule values are invalid")

    @property
    def final_year(self) -> int:
        return self.first_year + self.payment_count - 1

    def cashflows(self) -> tuple[tuple[int, float], ...]:
        return tuple((self.first_year + offset, self.annual_payment) for offset in range(self.payment_count))

    def present_value(self, annual_rate: float, as_of_year: int | None = None) -> float:
        if annual_rate <= -1:
            raise ValueError("annual_rate must exceed -100 percent")
        anchor = self.first_year if as_of_year is None else as_of_year
        if anchor > self.first_year:
            raise ValueError("as_of_year cannot follow the first liability")
        timing_shift = 0 if self.beginning_of_year else 1
        return float(
            sum(
                amount / (1 + annual_rate) ** (year - anchor + timing_shift)
                for year, amount in self.cashflows()
            )
        )


@dataclass(frozen=True)
class LiquidityReserve:
    flexibility_floor: float = 50_000.0
    months_of_next_payment: float = 12.0

    def __post_init__(self) -> None:
        if self.flexibility_floor < 0 or self.months_of_next_payment < 0:
            raise ValueError("Liquidity reserve inputs must be nonnegative")

    def required(self, annual_payment: float) -> float:
        return float(self.flexibility_floor + annual_payment * self.months_of_next_payment / 12.0)


@dataclass(frozen=True)
class FacilityContributionPolicy:
    confidence: float = 0.90
    funding_reliability_target: float = 0.90
    minimum_post_contribution_flexibility: float = 50_000.0

    def __post_init__(self) -> None:
        if not 0 < self.confidence < 1 or not 0 < self.funding_reliability_target <= 1:
            raise ValueError("Probability targets must lie in (0, 1]")
        if self.minimum_post_contribution_flexibility < 0:
            raise ValueError("Flexibility must be nonnegative")

    def sustainable_bounds(
        self, terminal_values: np.ndarray, protected_capital: float
    ) -> tuple[float, float]:
        values = np.asarray(terminal_values, dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError("Finite one-dimensional terminal values are required")
        available = np.maximum(values - protected_capital - self.minimum_post_contribution_flexibility, 0.0)
        tail = (1 - self.confidence) / 2
        return float(np.quantile(available, tail)), float(np.quantile(available, 1 - tail))


@dataclass(frozen=True)
class ClientMandate:
    contributions: tuple[tuple[int, float], ...] = ((2027, 300_000.0), (2028, 150_000.0))
    liabilities: LiabilitySchedule = LiabilitySchedule()
    liquidity: LiquidityReserve = LiquidityReserve()
    facility: FacilityContributionPolicy = FacilityContributionPolicy()
    risk_preference: float = 0.50
    no_outside_funding: bool = True
    separate_competition_performance: bool = True

    def __post_init__(self) -> None:
        if not self.contributions or any(
            year >= self.liabilities.first_year or amount <= 0 for year, amount in self.contributions
        ):
            raise ValueError("Contributions must be positive and precede liabilities")
        if not 0 <= self.risk_preference <= 1:
            raise ValueError("risk_preference must lie in [0, 1]")

    def contribution_future_value(self, annual_return: float) -> float:
        if annual_return <= -1:
            raise ValueError("annual_return must exceed -100 percent")
        return float(
            sum(
                amount * (1 + annual_return) ** (self.liabilities.first_year - year)
                for year, amount in self.contributions
            )
        )


@dataclass(frozen=True)
class FundingFrontier:
    required_operating_reserve: float
    liquidity_reserve: float
    funding_shortfall_probability: float
    funding_reliability: float
    facility_contribution_bounds: tuple[float, float]
    funded_ratio_quantiles: tuple[float, float, float]


def evaluate_funding_frontier(
    mandate: ClientMandate,
    terminal_values_2033: np.ndarray,
    reserve_discount_rate: float,
) -> FundingFrontier:
    values = np.asarray(terminal_values_2033, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Finite terminal-value scenarios are required")
    operating = mandate.liabilities.present_value(reserve_discount_rate)
    liquidity = mandate.liquidity.required(mandate.liabilities.annual_payment)
    protected = operating + liquidity
    funded = values / protected if protected > 0 else np.full_like(values, np.inf)
    shortfall = float(np.mean(values < protected))
    return FundingFrontier(
        required_operating_reserve=operating,
        liquidity_reserve=liquidity,
        funding_shortfall_probability=shortfall,
        funding_reliability=1 - shortfall,
        facility_contribution_bounds=mandate.facility.sustainable_bounds(values, operating),
        funded_ratio_quantiles=tuple(float(value) for value in np.quantile(funded, [0.05, 0.5, 0.95])),
    )


def dynamic_sleeve_weights(
    mandate: ClientMandate,
    *,
    year: int,
    current_value: float,
    reserve_discount_rate: float,
    forecast_uncertainty: float,
) -> dict[str, float]:
    """Allocate FUND, GUARD and GROW without inventing security-level weights."""
    if current_value <= 0 or not np.isfinite(current_value):
        raise ValueError("current_value must be positive and finite")
    if forecast_uncertainty < 0 or not np.isfinite(forecast_uncertainty):
        raise ValueError("forecast_uncertainty must be finite and nonnegative")
    operating = mandate.liabilities.present_value(
        reserve_discount_rate, min(year, mandate.liabilities.first_year)
    )
    guard_required = mandate.liquidity.required(mandate.liabilities.annual_payment)
    time_pressure = 1 / (1 + max(mandate.liabilities.first_year - year, 0))
    fund = min(0.90, operating / current_value * (0.65 + 0.35 * time_pressure))
    guard = min(0.50, guard_required / current_value + min(forecast_uncertainty, 1.0) * 0.15)
    if fund + guard > 1:
        scale = 1 / (fund + guard)
        fund *= scale
        guard *= scale
    grow_capacity = max(1 - fund - guard, 0.0)
    grow = grow_capacity * mandate.risk_preference
    guard += grow_capacity - grow
    weights = {"FUND": float(fund), "GUARD": float(guard), "GROW": float(grow)}
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}
