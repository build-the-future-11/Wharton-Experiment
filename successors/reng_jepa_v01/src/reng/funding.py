"""Separate long-horizon FUND/GUARD/GROW liability engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiabilitySchedule:
    first_year: int = 2033
    payment_count: int = 10
    annual_payment: float = 50_000.0

    @property
    def years(self) -> np.ndarray:
        return np.arange(self.first_year, self.first_year + self.payment_count)

    def present_value(self, annual_discount_rate: float, *, as_of_year: int = 2033) -> float:
        if annual_discount_rate <= -1 or as_of_year > self.first_year:
            raise ValueError("Invalid liability valuation convention")
        offsets = self.years - as_of_year
        return float(np.sum(self.annual_payment / (1 + annual_discount_rate) ** offsets))


@dataclass(frozen=True)
class FundingPathResult:
    success: bool
    ending_wealth: float
    total_shortfall: float
    worst_shortfall: float
    payment_failures: int


@dataclass(frozen=True)
class FundingSummary:
    facility_contribution: float
    success_frequency: float
    expected_shortfall: float
    worst_shortfall: float
    terminal_wealth_median: float


def resample_monthly_vectors(
    monthly_returns: np.ndarray,
    months: int,
    paths: int,
    *,
    block_months: int = 12,
    seed: int = 11,
) -> np.ndarray:
    history = np.asarray(monthly_returns, dtype=float)
    if history.ndim != 2 or len(history) < block_months or months <= 0 or paths <= 0:
        raise ValueError("Invalid monthly history or requested simulation")
    if not np.isfinite(history).all() or np.any(history <= -1):
        raise ValueError("Monthly simple returns must be finite and above -100%")
    rng = np.random.default_rng(seed)
    output = np.empty((paths, months, history.shape[1]), dtype=float)
    for path in range(paths):
        cursor = 0
        while cursor < months:
            start = int(rng.integers(0, len(history) - block_months + 1))
            take = min(block_months, months - cursor)
            output[path, cursor : cursor + take] = history[start : start + take]
            cursor += take
    return output


def simulate_funding_path(
    monthly_portfolio_returns: np.ndarray,
    *,
    starting_wealth: float = 500_000.0,
    contributions: dict[int, float] | None = None,
    facility_month: int | None = None,
    facility_contribution: float = 0.0,
    liability_schedule: LiabilitySchedule | None = None,
    start_year: int = 2027,
    terminal_reserve: float = 0.0,
) -> FundingPathResult:
    returns = np.asarray(monthly_portfolio_returns, dtype=float)
    liability_schedule = liability_schedule or LiabilitySchedule()
    if returns.ndim != 1 or np.any(returns <= -1) or starting_wealth < 0 or terminal_reserve < 0:
        raise ValueError("Invalid funding path inputs")
    contributions = {0: 300_000.0, 12: 150_000.0} if contributions is None else dict(contributions)
    wealth = float(starting_wealth)
    total_shortfall = 0.0
    worst_shortfall = 0.0
    failures = 0
    payment_months = {(year - start_year) * 12 for year in liability_schedule.years}
    for month, monthly_return in enumerate(returns):
        wealth += float(contributions.get(month, 0.0))
        if facility_month == month:
            wealth -= facility_contribution
        if month in payment_months:
            wealth -= liability_schedule.annual_payment
        if wealth < 0:
            shortfall = -wealth
            total_shortfall += shortfall
            worst_shortfall = max(worst_shortfall, shortfall)
            failures += 1
            wealth = 0.0
        wealth *= 1.0 + monthly_return
    if wealth < terminal_reserve:
        shortfall = terminal_reserve - wealth
        total_shortfall += shortfall
        worst_shortfall = max(worst_shortfall, shortfall)
        failures += 1
    return FundingPathResult(failures == 0, wealth, total_shortfall, worst_shortfall, failures)


def funding_frontier(
    simulated_asset_returns: np.ndarray,
    weights: np.ndarray,
    facility_grid: np.ndarray,
    *,
    facility_month: int,
    starting_wealth: float = 500_000.0,
    terminal_reserve: float = 0.0,
) -> list[FundingSummary]:
    scenarios = np.asarray(simulated_asset_returns, dtype=float)
    allocation = np.asarray(weights, dtype=float)
    grid = np.asarray(facility_grid, dtype=float)
    if scenarios.ndim != 3 or allocation.shape != (scenarios.shape[2],):
        raise ValueError("Funding scenario/allocation dimensions differ")
    if not np.isclose(allocation.sum(), 1.0) or (allocation < 0).any():
        raise ValueError("Funding weights must be long-only and fully invested")
    portfolio_returns = scenarios @ allocation
    output = []
    for contribution in grid:
        wealth = np.full(len(portfolio_returns), float(starting_wealth), dtype=float)
        total_shortfall = np.zeros(len(portfolio_returns), dtype=float)
        worst_shortfall = np.zeros(len(portfolio_returns), dtype=float)
        failures = np.zeros(len(portfolio_returns), dtype=int)
        payment_months = set((LiabilitySchedule().years - 2027) * 12)
        for month in range(portfolio_returns.shape[1]):
            if month == 0:
                wealth += 300_000.0
            elif month == 12:
                wealth += 150_000.0
            if month == facility_month:
                wealth -= float(contribution)
            if month in payment_months:
                wealth -= LiabilitySchedule().annual_payment
            shortfall = np.maximum(-wealth, 0.0)
            failed = shortfall > 0
            total_shortfall += shortfall
            worst_shortfall = np.maximum(worst_shortfall, shortfall)
            failures += failed
            wealth = np.maximum(wealth, 0.0) * (1.0 + portfolio_returns[:, month])
        terminal_shortfall = np.maximum(terminal_reserve - wealth, 0.0)
        total_shortfall += terminal_shortfall
        worst_shortfall = np.maximum(worst_shortfall, terminal_shortfall)
        failures += terminal_shortfall > 0
        output.append(
            FundingSummary(
                facility_contribution=float(contribution),
                success_frequency=float(np.mean(failures == 0)),
                expected_shortfall=float(total_shortfall.mean()),
                worst_shortfall=float(worst_shortfall.max()),
                terminal_wealth_median=float(np.median(wealth)),
            )
        )
    return output


def sleeve_weights(
    wealth: float,
    next_12_month_liabilities: float,
    later_liability_value: float,
    reserve: float,
) -> dict[str, float]:
    if wealth <= 0 or min(next_12_month_liabilities, later_liability_value, reserve) < 0:
        raise ValueError("Invalid sleeve inputs")
    guard_dollars = min(wealth, next_12_month_liabilities + reserve)
    fund_dollars = min(max(wealth - guard_dollars, 0.0), later_liability_value)
    grow_dollars = max(wealth - guard_dollars - fund_dollars, 0.0)
    weights = {
        "FUND": fund_dollars / wealth,
        "GUARD": guard_dollars / wealth,
        "GROW": grow_dollars / wealth,
    }
    if not np.isclose(sum(weights.values()), 1.0):
        raise AssertionError("Accounting sleeves must exhaust wealth exactly once")
    return weights
