"""Exact Laura Gao cash-flow and facility-range calculations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LauraMandate:
    initial_contribution: float = 300_000.0
    initial_year: int = 2027
    second_contribution: float = 150_000.0
    second_year: int = 2028
    reserve_year: int = 2033
    operating_payment: float = 50_000.0
    payment_count: int = 10
    final_payment_year: int = 2042

    def __post_init__(self) -> None:
        if min(self.initial_contribution, self.second_contribution, self.operating_payment) <= 0:
            raise ValueError("Cash flows must be positive")
        if self.payment_count != self.final_payment_year - self.reserve_year + 1:
            raise ValueError("Payment count must match the inclusive 2033-2042 schedule")
        if not self.initial_year < self.second_year < self.reserve_year:
            raise ValueError("Contribution and reserve years are inconsistent")

    def contributions(self) -> tuple[tuple[int, float], ...]:
        return (
            (self.initial_year, self.initial_contribution),
            (self.second_year, self.second_contribution),
        )

    def operating_cashflows(self) -> tuple[tuple[int, float], ...]:
        return tuple(
            (self.reserve_year + offset, self.operating_payment) for offset in range(self.payment_count)
        )

    def annuity_due_reserve(self, annual_discount_rate: float) -> float:
        """Reserve at the beginning of 2033, immediately before payment one."""
        if annual_discount_rate <= -1.0:
            raise ValueError("Discount rate must exceed -100 percent")
        return float(
            sum(
                self.operating_payment / ((1.0 + annual_discount_rate) ** offset)
                for offset in range(self.payment_count)
            )
        )

    def facility_contribution_interval(
        self,
        portfolio_values_2033: Iterable[float],
        operating_reserve: float,
        flexibility_floor: float,
        confidence: float,
    ) -> tuple[float, float]:
        """Central credible interval for what remains after reserve and flexibility.

        This is a mechanical research calculation, not a recommendation. The team
        must choose and justify confidence and flexibility before protected use.
        """
        values = np.asarray(tuple(portfolio_values_2033), dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError("Finite one-dimensional scenario values are required")
        if operating_reserve < 0 or flexibility_floor < 0:
            raise ValueError("Reserve and flexibility floor must be nonnegative")
        if not 0.0 < confidence < 1.0:
            raise ValueError("Confidence must lie strictly between zero and one")
        available = np.maximum(values - operating_reserve - flexibility_floor, 0.0)
        tail = (1.0 - confidence) / 2.0
        return float(np.quantile(available, tail)), float(np.quantile(available, 1.0 - tail))
