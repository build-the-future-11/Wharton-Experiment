"""Information clock: availability and maturity times."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ClockEvent:
    name: str
    time: datetime
    available: bool = True


class InformationClock:
    """Tracks when features are available vs when labels mature."""

    def __init__(self, decision_time: datetime, horizon_days: int = 1, reporting_lag_days: int = 0):
        self.decision_time = decision_time
        self.horizon_days = horizon_days
        self.reporting_lag_days = reporting_lag_days

    @property
    def feature_availability(self) -> datetime:
        return self.decision_time

    @property
    def label_maturity(self) -> datetime:
        return self.decision_time + timedelta(
            days=self.horizon_days + self.reporting_lag_days
        )

    def is_label_observable(self, as_of: datetime) -> bool:
        return as_of >= self.label_maturity

    def filter_matured(
        self,
        decision_times: Sequence[datetime],
        as_of: datetime,
    ) -> list[int]:
        indices: list[int] = []
        for i, t in enumerate(decision_times):
            clk = InformationClock(t, self.horizon_days, self.reporting_lag_days)
            if clk.is_label_observable(as_of):
                indices.append(i)
        return indices

    def timeline(self) -> list[ClockEvent]:
        return [
            ClockEvent("features_available", self.feature_availability, True),
            ClockEvent("horizon_end", self.decision_time + timedelta(days=self.horizon_days), False),
            ClockEvent("label_mature", self.label_maturity, True),
        ]
