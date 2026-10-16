"""Direct price/dividend targets with ex-date ownership and delayed availability."""

from dataclasses import dataclass
from datetime import date

import numpy as np

from ..arrays import FloatArray, frozen_f64
from ..domain import SessionPanel
from ..errors import ContractError, DataAdmissionError, IntegrityError, NumericalError
from ..metadata import outcome_available_us
from .calendar import CalendarSchedule, build_calendar_schedule


@dataclass(frozen=True, slots=True)
class IntervalTarget:
    """A direct endpoint price ratio plus non-reinvested cash per starting dollar."""

    anchor_index: int
    endpoint_index: int
    origin_us: int
    endpoint_us: int
    available_us: int
    g: FloatArray
    d: FloatArray

    def __post_init__(self) -> None:
        """Own immutable arrays and verify separate target domains and event order."""
        for name in ("g", "d"):
            object.__setattr__(self, name, frozen_f64(getattr(self, name), shape=(3,), name=name))
        if np.any(self.g <= 0) or np.any(self.d < 0):
            raise ContractError("positive price gross and nonnegative cash dividend required")
        if not self.origin_us < self.endpoint_us < self.available_us:
            raise ContractError("invalid interval target chronology")


def build_interval_targets(
    panel: SessionPanel,
    *,
    horizon: int,
    calendar: CalendarSchedule | None = None,
) -> tuple[IntervalTarget, ...]:
    """Compute H1 or direct H4 targets with origin < dividend date <= endpoint."""
    if type(horizon) is not int or horizon not in (1, 4):
        raise ContractError("only declared one/four weekly interval targets are supported")
    if panel.dates[-1] >= date(2017, 1, 1):
        raise DataAdmissionError("target values must be filtered to approved pre-2017 source scope")
    schedule = build_calendar_schedule(panel.dates) if calendar is None else calendar
    if schedule != build_calendar_schedule(panel.dates):
        raise IntegrityError("target calendar does not match admitted panel")
    targets: list[IntervalTarget] = []
    for start, end in zip(schedule.anchors, schedule.anchors[horizon:], strict=False):
        # Exclusive origin, inclusive endpoint credits entering-share entitlement.
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                dividends = np.sum(panel.dividends[start.session_index + 1 : end.session_index + 1], axis=0)
                g = panel.open[end.session_index] / panel.open[start.session_index]
                d = dividends / panel.open[start.session_index]
        except FloatingPointError as exc:
            raise NumericalError("nonfinite direct price/dividend target") from exc
        targets.append(
            IntervalTarget(
                start.index, end.index, start.quote_us, end.quote_us, outcome_available_us(end.date), g, d
            )
        )
    return tuple(targets)
