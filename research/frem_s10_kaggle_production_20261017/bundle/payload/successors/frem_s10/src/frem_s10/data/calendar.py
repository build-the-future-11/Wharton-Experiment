"""Timestamp-only weekly anchors, origin eligibility and fixed fold schedules."""

from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise

from ..errors import ChronologyError, ContractError, DataAdmissionError
from ..metadata import digest_parts, midnight_us, outcome_available_us, scheduled_open_us


@dataclass(frozen=True, slots=True)
class WeeklyAnchor:
    """The first admitted session of an ISO week and its ordered research clock."""

    index: int
    session_index: int
    date: date
    quote_us: int

    @property
    def decision_us(self) -> int:
        """Return the CL-01 decision instant, one microsecond after the quote."""
        return self.quote_us + 1

    @property
    def fill_us(self) -> int:
        """Return the ideal same-quote fill instant, two microseconds after quote."""
        return self.quote_us + 2


@dataclass(frozen=True, slots=True)
class OriginTiming:
    """Feature and target timestamps, calculated without looking at any values."""

    anchor_index: int
    endpoint_index: int
    origin_us: int
    endpoint_us: int
    available_us: int
    feature_available_us: int
    feature_interval_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CalendarExclusion:
    """An explicit timestamp-only origin exclusion; never an outcome filter."""

    anchor_index: int
    date: date
    reason: str


@dataclass(frozen=True, slots=True)
class CalendarSchedule:
    """Source calendar, complete H4 origin metadata and all incomplete origins."""

    dates: tuple[date, ...]
    anchors: tuple[WeeklyAnchor, ...]
    origins: tuple[OriginTiming, ...]
    exclusions: tuple[CalendarExclusion, ...]
    record_id: str


@dataclass(frozen=True, slots=True)
class FoldPlan:
    """Frozen i0+4k decision sequence and its unscored trailing session metadata."""

    fold_id: str
    start: date
    end_exclusive: date
    origin_indices: tuple[int, ...]
    endpoint_indices: tuple[int, ...]
    initial_exclusions: tuple[CalendarExclusion, ...]
    trailing_origin_indices: tuple[int, ...]
    unused_session_indices: tuple[int, ...]
    library_counts: tuple[int, ...]
    record_id: str


def weekly_anchors(dates: tuple[date, ...]) -> tuple[WeeklyAnchor, ...]:
    """Select earliest sessions by ISO week; require a reviewed complete calendar upstream."""
    if not dates or any(type(day) is not date for day in dates):
        raise ContractError("nonempty calendar-date tuple required")
    if dates != tuple(sorted(set(dates))):
        raise ChronologyError("calendar must be strictly increasing without duplicates")
    if any(day.weekday() >= 5 for day in dates):
        raise DataAdmissionError("weekend regular-session dates unsupported")
    anchors: list[WeeklyAnchor] = []
    previous_week: tuple[int, int] | None = None
    for session_index, day in enumerate(dates):
        iso = day.isocalendar()
        week = (iso.year, iso.week)
        if week != previous_week:
            anchors.append(WeeklyAnchor(len(anchors), session_index, day, scheduled_open_us(day)))
            previous_week = week
    for first, second in pairwise(anchors):
        first_monday = first.date - timedelta(days=first.date.weekday())
        second_monday = second.date - timedelta(days=second.date.weekday())
        if second_monday - first_monday != timedelta(days=7):
            raise DataAdmissionError("missing common-session ISO week; no inferred holiday or interval fill")
    return tuple(anchors)


def build_calendar_schedule(dates: tuple[date, ...]) -> CalendarSchedule:
    """Derive all complete feature/H4 timing records before any target is inspected."""
    anchors = weekly_anchors(dates)
    origins: list[OriginTiming] = []
    exclusions: list[CalendarExclusion] = []
    for anchor in anchors:
        i = anchor.index
        # The interval ending at this open is unavailable until the next midnight.
        if i < 14:
            exclusions.append(CalendarExclusion(i, anchor.date, "INSUFFICIENT_AVAILABLE_13_WEEK_HISTORY"))
            continue
        if i + 4 >= len(anchors):
            exclusions.append(CalendarExclusion(i, anchor.date, "INCOMPLETE_TRAILING_H4_ENDPOINT"))
            continue
        endpoint = anchors[i + 4]
        origins.append(
            OriginTiming(
                i,
                i + 4,
                anchor.quote_us,
                endpoint.quote_us,
                outcome_available_us(endpoint.date),
                outcome_available_us(anchors[i - 1].date),
                tuple(range(i - 14, i - 1)),
            )
        )
    return CalendarSchedule(
        dates,
        anchors,
        tuple(origins),
        tuple(exclusions),
        digest_parts("s10-calendar-schedule-v1", *(day.isoformat() for day in dates)),
    )


def _fold_window(fold_id: str) -> tuple[date, date]:
    windows = {
        "A": (date(2010, 1, 1), date(2013, 1, 1)),
        "B": (date(2013, 1, 1), date(2017, 1, 1)),
        "PILOT": (date(2008, 1, 1), date(2010, 1, 1)),
    }
    if fold_id not in windows:
        raise ContractError("unsupported S10 fold identity")
    return windows[fold_id]


def _first_eligible(
    schedule: CalendarSchedule,
    start: date,
    end: date,
) -> tuple[int, tuple[CalendarExclusion, ...]]:
    timing = {origin.anchor_index: origin for origin in schedule.origins}
    reasons = {excluded.anchor_index: excluded.reason for excluded in schedule.exclusions}
    candidates = tuple(anchor for anchor in schedule.anchors if start <= anchor.date < end)
    excluded: list[CalendarExclusion] = []
    for anchor in candidates:
        i = anchor.index
        if i not in timing:
            reason = reasons.get(i, "MISSING_REQUIRED_ORIGIN_METADATA")
        elif sum(record.available_us < anchor.decision_us for record in schedule.origins) < 64:
            reason = "INSUFFICIENT_64_MATURED_LIBRARY_RECORDS"
        else:
            return i, tuple(excluded)
        excluded.append(CalendarExclusion(i, anchor.date, reason))
    raise DataAdmissionError("fold has no eligible origin with complete required history")


def plan_fold(schedule: CalendarSchedule, fold_id: str) -> FoldPlan:
    """Freeze a fixed study fold using only dates, feature history and label maturity."""
    if schedule != build_calendar_schedule(schedule.dates):
        raise DataAdmissionError("calendar schedule differs from exact timestamp construction")
    start, end = _fold_window(fold_id)
    anchors = schedule.anchors
    timing = {origin.anchor_index: origin for origin in schedule.origins}
    first, excluded = _first_eligible(schedule, start, end)
    selected: list[int] = []
    trailing: list[int] = []
    for i in range(first, len(anchors), 4):
        if anchors[i].date >= end:
            break
        if i + 4 >= len(anchors) or anchors[i + 4].date >= end:
            trailing.append(i)
            continue
        if i not in timing:
            raise DataAdmissionError("scheduled fold origin lacks complete metadata", field=fold_id)
        if fold_id == "PILOT" and timing[i].available_us >= midnight_us(end):
            trailing.append(i)
            continue
        selected.append(i)
    if len(selected) < 24:
        raise DataAdmissionError("fold requires at least 24 complete four-interval blocks", field=fold_id)
    endpoints = tuple(i + 4 for i in selected)
    last_session = anchors[endpoints[-1]].session_index
    unused = tuple(i for i, day in enumerate(schedule.dates) if i > last_session and start <= day < end)
    counts = tuple(
        sum(record.available_us < anchors[i].decision_us for record in schedule.origins) for i in selected
    )
    identity = digest_parts(
        "s10-fold-plan-v1",
        schedule.record_id,
        fold_id,
        *(str(i) for i in selected),
        *(str(i) for i in endpoints),
    )
    return FoldPlan(
        fold_id,
        start,
        end,
        tuple(selected),
        endpoints,
        tuple(excluded),
        tuple(trailing),
        unused,
        counts,
        identity,
    )
