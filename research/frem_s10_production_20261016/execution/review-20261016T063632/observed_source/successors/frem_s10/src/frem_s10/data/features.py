"""Causal weekly contexts and the single immutable pre-2008 burn-in transform."""

from datetime import date
from itertools import pairwise

import numpy as np

from ..arrays import FloatArray, frozen_f64
from ..domain import OriginRecord, SessionPanel, Transform
from ..errors import ChronologyError, DataAdmissionError, IntegrityError, NumericalError
from ..metadata import digest_parts, midnight_us
from .calendar import CalendarSchedule, build_calendar_schedule
from .targets import IntervalTarget, build_interval_targets

BURNIN_CUTOFF_US = midnight_us(date(2008, 1, 1))


def build_context(
    intervals: tuple[IntervalTarget, ...],
    *,
    decision_us: int,
) -> FloatArray:
    """Map exactly thirteen consecutive fully available intervals to asset-major R^12."""
    if len(intervals) != 13:
        raise DataAdmissionError("exactly thirteen available weekly intervals required")
    for first, second in pairwise(intervals):
        if first.endpoint_index != second.anchor_index or first.endpoint_us != second.origin_us:
            raise ChronologyError("feature history intervals must be consecutive")
    if any(interval.endpoint_index != interval.anchor_index + 1 for interval in intervals):
        raise ChronologyError("feature history must contain one-week intervals")
    if any(interval.available_us >= decision_us for interval in intervals):
        raise ChronologyError("feature interval is not strictly available before decision")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            returns = np.log(np.stack([interval.g + interval.d for interval in intervals]))
            context = np.stack(
                (
                    returns[-1],
                    np.sum(returns[-4:], axis=0),
                    np.sum(returns, axis=0),
                    np.std(returns, axis=0, ddof=0),
                ),
                axis=1,
            )
    except FloatingPointError as exc:
        raise NumericalError("nonfinite weekly feature arithmetic") from exc
    return frozen_f64(context.reshape(12), shape=(12,), name="raw_context")


def build_origin_records(
    panel: SessionPanel,
    *,
    calendar: CalendarSchedule | None = None,
) -> tuple[OriginRecord, ...]:
    """Build admitted records while keeping current/future targets out of context identity."""
    schedule = build_calendar_schedule(panel.dates) if calendar is None else calendar
    if schedule != build_calendar_schedule(panel.dates):
        raise IntegrityError("origin calendar does not match admitted panel")
    weekly = build_interval_targets(panel, horizon=1, calendar=schedule)
    h4 = build_interval_targets(panel, horizon=4, calendar=schedule)
    records: list[OriginRecord] = []
    for timing in schedule.origins:
        i = timing.anchor_index
        intervals = tuple(weekly[j] for j in timing.feature_interval_indices)
        context = build_context(intervals, decision_us=timing.origin_us + 1)
        # Source bytes remain provenance; the target payload never enters this origin ID.
        identity = digest_parts(
            "s10-origin-context-v1",
            *panel.source_ids,
            str(timing.origin_us),
            str(timing.endpoint_us),
            str(timing.feature_available_us),
            context.astype("<f8").tobytes(),
        )
        records.append(
            OriginRecord(
                identity,
                i,
                timing.origin_us,
                timing.endpoint_us,
                timing.available_us,
                timing.feature_available_us,
                context,
                h4[i].g,
                h4[i].d,
                weekly[i].g,
                weekly[i].d,
            )
        )
    return tuple(records)


def fit_burnin_transform(records: tuple[OriginRecord, ...]) -> Transform:
    """Fit once from strictly pre-2008 matured records with fixed population-scale rules."""
    ids = tuple(record.record_id for record in records)
    if len(set(ids)) != len(ids):
        raise IntegrityError("duplicate origin identities supplied for burn-in")
    ordering = tuple((record.origin_us, record.record_id) for record in records)
    if ordering != tuple(sorted(ordering)):
        raise ChronologyError("burn-in source records must be sorted by origin and identity")
    eligible = tuple(record for record in records if record.available_us < BURNIN_CUTOFF_US)
    if len(eligible) < 64:
        raise DataAdmissionError("at least 64 strictly pre-2008 matured burn-in records required")
    contexts = np.stack([record.x for record in eligible])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            returns = np.stack([record.r for record in eligible])
            mean = np.mean(contexts, axis=0)
            raw_scale = np.std(contexts, axis=0, ddof=0)
            scale = np.where(raw_scale < 1e-12, 1.0, raw_scale)
            score_scale = np.maximum(np.std(returns, axis=0, ddof=0), 1e-4)
    except FloatingPointError as exc:
        raise NumericalError("nonfinite burn-in transform arithmetic") from exc
    eligible_ids = tuple(record.record_id for record in eligible)
    identity = digest_parts(
        "s10-pre2008-transform-v1",
        str(BURNIN_CUTOFF_US),
        *eligible_ids,
        mean.astype("<f8").tobytes(),
        scale.astype("<f8").tobytes(),
        score_scale.astype("<f8").tobytes(),
    )
    return Transform(mean, scale, score_scale, eligible_ids, identity)
