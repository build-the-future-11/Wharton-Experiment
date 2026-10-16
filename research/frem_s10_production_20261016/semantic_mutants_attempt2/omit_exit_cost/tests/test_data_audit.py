"""Constructed data/clock/target audits; no retained market values are read."""

import dataclasses
import hashlib
import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from frem_s10.arrays import FloatArray
from frem_s10.data.admission import DataContract, admit_sources, calendar_identity
from frem_s10.data.calendar import build_calendar_schedule, plan_fold, weekly_anchors
from frem_s10.data.features import (
    BURNIN_CUTOFF_US,
    build_context,
    build_origin_records,
    fit_burnin_transform,
)
from frem_s10.data.source_reader import ScopedSourceReader, SourceIdentity, strict_json_loads
from frem_s10.data.targets import build_interval_targets
from frem_s10.domain import OriginRecord, SessionPanel
from frem_s10.errors import ChronologyError, ContractError, DataAdmissionError, IntegrityError, NumericalError
from frem_s10.metadata import ASSETS, digest_parts, from_utc_us, midnight_us, scheduled_open_us


class SourceDocument(TypedDict):
    symbol: str
    rows: list[dict[str, object]]
    split_events: dict[str, dict[str, object]]
    dividends: dict[str, dict[str, object]]


def _dates(start: date = date(2005, 1, 3), weeks: int = 30) -> tuple[date, ...]:
    return tuple(start + timedelta(weeks=i) for i in range(weeks))


def _panel(
    dates: tuple[date, ...] | None = None,
    opening: FloatArray | None = None,
    dividends: FloatArray | None = None,
) -> SessionPanel:
    dates = _dates() if dates is None else dates
    count = len(dates)
    opening = np.arange(count * 3, dtype=float).reshape(count, 3) + 100 if opening is None else opening
    dividends = np.zeros((count, 3)) if dividends is None else dividends
    return SessionPanel(
        dates,
        opening,
        opening * 1.01,
        dividends,
        ("source-SPY", "source-TLT", "source-GLD"),
        tuple(() for _ in dates),
    )


def _contract(dates: tuple[date, ...]) -> DataContract:
    return DataContract(dates[0], dates[-1] + timedelta(days=1), dates, calendar_identity(dates))


def _write_sources(
    tmp_path: Path,
    dates: tuple[date, ...],
    mutate: Callable[[SourceDocument], None] | None = None,
) -> tuple[ScopedSourceReader, tuple[SourceIdentity, ...]]:
    identities: list[SourceIdentity] = []
    for symbol in ASSETS:
        raw: SourceDocument = {
            "symbol": symbol,
            "rows": [
                {"timestamp": scheduled_open_us(day) // 1_000_000, "open": 100.0 + i, "close": 100.5 + i}
                for i, day in enumerate(dates)
            ],
            "split_events": {},
            "dividends": {},
        }
        if mutate is not None:
            mutate(raw)
        payload = json.dumps(raw).encode()
        relative = f"{symbol}.json"
        (tmp_path / relative).write_bytes(payload)
        identities.append(
            SourceIdentity(
                relative, hashlib.sha256(payload).hexdigest(), symbol, midnight_us(date(2026, 10, 15))
            )
        )
    ordered = tuple(identities)
    return ScopedSourceReader(tmp_path, ordered), ordered


def test_source_contract_reconciles_exact_calendars_and_reports_retrospective_limits(tmp_path: Path) -> None:
    dates = _dates(weeks=5)
    reader, identities = _write_sources(tmp_path, dates)
    admitted = admit_sources(reader, identities, contract=_contract(dates))
    assert admitted.panel.dates == dates
    assert admitted.panel.open.shape == (5, 3)
    assert admitted.excluded_date_counts == (0, 0, 0)
    assert "CURRENT_VINTAGE_RETROSPECTIVE_NOT_PIT_CERTIFIED" in admitted.limitations


@pytest.mark.parametrize("defect", ["missing", "duplicate", "zero", "negative", "split", "clock", "unknown"])
def test_source_defects_fail_before_silent_intersection_or_filling(tmp_path: Path, defect: str) -> None:
    dates = _dates(weeks=5)

    def mutate(raw: SourceDocument) -> None:
        if raw["symbol"] != "TLT":
            return
        if defect == "missing":
            raw["rows"].pop(2)
        elif defect == "duplicate":
            raw["rows"].insert(2, raw["rows"][1])
        elif defect in {"zero", "negative"}:
            raw["rows"][2]["open"] = 0 if defect == "zero" else -1
        elif defect == "split":
            raw["split_events"] = {"a": {"date": scheduled_open_us(dates[2]) // 1_000_000}}
        elif defect == "clock":
            original = raw["rows"][2]["timestamp"]
            assert isinstance(original, int)
            raw["rows"][2]["timestamp"] = original + 1
        elif defect == "unknown":
            raw["rows"][2]["adjusted_close"] = 100

    reader, identities = _write_sources(tmp_path, dates, mutate)
    with pytest.raises((ContractError, DataAdmissionError)):
        admit_sources(reader, identities, contract=_contract(dates))


def test_out_of_scope_values_are_not_used_or_validated_as_market_observations(tmp_path: Path) -> None:
    dates = _dates(weeks=5)

    def mutate(raw: SourceDocument) -> None:
        raw["rows"].append(
            {
                "timestamp": scheduled_open_us(date(2025, 1, 6)) // 1_000_000,
                "open": "DO_NOT_USE",
                "close": None,
            }
        )
        raw["dividends"] = {
            "future": {"date": scheduled_open_us(date(2025, 1, 6)) // 1_000_000, "amount": "DO_NOT_USE"}
        }

    reader, identities = _write_sources(tmp_path, dates, mutate)
    admitted = admit_sources(reader, identities, contract=_contract(dates))
    assert admitted.excluded_date_counts == (1, 1, 1)
    assert admitted.panel.dates == dates


@pytest.mark.parametrize("defect", ["negative", "duplicate", "absent"])
def test_dividend_corruption_is_rejected(tmp_path: Path, defect: str) -> None:
    dates = _dates(weeks=5)

    def mutate(raw: SourceDocument) -> None:
        day = dates[2] + timedelta(days=1) if defect == "absent" else dates[2]
        event: dict[str, object] = {
            "date": scheduled_open_us(day) // 1_000_000,
            "amount": -1 if defect == "negative" else 2,
        }
        raw["dividends"] = {"one": event}
        if defect == "duplicate":
            raw["dividends"]["two"] = event.copy()

    reader, identities = _write_sources(tmp_path, dates, mutate)
    with pytest.raises((ContractError, DataAdmissionError)):
        admit_sources(reader, identities, contract=_contract(dates))


def test_wrong_calendar_identity_units_and_unreviewed_source_fail(tmp_path: Path) -> None:
    dates = _dates(weeks=5)
    contract = _contract(dates)
    with pytest.raises(IntegrityError):
        dataclasses.replace(contract, calendar_id="incorrect")
    with pytest.raises(ContractError):
        dataclasses.replace(contract, dividend_units="total_return")
    reader, identities = _write_sources(tmp_path, dates)
    with pytest.raises(IntegrityError):
        reader.read_json("undeclared.json")
    (tmp_path / "SPY.json").write_text("{}")
    with pytest.raises(IntegrityError):
        admit_sources(reader, identities, contract=contract)


def test_source_reader_rejects_traversal_symlinks_and_overlarge_files(tmp_path: Path) -> None:
    with pytest.raises(ContractError):
        SourceIdentity("../source.json", "0" * 64, "SPY", 0)
    payload = b"{}"
    (tmp_path / "real.json").write_bytes(payload)
    (tmp_path / "link.json").symlink_to(tmp_path / "real.json")
    identity = SourceIdentity("link.json", hashlib.sha256(payload).hexdigest(), "SPY", 0)
    reader = ScopedSourceReader(tmp_path, (identity,))
    with pytest.raises(IntegrityError):
        reader.read_json("link.json")
    real = dataclasses.replace(identity, logical_path="real.json")
    small_reader = ScopedSourceReader(tmp_path, (real,), max_bytes=1)
    with pytest.raises(IntegrityError):
        small_reader.read_json("real.json")


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}', "[]", "{invalid"])
def test_source_parser_rejects_ambiguous_or_nonfinite_json(payload: str) -> None:
    with pytest.raises(ContractError):
        strict_json_loads(payload)


def test_weekly_anchor_clock_handles_dst_holidays_and_iso_year_boundaries() -> None:
    winter = weekly_anchors((date(2006, 1, 3), date(2006, 1, 4)))
    dst = weekly_anchors((date(2006, 3, 31), date(2006, 4, 3)))
    year = weekly_anchors((date(2006, 12, 29), date(2007, 1, 2)))
    assert tuple(anchor.session_index for anchor in winter) == (0,)
    assert len(year) == 2
    assert from_utc_us(winter[0].quote_us).strftime("%H:%M") == "14:30"
    assert from_utc_us(dst[1].quote_us).strftime("%H:%M") == "13:30"
    anchors = (*winter, *dst, *year)
    assert all(anchor.decision_us == anchor.quote_us + 1 for anchor in anchors)
    assert all(anchor.fill_us == anchor.quote_us + 2 for anchor in anchors)


def test_direct_h4_target_cash_dividends_excludes_origin_includes_endpoint() -> None:
    dates = _dates(weeks=5)
    prices = np.repeat(np.array([100, 120, 90, 110, 125], dtype=float)[:, None], 3, axis=1)
    dividends = np.repeat(np.array([7, 2, 0, 0, 3], dtype=float)[:, None], 3, axis=1)
    panel = _panel(dates, prices, dividends)
    target = build_interval_targets(panel, horizon=4)[0]
    np.testing.assert_array_equal(target.g, np.full(3, 1.25))
    np.testing.assert_array_equal(target.d, np.full(3, 0.05))
    weekly = build_interval_targets(panel, horizon=1)
    reinvested = np.prod(np.stack([interval.g + interval.d for interval in weekly]), axis=0)
    assert not np.allclose(target.g + target.d, reinvested)
    assert target.available_us == midnight_us(dates[-1] + timedelta(days=1))


def test_target_cash_sum_includes_nonanchor_daily_dividend() -> None:
    dates = tuple(
        date(2005, 1, 3) + timedelta(days=i)
        for i in range(29)
        if (date(2005, 1, 3) + timedelta(days=i)).weekday() < 5
    )
    dividends = np.zeros((len(dates), 3))
    dividends[2] = [1, 2, 3]
    panel = _panel(dates, np.full((len(dates), 3), 100.0), dividends)
    np.testing.assert_array_equal(build_interval_targets(panel, horizon=4)[0].d, [0.01, 0.02, 0.03])


def test_context_is_asset_major_and_excludes_current_open_label() -> None:
    panel = _panel()
    records = build_origin_records(panel)
    first = records[0]
    assert first.anchor_index == 14
    weekly_logs = np.log(panel.open[1:14] / panel.open[:13])
    expected = []
    for asset in range(3):
        series = weekly_logs[:, asset]
        expected.extend([series[-1], sum(series[-4:]), sum(series), np.std(series, ddof=0)])
    np.testing.assert_allclose(first.x, expected, rtol=1e-14, atol=1e-14)
    assert first.feature_available_us == midnight_us(panel.dates[13] + timedelta(days=1))
    assert first.feature_available_us < first.origin_us


def test_feature_availability_is_strict_and_missing_interval_fails() -> None:
    weekly = build_interval_targets(_panel(), horizon=1)[:13]
    with pytest.raises(ChronologyError):
        build_context(weekly, decision_us=weekly[-1].available_us)
    build_context(weekly, decision_us=weekly[-1].available_us + 1)
    with pytest.raises(DataAdmissionError):
        build_context(weekly[:-1], decision_us=weekly[-1].available_us + 1)
    corrupted = (*weekly[:5], dataclasses.replace(weekly[5], anchor_index=99), *weekly[6:])
    with pytest.raises(ChronologyError):
        build_context(corrupted, decision_us=weekly[-1].available_us + 1)


def test_future_target_perturbation_does_not_change_observable_origin_identity() -> None:
    panel = _panel()
    original = {record.anchor_index: record for record in build_origin_records(panel)}
    prices = panel.open.copy()
    dividends = panel.dividends.copy()
    prices[20:] *= 4
    dividends[20:] = 1000
    changed_panel = _panel(panel.dates, prices, dividends)
    changed = {record.anchor_index: record for record in build_origin_records(changed_panel)}
    assert not np.array_equal(original[20].g + original[20].d, changed[20].g + changed[20].d)
    for i in range(14, 21):
        assert original[i].record_id == changed[i].record_id
        np.testing.assert_array_equal(original[i].x, changed[i].x)


def _burnin_records(count: int = 64) -> tuple[OriginRecord, ...]:
    records: list[OriginRecord] = []
    for i in range(count):
        origin = scheduled_open_us(date(2005, 1, 3) + timedelta(weeks=i))
        records.append(
            OriginRecord(
                f"record-{i:03d}",
                i,
                origin,
                origin + 10_000_000,
                origin + 20_000_000,
                origin - 10_000_000,
                np.full(12, 7.0),
                np.ones(3),
                np.zeros(3),
            )
        )
    return tuple(records)


def test_burnin_63_64_strict_cutoff_population_scaling_and_immutability() -> None:
    records = _burnin_records()
    with pytest.raises(DataAdmissionError):
        fit_burnin_transform(records[:-1])
    at_boundary = dataclasses.replace(records[-1], available_us=BURNIN_CUTOFF_US)
    with pytest.raises(DataAdmissionError):
        fit_burnin_transform((*records[:-1], at_boundary))
    transform = fit_burnin_transform(records)
    np.testing.assert_array_equal(transform.feature_mean, np.full(12, 7.0))
    np.testing.assert_array_equal(transform.feature_scale, np.ones(12))
    np.testing.assert_array_equal(transform.score_scale, np.full(3, 1e-4))
    with pytest.raises(ValueError):
        transform.feature_mean[0] = 500
    later = dataclasses.replace(
        records[-1],
        record_id="later",
        origin_us=BURNIN_CUTOFF_US,
        endpoint_us=BURNIN_CUTOFF_US + 1,
        available_us=BURNIN_CUTOFF_US + 2,
        feature_available_us=BURNIN_CUTOFF_US - 1,
        x=np.full(12, 1e50),
        g=np.full(3, 1e100),
    )
    assert fit_burnin_transform((*records, later)).record_id == transform.record_id


def test_burnin_population_scales_match_independent_closed_form() -> None:
    records = tuple(
        dataclasses.replace(record, x=np.full(12, float(i)), g=np.full(3, 1 + i / 100))
        for i, record in enumerate(_burnin_records())
    )
    transform = fit_burnin_transform(records)
    sd = np.sqrt((64**2 - 1) / 12)
    np.testing.assert_allclose(transform.feature_mean, 31.5)
    np.testing.assert_allclose(transform.feature_scale, sd)
    np.testing.assert_allclose(transform.score_scale, sd / 100)


def test_calendar_plan_has_fixed_four_anchor_progression_and_unused_tail() -> None:
    schedule = build_calendar_schedule(_dates(weeks=627))
    for fold_id in ("PILOT", "A", "B"):
        plan = plan_fold(schedule, fold_id)
        assert len(plan.origin_indices) >= 24
        assert all(b - a == 4 for a, b in zip(plan.origin_indices, plan.origin_indices[1:], strict=False))
        assert all(b - a == 4 for a, b in zip(plan.origin_indices, plan.endpoint_indices, strict=True))
        assert all(schedule.anchors[i].date < plan.end_exclusive for i in plan.endpoint_indices)
        assert all(count >= 64 for count in plan.library_counts)
        last = schedule.anchors[plan.endpoint_indices[-1]].session_index
        assert plan.unused_session_indices == tuple(
            i for i, day in enumerate(schedule.dates) if i > last and plan.start <= day < plan.end_exclusive
        )
    assert plan_fold(schedule, "PILOT").origin_indices[0] >= 82


def test_calendar_plan_records_initial_insufficiency_and_cannot_fabricate_eligibility() -> None:
    schedule = build_calendar_schedule(_dates(date(2008, 12, 1), weeks=220))
    plan = plan_fold(schedule, "A")
    assert plan.initial_exclusions
    assert all(item.reason == "INSUFFICIENT_64_MATURED_LIBRARY_RECORDS" for item in plan.initial_exclusions)
    assert plan.origin_indices[0] == 82
    forged = dataclasses.replace(schedule, origins=())
    with pytest.raises(DataAdmissionError):
        plan_fold(forged, "A")
    with pytest.raises(DataAdmissionError):
        plan_fold(build_calendar_schedule(_dates(weeks=30)), "A")


def test_target_and_origin_builders_reject_forged_calendar() -> None:
    panel = _panel()
    schedule = build_calendar_schedule(panel.dates)
    forged = dataclasses.replace(schedule, record_id="forged")
    with pytest.raises(IntegrityError):
        build_interval_targets(panel, horizon=4, calendar=forged)
    with pytest.raises(IntegrityError):
        build_origin_records(panel, calendar=forged)


def test_missing_iso_week_fails_instead_of_constructing_an_irregular_target() -> None:
    dates = _dates()
    with pytest.raises(DataAdmissionError):
        build_calendar_schedule((*dates[:10], *dates[11:]))


def test_targets_reject_postcutoff_values_even_if_admission_was_bypassed() -> None:
    panel = _panel(_dates(date(2016, 12, 19), weeks=5))
    with pytest.raises(DataAdmissionError):
        build_interval_targets(panel, horizon=4)
    with pytest.raises(DataAdmissionError):
        build_origin_records(panel)


def test_finite_inputs_with_overflow_fail_as_numerical_errors() -> None:
    dates = _dates(weeks=5)
    panel = _panel(dates, np.full((5, 3), 1.0), np.full((5, 3), 1e308))
    with pytest.raises(NumericalError):
        build_interval_targets(panel, horizon=4)
    records = tuple(dataclasses.replace(record, x=np.full(12, 1e308)) for record in _burnin_records())
    with pytest.raises(NumericalError):
        fit_burnin_transform(records)


def test_admission_preserves_exact_per_asset_event_ids_including_zero_amount(tmp_path: Path) -> None:
    dates = _dates(weeks=5)

    def mutate(raw: SourceDocument) -> None:
        raw["dividends"] = {
            "source-event": {
                "date": scheduled_open_us(dates[2]) // 1_000_000,
                "amount": {"SPY": 0.0, "TLT": 2.0, "GLD": 3.0}[raw["symbol"]],
            }
        }

    reader, identities = _write_sources(tmp_path, dates, mutate)
    panel = admit_sources(reader, identities, contract=_contract(dates)).panel
    aligned = panel.require_asset_event_ids()
    expected = tuple(
        digest_parts(
            "s10-dividend-event-v1", identity.sha256, identity.symbol, dates[2].isoformat(), "source-event"
        )
        for identity in identities
    )
    assert aligned[2] == expected
    assert panel.event_ids[2] == expected
    assert aligned[0] == (None, None, None)
    assert panel.dividends[2, 0] == 0 and aligned[2][0] is not None


def test_asset_event_provenance_rejects_missing_misaligned_or_duplicate_identity() -> None:
    dates = _dates(weeks=1)
    panel = _panel(dates, dividends=np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ContractError, match="requires admitted"):
        panel.require_asset_event_ids()
    with pytest.raises(ContractError, match="positive dividend"):
        dataclasses.replace(panel, asset_event_ids=((None, None, None),))
    with pytest.raises(ContractError, match="flattened"):
        dataclasses.replace(
            panel, event_ids=(("event-SPY",),), asset_event_ids=(("other-event", None, None),)
        )
    with pytest.raises(ContractError, match="duplicate"):
        dataclasses.replace(
            panel, event_ids=(("event", "event"),), asset_event_ids=(("event", "event", None),)
        )
    admitted = dataclasses.replace(
        panel, event_ids=(("event-SPY",),), asset_event_ids=(("event-SPY", None, None),)
    )
    assert admitted.require_asset_event_ids() == (("event-SPY", None, None),)
