"""Proof of pilot value filtering before target construction, using generated data."""

import hashlib
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from frem_s10.data.admission import DataContract, calendar_identity
from frem_s10.data.precision import prepare_precision_inputs
from frem_s10.data.source_reader import ScopedSourceReader, SourceIdentity
from frem_s10.errors import ChronologyError, IntegrityError
from frem_s10.metadata import ASSETS, midnight_us, scheduled_open_us


def fixture(
    tmp_path: Path,
) -> tuple[ScopedSourceReader, tuple[SourceIdentity, ...], DataContract, tuple[date, ...]]:
    dates = tuple(date(2004, 1, 5) + timedelta(weeks=i) for i in range(678))
    dates = tuple(day for day in dates if day < date(2017, 1, 1))
    pilot = tuple(day for day in dates if day < date(2010, 1, 1))
    identities = []
    for symbol in ASSETS:
        rows = [
            {
                "timestamp": scheduled_open_us(day) // 1_000_000,
                "open": 100.0 + i if day < date(2010, 1, 1) else "FORBIDDEN_FOLD_VALUE",
                "close": 100.2 + i if day < date(2010, 1, 1) else None,
            }
            for i, day in enumerate(dates)
        ]
        payload = json.dumps({"symbol": symbol, "rows": rows, "dividends": {}, "split_events": {}}).encode()
        name = symbol + ".json"
        (tmp_path / name).write_bytes(payload)
        identities.append(
            SourceIdentity(name, hashlib.sha256(payload).hexdigest(), symbol, midnight_us(date(2026, 10, 15)))
        )
    contract = DataContract(pilot[0], date(2010, 1, 1), pilot, calendar_identity(pilot))
    exact = tuple(identities)
    return ScopedSourceReader(tmp_path, exact), exact, contract, dates


def test_pilot_never_builds_targets_from_fold_values(tmp_path: Path) -> None:
    reader, sources, contract, dates = fixture(tmp_path)
    inputs = prepare_precision_inputs(
        reader,
        sources,
        pilot_contract=contract,
        projected_sessions=dates,
        projected_calendar_id=calendar_identity(dates),
        evidence_scope="GENERATED_ENGINEERING",
    )
    assert len(inputs.pilot.origin_indices) >= 24
    assert len(inputs.projected_a.origin_indices) >= 24
    assert len(inputs.projected_b.origin_indices) >= 24
    assert inputs.admission.panel.dates == contract.expected_sessions
    assert all(record.available_us < midnight_us(date(2010, 1, 1)) for record in inputs.records)
    assert len(inputs.record_id) == 64
    assert inputs.projected_calendar.dates[-1].year == 2016
    assert inputs.admission.panel.dates[-1].year == 2009
    with pytest.raises(IntegrityError):
        replace(
            inputs,
            projected_a=replace(inputs.projected_a, origin_indices=inputs.projected_a.origin_indices[:-1]),
        )


def test_pilot_rejects_later_value_declaration_before_source_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader, sources, contract, dates = fixture(tmp_path)

    def forbidden(path: str) -> dict[str, object]:
        raise AssertionError("source read before scope rejection")

    monkeypatch.setattr(reader, "read_json", forbidden)
    later = replace(contract, end_exclusive=date(2011, 1, 1))
    with pytest.raises(ChronologyError):
        prepare_precision_inputs(
            reader,
            sources,
            pilot_contract=later,
            projected_sessions=dates,
            projected_calendar_id=calendar_identity(dates),
            evidence_scope="GENERATED_ENGINEERING",
        )
    with pytest.raises(IntegrityError):
        prepare_precision_inputs(
            reader,
            sources,
            pilot_contract=contract,
            projected_sessions=dates[1:],
            projected_calendar_id=calendar_identity(dates[1:]),
        )


def test_real_generated_pilot_retains_all_fixed_blocks(tmp_path: Path) -> None:
    from frem_s10.experiment.precision import run_precision

    reader, sources, contract, dates = fixture(tmp_path)
    inputs = prepare_precision_inputs(
        reader,
        sources,
        pilot_contract=contract,
        projected_sessions=dates,
        projected_calendar_id=calendar_identity(dates),
        evidence_scope="GENERATED_ENGINEERING",
    )
    result = run_precision(inputs)
    selected = {r.anchor_index: r for r in inputs.records}
    assert result.origin_ids == tuple(selected[i].record_id for i in inputs.pilot.origin_indices)
    assert result.weekly_issue_count >= len(result.origin_ids)
    assert result.result.projected_blocks == len(inputs.projected_a.origin_indices) + len(
        inputs.projected_b.origin_indices
    )
    assert result.result.status in {"PRECISION_PASS", "PRECISION_INADEQUATE", "PRECISION_UNIDENTIFIABLE"}
    assert result.evidence_scope == "GENERATED_ENGINEERING"
    assert result.authorizes_campaign is False
    assert result.candidate.shape == result.equal.shape == result.global_stack.shape
    assert not result.candidate.flags.writeable
    assert all(len(set(ids)) == 3 for ids in result.forecast_ids)

    assert len(result.local_fallback_reasons) == len(result.origin_ids)
    assert sum(count for _, count in result.local_fallback_counts) <= len(result.origin_ids)
    assert dict(result.local_fallback_counts)["INSUFFICIENT_LOCAL_RECORDS"] > 0
    with pytest.raises(IntegrityError):
        replace(result, local_fallback_reasons=("INVENTED_REASON",) * len(result.origin_ids))
    assert 0 < result.cold_start_count <= len(result.origin_ids)
    assert tuple(block for block, _ in result.bootstrap) == (2, 4)
    assert all(
        array.shape == (5000, len(result.origin_ids)) and not array.flags.writeable
        for _, array in result.bootstrap
    )
    assert all(0 <= count <= len(result.origin_ids) for _, count in result.exact_zero_difference_counts)
    with pytest.raises(IntegrityError):
        replace(result, authorizes_campaign=True)
    with pytest.raises(IntegrityError):
        replace(result, evidence_scope="REAL_ADMITTED")
