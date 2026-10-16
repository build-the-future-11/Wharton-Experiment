"""Strict source/calendar/action admission for the fixed retrospective ETF scope."""

from dataclasses import dataclass
from datetime import date

import numpy as np

from ..domain import SessionPanel
from ..errors import ContractError, DataAdmissionError, IntegrityError
from ..metadata import (
    ASSETS,
    MICROSECONDS_PER_SECOND,
    digest_parts,
    finite_float,
    from_utc_us,
    scheduled_open_us,
)
from .source_reader import ScopedSourceReader, SourceIdentity, require_fields


def calendar_identity(dates: tuple[date, ...]) -> str:
    """Hash an exact expected session sequence without asserting its authority."""
    return digest_parts("s10-calendar-v1", *(day.isoformat() for day in dates))


@dataclass(frozen=True, slots=True)
class DataContract:
    """Reviewed calendar and restrictive source conventions; values are not inferred."""

    start_date: date
    end_exclusive: date
    expected_sessions: tuple[date, ...]
    calendar_id: str
    currency: str = "USD"
    price_units: str = "USD_per_share_unadjusted"
    dividend_units: str = "USD_per_share_cash_ex_date"
    action_convention: str = "no_splits_no_terminal_zero"
    availability_convention: str = "midnight_UTC_after_endpoint_session"
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Reject unsupported source/calendar assumptions before IO."""
        if type(self.start_date) is not date or type(self.end_exclusive) is not date:
            raise ContractError("scope boundaries must be calendar dates")
        if not self.start_date < self.end_exclusive <= date(2017, 1, 1):
            raise ContractError("S10 source scope must end before 2017")
        if (
            not self.expected_sessions
            or any(type(day) is not date for day in self.expected_sessions)
            or self.expected_sessions != tuple(sorted(set(self.expected_sessions)))
        ):
            raise ContractError("reviewed nonempty unique ordered expected calendar required")
        if any(not self.start_date <= day < self.end_exclusive for day in self.expected_sessions):
            raise ContractError("expected calendar escapes declared source scope")
        if any(day.weekday() >= 5 for day in self.expected_sessions):
            raise DataAdmissionError("weekend regular-session date unsupported")
        if self.calendar_id != calendar_identity(self.expected_sessions):
            raise IntegrityError("expected session calendar identity mismatch")
        conventions = (
            self.currency,
            self.price_units,
            self.dividend_units,
            self.action_convention,
            self.availability_convention,
        )
        if conventions != (
            "USD",
            "USD_per_share_unadjusted",
            "USD_per_share_cash_ex_date",
            "no_splits_no_terminal_zero",
            "midnight_UTC_after_endpoint_session",
        ):
            raise ContractError("unsupported units, actions or availability convention")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ContractError("unsupported data-contract schema version")


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """An admitted value panel plus source identity and explicit retrospective limits."""

    panel: SessionPanel
    calendar_id: str
    source_hashes: tuple[str, str, str]
    excluded_date_counts: tuple[int, int, int]
    limitations: tuple[str, ...] = (
        "CURRENT_VINTAGE_RETROSPECTIVE_NOT_PIT_CERTIFIED",
        "ASSUMED_OPEN_AND_AVAILABILITY_NOT_OBSERVED_VENDOR_LATENCY",
        "DIVIDEND_EX_DATE_CASH_NOT_PAYMENT_DATE_SETTLEMENT",
        "REDISTRIBUTION_RIGHTS_NOT_ESTABLISHED",
    )


@dataclass(frozen=True, slots=True)
class _AssetRows:
    dates: tuple[date, ...]
    opens: tuple[float, ...]
    closes: tuple[float, ...]
    dividends: tuple[float, ...]
    event_ids: tuple[tuple[str, ...], ...]
    excluded_count: int


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name}: object required", field=name)
    return value


def _timestamp(value: object) -> int:
    seconds = finite_float(value, name="source_timestamp")
    micros = seconds * MICROSECONDS_PER_SECOND
    if not micros.is_integer() or abs(micros) >= 2**63:
        raise ContractError("source timestamp must resolve to integer UTC microseconds")
    return int(micros)


def _read_quotes(
    rows: object, contract: DataContract
) -> tuple[
    tuple[date, ...],
    tuple[float, ...],
    tuple[float, ...],
    int,
]:
    if not isinstance(rows, list):
        raise ContractError("source rows must be an array")
    dates: list[date] = []
    opens: list[float] = []
    closes: list[float] = []
    excluded = 0
    previous: int | None = None
    for item in rows:
        row = _object(item, "quote_row")
        require_fields(row, {"timestamp", "open", "close"}, name="quote_row")
        timestamp = _timestamp(row["timestamp"])
        if previous is not None and timestamp <= previous:
            raise DataAdmissionError("source quote timestamps duplicate or regress")
        previous = timestamp
        day = from_utc_us(timestamp).date()
        if not contract.start_date <= day < contract.end_exclusive:
            excluded += 1
            continue  # Filter dates before inspecting any out-of-scope quote values.
        if timestamp != scheduled_open_us(day):
            raise DataAdmissionError("source timestamp differs from assumed regular-session open")
        dates.append(day)
        opens.append(finite_float(row["open"], name="open", strictly_positive=True))
        closes.append(finite_float(row["close"], name="close", strictly_positive=True))
    if tuple(dates) != contract.expected_sessions:
        raise DataAdmissionError("source calendar differs from reviewed expected session sequence")
    return tuple(dates), tuple(opens), tuple(closes), excluded


def _check_splits(raw_splits: object, contract: DataContract) -> None:
    splits = _object(raw_splits, "split_events")
    for item in splits.values():
        split = _object(item, "split_event")
        if "date" not in split:
            raise DataAdmissionError("split event lacks a scope timestamp")
        day = from_utc_us(_timestamp(split["date"])).date()
        if contract.start_date <= day < contract.end_exclusive:
            raise DataAdmissionError("split within source scope is unsupported")


def _read_dividends(
    raw_dividends: object,
    dates: tuple[date, ...],
    identity: SourceIdentity,
    contract: DataContract,
) -> tuple[tuple[float, ...], tuple[tuple[str, ...], ...]]:
    dividends = [0.0] * len(dates)
    events: list[list[str]] = [[] for _ in dates]
    date_indices = {day: index for index, day in enumerate(dates)}
    seen_dates: set[date] = set()
    for ordinal, item in sorted(_object(raw_dividends, "dividends").items()):
        event = _object(item, "dividend_event")
        require_fields(event, {"date", "amount"}, name="dividend_event")
        day = from_utc_us(_timestamp(event["date"])).date()
        if not contract.start_date <= day < contract.end_exclusive:
            continue  # No out-of-scope dividend amount is inspected.
        if day not in date_indices or day in seen_dates:
            raise DataAdmissionError("dividend ex-date is absent or duplicated")
        seen_dates.add(day)
        amount = finite_float(event["amount"], name="cash_dividend", minimum=0)
        index = date_indices[day]
        dividends[index] = amount
        events[index].append(
            digest_parts("s10-dividend-event-v1", identity.sha256, identity.symbol, day.isoformat(), ordinal)
        )
    return tuple(dividends), tuple(tuple(group) for group in events)


def _read_asset(raw: dict[str, object], identity: SourceIdentity, contract: DataContract) -> _AssetRows:
    require_fields(raw, {"symbol", "rows", "split_events", "dividends"}, name="retained_source_v1")
    if raw["symbol"] != identity.symbol:
        raise IntegrityError("source symbol differs from admitted identity")
    dates, opens, closes, excluded = _read_quotes(raw["rows"], contract)
    _check_splits(raw["split_events"], contract)
    dividends, events = _read_dividends(raw["dividends"], dates, identity, contract)
    return _AssetRows(dates, opens, closes, dividends, events, excluded)


def admit_sources(
    reader: ScopedSourceReader, identities: tuple[SourceIdentity, ...], *, contract: DataContract
) -> AdmissionResult:
    """Validate three exact sources and expose only their admitted date-filtered values."""
    if tuple(identity.symbol for identity in identities) != ASSETS:
        raise DataAdmissionError("exact ordered SPY/TLT/GLD source identities required")
    admitted: list[_AssetRows] = []
    for identity in identities:
        if reader.identities.get(identity.logical_path) != identity:
            raise IntegrityError("source identity does not match reader allowlist")
        admitted.append(_read_asset(reader.read_json(identity.logical_path), identity, contract))
    hashes = (identities[0].sha256, identities[1].sha256, identities[2].sha256)
    events = tuple(
        tuple(event for asset in admitted for event in asset.event_ids[index])
        for index in range(len(contract.expected_sessions))
    )
    panel = SessionPanel(
        contract.expected_sessions,
        np.asarray([asset.opens for asset in admitted], dtype=np.float64).T,
        np.asarray([asset.closes for asset in admitted], dtype=np.float64).T,
        np.asarray([asset.dividends for asset in admitted], dtype=np.float64).T,
        hashes,
        events,
    )
    return AdmissionResult(
        panel,
        contract.calendar_id,
        hashes,
        (admitted[0].excluded_count, admitted[1].excluded_count, admitted[2].excluded_count),
    )
