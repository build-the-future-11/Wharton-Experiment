"""Standard-library-only identities, units and UTC research-clock helpers."""

import hashlib
import math
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from .errors import ContractError

ASSETS = ("SPY", "TLT", "GLD")
FEATURES = ("last_total_log_return", "sum_4", "sum_13", "population_sd_13")
EXPERTS = ("U", "R", "C")
MICROSECONDS_PER_SECOND = 1_000_000
MICROSECONDS_PER_DAY = 86_400_000_000
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class Unit(StrEnum):
    """Units that remain distinct despite identical numerical shapes."""

    PRICE_GROSS = "price_gross"
    DIVIDEND_PER_START_DOLLAR = "cash_dividend_per_start_dollar"
    TOTAL_SIMPLE_RETURN = "total_simple_return"
    USD = "USD"
    SHARES = "shares"
    PRICE_USD = "USD_per_share"
    WEIGHT = "portfolio_weight"
    PROBABILITY = "probability"
    CONTEXT = "raw_context"
    STANDARDIZED_RETURN = "standardized_total_return"


def strict_int(value: object, *, name: str, minimum: int | None = None) -> int:
    """Require an ordinary integer, rejecting booleans and implicit conversion."""
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ContractError(f"{name}: integer outside declared domain", field=name)
    return value


def finite_float(
    value: object, *, name: str, minimum: float | None = None, strictly_positive: bool = False
) -> float:
    """Validate a scalar without accepting booleans or nonfinite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name}: finite numeric scalar required", field=name)
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ContractError(f"{name}: scalar outside declared domain", field=name)
    if strictly_positive and result <= 0:
        raise ContractError(f"{name}: positive scalar required", field=name)
    return result


def utc_us(value: datetime) -> int:
    """Convert an aware instant exactly to signed UTC microseconds."""
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ContractError("timezone-aware datetime required", field="timestamp")
    elapsed = value.astimezone(UTC) - EPOCH
    return (elapsed.days * 86_400 + elapsed.seconds) * MICROSECONDS_PER_SECOND + elapsed.microseconds


def from_utc_us(value: int) -> datetime:
    """Decode a strict integer microsecond timestamp into an aware UTC instant."""
    return EPOCH + timedelta(microseconds=strict_int(value, name="timestamp_us"))


def parse_utc_timestamp(value: object) -> int:
    """Parse an ISO timestamp only when its timezone is explicit."""
    if not isinstance(value, str):
        raise ContractError("ISO timestamp string required", field="timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ContractError("invalid ISO timestamp", field="timestamp") from exc
    return utc_us(parsed)


def midnight_us(day: date) -> int:
    """Return UTC midnight at the beginning of a calendar date."""
    if type(day) is not date:
        raise ContractError("calendar date required", field="date")
    return utc_us(datetime.combine(day, time(), tzinfo=UTC))


def scheduled_open_us(day: date) -> int:
    """Return the CL-01 assumed 09:30 Eastern regular-session opening instant."""
    if type(day) is not date:
        raise ContractError("calendar date required", field="date")
    local = datetime.combine(day, time(9, 30), tzinfo=ZoneInfo("America/New_York"))
    return utc_us(local)


def outcome_available_us(endpoint: date) -> int:
    """H4 and weekly outcomes become available at the next UTC midnight."""
    return midnight_us(endpoint + timedelta(days=1))


def digest_parts(kind: str, *parts: str | bytes) -> str:
    """Hash length-delimited typed parts without ambiguous concatenation."""
    digest = hashlib.sha256()
    for item in (kind, *parts):
        payload = item.encode("utf-8") if isinstance(item, str) else item
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
