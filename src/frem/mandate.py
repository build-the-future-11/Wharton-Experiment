"""Dated mandate validation and deterministic liability arithmetic.

This module never retrieves data or executes a portfolio.  Currency amounts in
one calculation must share the mandate's base currency.  Scenario probabilities
are assumptions, not empirically established funding guarantees.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

VERIFIED_SOURCE_KINDS = frozenset({"official", "user_document", "team_policy"})
SUPPORTED_OPTIMIZER_RULE_TYPES = frozenset(
    {
        "market_weight_cap",
        "market_weight_floor",
        "sector_weight_cap",
        "sector_weight_floor",
        "asset_class_weight_cap",
        "asset_class_weight_floor",
        "maximum_asset_weight",
        "minimum_cash_weight",
        "maximum_cash_weight",
        "gross_turnover_cap",
        "max_trade_adv_fraction",
        "max_position_adv_days",
        "foreign_currency_weight_cap",
        "eligible_universe",
        "long_only",
        "unlevered",
    }
)


def utc_timestamp(value: str | date | datetime | np.datetime64 | int) -> datetime:
    """Normalize dates and timestamps; integers are UTC nanoseconds."""
    if isinstance(value, (int, np.integer)):
        return datetime.fromtimestamp(int(value) / 1_000_000_000, UTC)
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("NaT is not a valid information timestamp")
        return utc_timestamp(int(value.astype("datetime64[ns]").astype(np.int64)))
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    text = str(value).replace("Z", "+00:00")
    return utc_timestamp(datetime.fromisoformat(text))


def _finite_number(
    value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not Boolean")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and out > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return out


@dataclass(frozen=True)
class Provenance:
    source_id: str
    title: str
    location: str
    kind: str
    verified_at: str
    effective_from: str
    effective_until: str | None = None
    revision: str | None = None
    scope: str = ""
    notes: str = ""

    def issues(self, as_of: str | date | datetime) -> list[str]:
        problems: list[str] = []
        if not all((self.source_id, self.title, self.location)):
            problems.append("source identity, title and location are required")
        if self.kind not in VERIFIED_SOURCE_KINDS:
            problems.append(f"source kind {self.kind!r} is not verified")
        try:
            decision = utc_timestamp(as_of)
            verified = utc_timestamp(self.verified_at)
            begins = utc_timestamp(self.effective_from)
            ends = utc_timestamp(self.effective_until) if self.effective_until else None
            if verified > decision:
                problems.append("source was verified after the decision timestamp")
            if begins > decision:
                problems.append("source is not effective at the decision timestamp")
            if ends is not None and decision > ends:
                problems.append("source has expired at the decision timestamp")
            if ends is not None and ends < begins:
                problems.append("source effective interval is inverted")
        except (ValueError, TypeError, OverflowError) as exc:
            problems.append(f"invalid source dates: {exc}")
        return problems


@dataclass(frozen=True)
class RuleRecord:
    """A rule's meaning is typed so a price band cannot become an exposure cap."""

    rule_id: str
    rule_type: str
    subject: str
    value: Any
    source_id: str
    interpretation: str
    enforced_by_optimizer: bool = False


@dataclass(frozen=True)
class CashFlow:
    date: str
    amount: float
    kind: str
    currency: str = "USD"
    source_id: str = ""

    def __post_init__(self) -> None:
        utc_timestamp(self.date)
        _finite_number(self.amount, "cash-flow amount", minimum=0)
        if self.kind not in {"contribution", "operating", "facility"}:
            raise ValueError(f"unsupported cash-flow kind {self.kind!r}")
        if not self.currency:
            raise ValueError("cash-flow currency is required")


@dataclass(frozen=True)
class Instrument:
    """An instrument-master snapshot known before a portfolio decision.

    ``market`` is the exposure market used for geographic caps, not necessarily
    the exchange where a depositary receipt trades.  ADV and price use base
    currency.  An ETF's sector/country labels must reflect approved exposure
    classification; opaque mixed funds cannot certify look-through caps.
    """

    asset_id: str
    symbol: str
    market: str
    sector: str
    asset_class: str
    currency: str
    available_at: str
    metadata_asof: str
    approval_source_id: str
    approved: bool = False
    tradable: bool = False
    listing_date: str | None = None
    delisting_date: str | None = None
    price_base: float | None = None
    average_daily_value_base: float | None = None
    lot_size: int = 1
    odd_lots_allowed: bool = False
    max_weight: float | None = None
    foreign_buy_headroom_base: float | None = None
    modified_duration: float | None = None
    convexity: float | None = None
    spread_duration: float | None = None
    exposure_verified: bool = False
    classification_source_id: str = ""
    market_exposures: dict[str, float] = field(default_factory=dict)
    sector_exposures: dict[str, float] = field(default_factory=dict)
    currency_exposures: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Instrument:
        fields = cls.__dataclass_fields__
        obj = {k: v for k, v in dict(value).items() if k in fields}
        if "asset_id" in obj:
            obj["asset_id"] = str(obj["asset_id"])
        for key in ("approved", "tradable", "odd_lots_allowed", "exposure_verified"):
            if key in obj and not isinstance(obj[key], (bool, np.bool_)):
                raise ValueError(f"{key} must be a Boolean; strings such as 'False' are not accepted")
        for key in (
            "price_base",
            "average_daily_value_base",
            "max_weight",
            "foreign_buy_headroom_base",
            "modified_duration",
            "convexity",
            "spread_duration",
        ):
            if key in obj and obj[key] is not None:
                number = float(obj[key])
                if np.isnan(number):
                    obj[key] = None  # Optional blanks in a mixed stock/bond CSV.
                elif not np.isfinite(number):
                    raise ValueError(f"{key} may not be infinite")
                else:
                    obj[key] = number
        for key in ("market_exposures", "sector_exposures", "currency_exposures"):
            if key in obj and isinstance(obj[key], str):
                obj[key] = json.loads(obj[key]) if obj[key].strip() else {}
        return cls(**obj)

    def issues(self, as_of: str | date | datetime, mandate: Mandate) -> list[str]:
        out: list[str] = []
        decision = utc_timestamp(as_of)
        for name in ("asset_id", "symbol", "market", "sector", "asset_class", "currency"):
            if not getattr(self, name):
                out.append(f"missing {name}")
        if not isinstance(self.approved, (bool, np.bool_)) or not self.approved:
            out.append("instrument is not explicitly approved")
        for flag in ("tradable", "odd_lots_allowed", "exposure_verified"):
            if not isinstance(getattr(self, flag), (bool, np.bool_)):
                out.append(f"{flag} must be explicitly Boolean")
        if self.asset_id not in mandate.eligible_asset_ids:
            out.append("instrument is absent from the dated eligible universe")
        if self.market not in mandate.allowed_markets:
            out.append("market is outside the mandate")
        if self.asset_class not in mandate.allowed_asset_classes:
            out.append("asset class is outside the mandate")
        if not self.exposure_verified:
            out.append("geographic and sector exposure classification is unverified")
        for name in ("market_exposures", "sector_exposures", "currency_exposures"):
            fractions = getattr(self, name)
            if fractions:
                if not isinstance(fractions, Mapping):
                    out.append(f"{name} must be a mapping")
                    continue
                try:
                    numbers = np.asarray(list(fractions.values()), dtype=float)
                    if (
                        not all(fractions)
                        or not np.all(np.isfinite(numbers))
                        or np.any(numbers < 0)
                        or not np.isclose(numbers.sum(), 1, atol=1e-6, rtol=0)
                    ):
                        out.append(f"{name} must be nonnegative finite exposures summing to one")
                    if name == "market_exposures" and any(
                        key not in mandate.allowed_markets and value > 0 for key, value in fractions.items()
                    ):
                        out.append("look-through exposure includes a market outside the mandate")
                except (ValueError, TypeError):
                    out.append(f"invalid {name}")
        for attr in ("approval_source_id", "classification_source_id"):
            source = mandate.sources.get(getattr(self, attr))
            if source is None:
                out.append(f"missing dated {attr}")
            elif source.issues(as_of):
                out.append(f"{attr} is invalid at the decision timestamp")
        try:
            available = utc_timestamp(self.available_at)
            observation = utc_timestamp(self.metadata_asof)
            if available > decision or observation > decision:
                out.append("instrument metadata contains future information")
            if observation > available:
                out.append("metadata observation is later than its availability timestamp")
            age = (decision - observation).total_seconds() / 86400.0
            if age > mandate.max_metadata_age_days:
                out.append("instrument metadata is stale")
            if self.listing_date and utc_timestamp(self.listing_date) > decision:
                out.append("instrument is not yet listed")
            if self.delisting_date and utc_timestamp(self.delisting_date) <= decision:
                out.append("instrument has delisted")
        except (ValueError, TypeError, OverflowError) as exc:
            out.append(f"invalid instrument dates: {exc}")
        if not isinstance(self.lot_size, (int, np.integer)) or self.lot_size < 1:
            out.append("lot size must be a positive integer")
        for name in ("price_base", "average_daily_value_base"):
            try:
                _finite_number(getattr(self, name), name, minimum=np.finfo(float).tiny)
            except (ValueError, TypeError):
                out.append(f"positive finite {name} is required")
        for name in (
            "max_weight",
            "foreign_buy_headroom_base",
            "modified_duration",
            "convexity",
            "spread_duration",
        ):
            value = getattr(self, name)
            if value is not None:
                try:
                    _finite_number(value, name, minimum=0, maximum=1 if name == "max_weight" else None)
                except (ValueError, TypeError) as exc:
                    out.append(str(exc))
        return out


@dataclass
class Mandate:
    mandate_id: str
    client_name: str
    base_currency: str
    status: str = "requires_inputs"
    sources: dict[str, Provenance] = field(default_factory=dict)
    rules: list[RuleRecord] = field(default_factory=list)
    cashflows: list[CashFlow] = field(default_factory=list)
    case_source_id: str | None = None
    participant_rules_source_id: str | None = None
    universe_source_id: str | None = None
    policy_source_id: str | None = None
    allowed_markets: list[str] = field(default_factory=list)
    allowed_asset_classes: list[str] = field(default_factory=list)
    eligible_asset_ids: list[str] = field(default_factory=list)
    min_cash_weight: float | None = None
    max_cash_weight: float | None = None
    max_asset_weight: float | None = None
    max_turnover: float | None = None
    max_trade_adv_fraction: float | None = None
    max_position_adv_days: float | None = None
    max_foreign_currency_weight: float | None = None
    market_weight_caps: dict[str, float] = field(default_factory=dict)
    market_weight_floors: dict[str, float] = field(default_factory=dict)
    sector_weight_caps: dict[str, float] = field(default_factory=dict)
    sector_weight_floors: dict[str, float] = field(default_factory=dict)
    asset_class_weight_caps: dict[str, float] = field(default_factory=dict)
    asset_class_weight_floors: dict[str, float] = field(default_factory=dict)
    max_metadata_age_days: float = 7.0
    long_only: bool = True
    allow_leverage: bool = False
    funding_confidence: float | None = None
    flexibility_buffer: float | None = None
    reserve_year: int | None = None
    communication_year: int | None = None
    missing_inputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Mandate:
        value = dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown mandate fields (possible misspelled constraints): {unknown}")
        value["sources"] = {
            k: Provenance(**v) if isinstance(v, Mapping) else v for k, v in value.get("sources", {}).items()
        }
        value["rules"] = [RuleRecord(**v) if isinstance(v, Mapping) else v for v in value.get("rules", [])]
        value["cashflows"] = [
            CashFlow(**v) if isinstance(v, Mapping) else v for v in value.get("cashflows", [])
        ]
        value["eligible_asset_ids"] = [str(x) for x in value.get("eligible_asset_ids", [])]
        return cls(**value)

    @classmethod
    def from_json(cls, path: str | Path) -> Mandate:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, as_of: str | date | datetime, require_allocation: bool = True) -> list[str]:
        out: list[str] = []
        try:
            utc_timestamp(as_of)
        except (ValueError, TypeError, OverflowError) as exc:
            return [f"invalid mandate decision date: {exc}"]
        if not all((self.mandate_id, self.client_name, self.base_currency)):
            out.append("mandate identifier, client and base currency are required")
        if any(c.currency != self.base_currency for c in self.cashflows):
            out.append("cashflows must be converted to base currency using an explicit dated FX assumption")
        if self.case_source_id:
            src = self.sources.get(self.case_source_id)
            if src is None:
                out.append("case source is missing")
            else:
                out.extend(f"case source: {x}" for x in src.issues(as_of))
        for rule in self.rules:
            if rule.source_id not in self.sources:
                out.append(f"rule {rule.rule_id} has no provenance")
            if rule.enforced_by_optimizer and rule.rule_type == "daily_price_band":
                out.append(
                    f"rule {rule.rule_id}: a daily price band cannot be enforced as a portfolio weight cap"
                )
            if rule.enforced_by_optimizer and rule.rule_type not in SUPPORTED_OPTIMIZER_RULE_TYPES:
                out.append(f"rule {rule.rule_id}: unsupported mandatory optimizer rule {rule.rule_type!r}")
            if rule.enforced_by_optimizer and rule.source_id in self.sources:
                out.extend(
                    f"rule {rule.rule_id}: {issue}" for issue in self.sources[rule.source_id].issues(as_of)
                )
            if rule.enforced_by_optimizer and rule.rule_type in SUPPORTED_OPTIMIZER_RULE_TYPES:
                group_fields = {
                    "market_weight_cap": self.market_weight_caps,
                    "market_weight_floor": self.market_weight_floors,
                    "sector_weight_cap": self.sector_weight_caps,
                    "sector_weight_floor": self.sector_weight_floors,
                    "asset_class_weight_cap": self.asset_class_weight_caps,
                    "asset_class_weight_floor": self.asset_class_weight_floors,
                }
                scalar_fields = {
                    "maximum_asset_weight": self.max_asset_weight,
                    "minimum_cash_weight": self.min_cash_weight,
                    "maximum_cash_weight": self.max_cash_weight,
                    "gross_turnover_cap": self.max_turnover,
                    "max_trade_adv_fraction": self.max_trade_adv_fraction,
                    "max_position_adv_days": self.max_position_adv_days,
                    "foreign_currency_weight_cap": self.max_foreign_currency_weight,
                }
                if rule.rule_type in group_fields:
                    actual = group_fields[rule.rule_type].get(rule.subject)
                elif rule.rule_type in scalar_fields:
                    actual = scalar_fields[rule.rule_type]
                elif rule.rule_type == "eligible_universe":
                    actual = self.eligible_asset_ids
                elif rule.rule_type == "long_only":
                    actual = self.long_only
                else:
                    actual = not self.allow_leverage
                if actual != rule.value:
                    out.append(
                        f"rule {rule.rule_id}: configured constraint differs from its recorded mandatory value"
                    )
        if not require_allocation:
            return out
        out.extend(self.missing_inputs)
        if self.status != "reviewed_research":
            out.append("allocation mandate has not been marked reviewed_research")
        for field_name in ("participant_rules_source_id", "universe_source_id", "policy_source_id"):
            source_id = getattr(self, field_name)
            source = self.sources.get(source_id or "")
            if source is None:
                out.append(f"{field_name} is required")
            else:
                out.extend(f"{field_name}: {x}" for x in source.issues(as_of))
                if field_name in {
                    "participant_rules_source_id",
                    "universe_source_id",
                } and source.kind not in {"official", "user_document"}:
                    out.append(f"{field_name} must cite supplied participant material, not a team policy")
        if not self.allowed_markets or not self.allowed_asset_classes or not self.eligible_asset_ids:
            out.append("dated allowed markets, asset classes and instrument universe are required")
        if len(set(self.eligible_asset_ids)) != len(self.eligible_asset_ids):
            out.append("eligible universe contains duplicate instrument IDs")
        for name in (
            "min_cash_weight",
            "max_cash_weight",
            "max_asset_weight",
            "max_trade_adv_fraction",
            "max_foreign_currency_weight",
        ):
            try:
                _finite_number(getattr(self, name), name, minimum=0, maximum=1)
            except (ValueError, TypeError):
                out.append(f"{name} must be explicitly supplied within [0, 1]")
        for name, upper in (
            ("max_turnover", 2.0),
            ("max_position_adv_days", None),
            ("max_metadata_age_days", None),
        ):
            try:
                _finite_number(getattr(self, name), name, minimum=0, maximum=upper)
            except (ValueError, TypeError):
                out.append(f"{name} must be an explicitly supplied nonnegative finite number")
        try:
            if (
                self.min_cash_weight is not None
                and self.max_cash_weight is not None
                and float(self.min_cash_weight) > float(self.max_cash_weight)
            ):
                out.append("minimum cash exceeds maximum cash")
        except (ValueError, TypeError):
            out.append("cash limits are not numeric")
        if self.long_only is not True or self.allow_leverage is not False:
            out.append("this optimizer supports explicitly long-only, unlevered portfolios")
        for prefix in ("market", "sector", "asset_class"):
            caps = getattr(self, f"{prefix}_weight_caps")
            floors = getattr(self, f"{prefix}_weight_floors")
            for label, values in (("caps", caps), ("floors", floors)):
                for key, number in values.items():
                    try:
                        _finite_number(number, f"{prefix} {key} {label}", minimum=0, maximum=1)
                    except (ValueError, TypeError) as exc:
                        out.append(str(exc))
            try:
                for key in caps.keys() & floors.keys():
                    if float(floors[key]) > float(caps[key]):
                        out.append(f"{prefix} floor exceeds cap for {key}")
                if sum(float(value) for value in floors.values()) > 1.0 + 1e-12:
                    out.append(f"{prefix} floors sum to more than total capital")
            except (ValueError, TypeError):
                out.append(f"{prefix} limits are not numeric")
        if self.reserve_year is not None:
            if not self.case_source_id or not any(value.kind == "operating" for value in self.cashflows):
                out.append(
                    "a liability mandate needs a verified case source and operating cash-flow schedule"
                )
            try:
                _finite_number(
                    self.funding_confidence, "funding confidence", minimum=np.finfo(float).eps, maximum=1
                )
                _finite_number(self.flexibility_buffer, "flexibility buffer", minimum=0)
            except (ValueError, TypeError):
                out.append("funding confidence and financial flexibility policy must be explicitly supplied")
        return list(dict.fromkeys(out))


def with_market_policy_cap(mandate: Mandate, market: str, weight: float, source: Provenance) -> Mandate:
    """Return a copy carrying a declared team cap; never relabel it as law."""
    _finite_number(weight, "market cap", minimum=0, maximum=1)
    if source.kind != "team_policy":
        raise ValueError("a discretionary market cap requires provenance kind='team_policy'")
    caps = {**mandate.market_weight_caps, market: float(weight)}
    sources = {**mandate.sources, source.source_id: source}
    rule = RuleRecord(
        f"policy-market-cap-{market}",
        "market_weight_cap",
        market,
        weight,
        source.source_id,
        "Team-defined exposure cap; not a foreign-investment law",
        True,
    )
    return replace(mandate, market_weight_caps=caps, sources=sources, rules=[*mandate.rules, rule])


def annuity_due_reserve(payment: float, periods: int, annual_rate: float) -> float:
    """PV of payments at times 0,...,periods-1; the first payment is immediate."""
    payment = _finite_number(payment, "payment", minimum=0)
    annual_rate = _finite_number(annual_rate, "annual_rate")
    if annual_rate <= -1 or not isinstance(periods, (int, np.integer)) or periods < 1:
        raise ValueError("annual_rate must exceed -1 and periods must be a positive integer")
    result = payment * np.power(1.0 + annual_rate, -np.arange(periods, dtype=float)).sum()
    if not np.isfinite(result):
        raise ValueError("reserve arithmetic overflowed under supplied assumptions")
    return float(result)


def discounted_cashflow_reserve(
    payments: Sequence[CashFlow], valuation_date: str, discount_factors: Sequence[float]
) -> float:
    """Price fixed liabilities from explicitly supplied date-matched discount factors.

    A discount factor of 1 is required for an immediate liability. This is an
    arithmetic price, not proof that eligible securities can lock those prices.
    """
    factors = np.asarray(discount_factors, dtype=float)
    if factors.shape != (len(payments),) or not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        raise ValueError("one positive finite discount factor is required per payment")
    as_of = utc_timestamp(valuation_date)
    if len({c.currency for c in payments}) > 1:
        raise ValueError("mixed-currency liabilities require explicit FX conversion")
    for cashflow, discount in zip(payments, factors, strict=True):
        when = utc_timestamp(cashflow.date)
        if when < as_of or cashflow.kind != "operating":
            raise ValueError("reserve payments must be future/immediate operating liabilities")
        if when == as_of and not np.isclose(discount, 1.0, atol=1e-12, rtol=0):
            raise ValueError("the immediate payment must not be discounted")
    return float(sum(c.amount * d for c, d in zip(payments, factors, strict=True)))


def project_contributions(
    contributions: Sequence[CashFlow], annual_returns: Mapping[int, float], target_year: int
) -> float:
    """Beginning-year contributions compound through target_year-1 inclusive.

    WInS balances and gains are deliberately absent from this interface.
    """
    if len({c.currency for c in contributions}) > 1:
        raise ValueError("mixed-currency contributions require explicit FX conversion")
    total = 0.0
    for cashflow in contributions:
        when = utc_timestamp(cashflow.date)
        if cashflow.kind != "contribution" or (when.month, when.day) != (1, 1):
            raise ValueError("use only beginning-year contributions in this function")
        if when.year > target_year:
            continue
        amount = float(cashflow.amount)
        for year in range(when.year, target_year):
            if year not in annual_returns:
                raise ValueError(f"missing return assumption for calendar year {year}")
            rate = _finite_number(annual_returns[year], f"return {year}", minimum=-1)
            amount *= 1.0 + rate
        total += amount
    return float(total)


def scenario_probabilities(count: int, probabilities: Sequence[float] | None = None) -> np.ndarray:
    if count < 1:
        raise ValueError("at least one scenario is required")
    p = np.full(count, 1.0 / count) if probabilities is None else np.asarray(probabilities, dtype=float)
    if p.shape != (count,) or not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("scenario probabilities must be finite, nonnegative, aligned and have positive mass")
    if not np.isclose(p.sum(), 1.0, atol=1e-8, rtol=0):
        raise ValueError("scenario probabilities must sum to one; silent renormalization is forbidden")
    return p / p.sum()


def weighted_quantile(
    values: Sequence[float], quantiles: Sequence[float] | float, probabilities: Sequence[float] | None = None
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("quantile values must be a finite one-dimensional vector")
    p = scenario_probabilities(len(x), probabilities)
    q = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if np.any((q < 0) | (q > 1)) or not np.all(np.isfinite(q)):
        raise ValueError("quantiles must lie in [0,1]")
    positive = p > 0
    x, p = x[positive], p[positive]
    order = np.argsort(x, kind="stable")
    x, p = x[order], p[order]
    indices = np.searchsorted(np.cumsum(p), q, side="left")
    return x[np.minimum(indices, len(x) - 1)]


def facility_contribution_analysis(
    projected_values: Sequence[float],
    reserve_amount: float,
    flexibility_buffer: float,
    *,
    interval_mass: float = 0.90,
    probabilities: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Conditional facility-capacity interval after protecting reserve and flexibility.

    Returns a mechanical capacity, not a recommended commitment. In scenarios
    with a reserve shortfall, capacity is zero and the shortfall is preserved.
    """
    values = np.asarray(projected_values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("projected portfolio values must be a nonnegative finite vector")
    reserve = _finite_number(reserve_amount, "reserve amount", minimum=0)
    buffer = _finite_number(flexibility_buffer, "flexibility buffer", minimum=0)
    mass = _finite_number(interval_mass, "interval mass", minimum=np.finfo(float).eps, maximum=1)
    p = scenario_probabilities(len(values), probabilities)
    capacity = np.maximum(values - reserve - buffer, 0.0)
    alpha = (1.0 - mass) / 2.0
    low, high = weighted_quantile(capacity, [alpha, 1.0 - alpha], p)
    return {
        "status": "conditional_arithmetic_only",
        "reserve_amount": reserve,
        "flexibility_buffer": buffer,
        "projected_portfolio_values": values.tolist(),
        "facility_capacity": capacity.tolist(),
        "reserve_shortfall": np.maximum(reserve - values, 0).tolist(),
        "assumed_scenario_reserve_coverage": float(p[values >= reserve].sum()),
        "assumed_scenario_reserve_and_buffer_coverage": float(p[values >= reserve + buffer].sum()),
        "interval": [float(low), float(high)],
        "requested_interval_mass": mass,
        "interval_scenario_mass": float(p[(capacity >= low) & (capacity <= high)].sum()),
        "limitations": [
            "Scenario weights are assumptions, not proven probabilities.",
            "Reserve adequacy still requires liability-asset matching or independently justified return paths.",
            "Capacity is a ceiling after the selected buffer, not a facility contribution recommendation.",
        ],
    }


def conditional_2031_analysis(
    balance_2031: float,
    annual_return_paths: Sequence[Sequence[float]],
    reserve_amount: float,
    flexibility_buffer: float,
    *,
    interval_mass: float = 0.90,
    probabilities: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Condition on observed beginning-2031 NAV; columns are years 2031 and 2032."""
    balance = _finite_number(balance_2031, "beginning-2031 portfolio value", minimum=0)
    paths = np.asarray(annual_return_paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] != 2 or not np.all(np.isfinite(paths)) or np.any(paths < -1):
        raise ValueError("supply finite [scenarios,2] simple returns for 2031 and 2032, each >= -1")
    target = balance * np.prod(1.0 + paths, axis=1)
    result = facility_contribution_analysis(
        target, reserve_amount, flexibility_buffer, interval_mass=interval_mass, probabilities=probabilities
    )
    result.update(
        {
            "conditioning_date": "2031-01-01",
            "target_date": "2033-01-01",
            "observed_beginning_2031_balance": balance,
            "projection_years": [2031, 2032],
        }
    )
    return result


def required_reserve_by_path(
    payments: Sequence[float], annual_return_paths: Sequence[Sequence[float]]
) -> np.ndarray:
    """Minimum initial reserve per supplied path; payment is made before growth.

    For K payments, each path has K-1 returns. Backward recursion uses
    V_k = payment_k + V_(k+1)/(1+r_k).  No last-payment-year return is needed.
    """
    amounts = np.asarray(payments, dtype=float)
    paths = np.asarray(annual_return_paths, dtype=float)
    if amounts.ndim != 1 or len(amounts) < 1 or not np.all(np.isfinite(amounts)) or np.any(amounts < 0):
        raise ValueError("payments must be a finite, nonnegative, nonempty vector")
    if paths.ndim != 2 or paths.shape[1] != len(amounts) - 1 or paths.shape[0] < 1:
        raise ValueError("a K-payment reserve needs exactly K-1 between-payment return columns")
    if not np.all(np.isfinite(paths)) or np.any(paths <= -1):
        raise ValueError("return paths must be finite and strictly greater than -1")
    required = np.full(paths.shape[0], amounts[-1])
    for index in range(len(amounts) - 2, -1, -1):
        required = amounts[index] + required / (1 + paths[:, index])
    if not np.all(np.isfinite(required)):
        raise ValueError("reserve requirements overflowed under supplied paths")
    return required


def reserve_path_analysis(
    reserve_amount: float,
    payments: Sequence[float],
    annual_return_paths: Sequence[Sequence[float]],
    *,
    confidence: float,
    probabilities: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Analyze supplied scenarios without generating or fitting them."""
    reserve = _finite_number(reserve_amount, "reserve", minimum=0)
    confidence = _finite_number(confidence, "confidence", minimum=np.finfo(float).eps, maximum=1)
    required = required_reserve_by_path(payments, annual_return_paths)
    p = scenario_probabilities(len(required), probabilities)
    amount = float(weighted_quantile(required, confidence, p)[0])
    return {
        "status": "conditional_arithmetic_only",
        "required_reserve_by_path": required.tolist(),
        "selected_reserve": reserve,
        "required_reserve_at_assumed_confidence": amount,
        "assumed_confidence": confidence,
        "assumed_scenario_funding_probability": float(p[required <= reserve].sum()),
        "initial_shortfall_by_path": np.maximum(required - reserve, 0).tolist(),
        "warning": "Finite supplied scenarios do not establish real-world high-certainty funding.",
    }


def cashflow_matching_reserve(
    prices: Sequence[float],
    security_cashflows: Sequence[Sequence[float]],
    liabilities: Sequence[float],
    *,
    max_units: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Minimum-cost nonnegative buy-and-hold funding of dated liabilities.

    Rows of ``security_cashflows`` align with liabilities; columns with quoted
    securities. Excess cash carries forward at zero interest. The first row
    includes the immediate payment: a security paying later cannot fund it.
    Add a one-unit cash instrument with price=1 and first-row cashflow=1 when
    immediate cash is eligible. Prices/flows and eligibility must be verified
    externally. Default risk and reinvestment assumptions are not certified.
    """
    from scipy.optimize import linprog

    prices_array = np.asarray(prices, dtype=float)
    flows = np.asarray(security_cashflows, dtype=float)
    obligations = np.asarray(liabilities, dtype=float)
    if prices_array.ndim != 1 or flows.shape != (len(obligations), len(prices_array)):
        raise ValueError("cashflow matrix must have [liability dates, securities] shape")
    if len(prices_array) == 0 or len(obligations) == 0:
        raise ValueError("at least one security and liability date are required")
    if (
        not np.all(np.isfinite(prices_array))
        or np.any(prices_array <= 0)
        or not np.all(np.isfinite(flows))
        or np.any(flows < 0)
        or not np.all(np.isfinite(obligations))
        or np.any(obligations < 0)
    ):
        raise ValueError("positive prices and nonnegative finite cashflows/liabilities are required")
    bounds = [(0, None)] * len(prices_array)
    if max_units is not None:
        limits = np.asarray(max_units, dtype=float)
        if limits.shape != prices_array.shape or np.any(limits < 0) or not np.all(np.isfinite(limits)):
            raise ValueError("max_units must be aligned, nonnegative and finite")
        bounds = [(0, float(x)) for x in limits]
    cumulative_flows = np.cumsum(flows, axis=0)
    cumulative_liabilities = np.cumsum(obligations)
    solved = linprog(
        prices_array, A_ub=-cumulative_flows, b_ub=-cumulative_liabilities, bounds=bounds, method="highs"
    )
    if not solved.success:
        return {"status": "infeasible", "solver_message": str(solved.message), "units": None, "cost": None}
    balances = cumulative_flows @ solved.x - cumulative_liabilities
    if np.min(balances) < -1e-7:
        return {
            "status": "infeasible",
            "solver_message": "Post-solve funding constraints failed",
            "units": None,
            "cost": None,
        }
    return {
        "status": "conditional_cashflow_match",
        "units": solved.x.tolist(),
        "cost": float(prices_array @ solved.x),
        "after_payment_cash_balances": balances.tolist(),
        "limitations": "Assumes stated payments occur, securities are eligible, and quoted prices are obtainable; credit, FX, fees and lot sizes require separate verification.",
    }
