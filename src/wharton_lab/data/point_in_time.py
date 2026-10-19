"""Opt-in point-in-time records and explicit file admission; no cache or fills.

Observations require event_time < decision_time and available_time <= decision.
Membership is effective at its effective_time, provided it has been published.
Revisions are opaque identifiers: time, never lexical revision order, selects
a vintage. A snapshot excludes assets without a known effective inclusion and
retains the caller's order for the eligible remainder. Its tuple-valued matrix
can be converted with numpy.asarray; ``records`` retains observation provenance.

File admission validates supplied provenance metadata and exact bytes. It does
not independently certify a provider, licence, availability claim, or source
truth. It never invokes or replaces the legacy ETF/cache loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Iterable, Sequence


def _time(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value.strip()


def _order(values: Sequence[str], name: str) -> tuple[str, ...]:
    if values is None or isinstance(values, (str, bytes, set, frozenset, dict)):
        raise ValueError(f"{name} must be an explicit sequence of names")
    result = tuple(_text(value, name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate names")
    return result


@dataclass(frozen=True)
class Observation:
    """One published vintage of a numeric observation; revision is an identifier."""

    asset: str
    field: str
    event_time: datetime
    available_time: datetime
    value: float
    revision: str = "0"

    def __post_init__(self) -> None:
        for name in ("asset", "field", "revision"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("event_time", "available_time"):
            object.__setattr__(self, name, _time(getattr(self, name), name))
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise ValueError("observation value must be a finite real number")
        try:
            value = float(self.value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("observation value must be finite") from exc
        if not math.isfinite(value):
            raise ValueError("observation value must be finite")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class MembershipUpdate:
    """Published universe inclusion/exclusion, possibly announced in advance."""

    asset: str
    effective_time: datetime
    available_time: datetime
    included: bool
    revision: str = "0"

    def __post_init__(self) -> None:
        for name in ("asset", "revision"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("effective_time", "available_time"):
            object.__setattr__(self, name, _time(getattr(self, name), name))
        if type(self.included) is not bool:
            raise ValueError("included must be an explicit bool")


def _vintages(records: Iterable, expected_type: type) -> tuple:
    """Reject reused conflicting IDs or ambiguous publication-time ties."""
    by_revision, by_publication = {}, {}
    for record in records:
        if not isinstance(record, expected_type):
            raise ValueError(f"records must be {expected_type.__name__} instances")
        if isinstance(record, Observation):
            key = (record.asset, record.field, record.event_time)
        else:
            key = (record.asset, record.effective_time)
        revision_key = (*key, record.revision)
        publication_key = (*key, record.available_time)
        for index, identity in ((by_revision, revision_key), (by_publication, publication_key)):
            if identity in index and index[identity] != record:
                raise ValueError(f"conflicting or ambiguous vintage for {key}")
            index[identity] = record
    return tuple(by_revision.values())  # Exact repeated records are idempotent.


def _membership_asof(updates: Iterable[MembershipUpdate], decision: datetime) -> dict:
    known = {}
    for update in _vintages(updates, MembershipUpdate):
        if update.effective_time > decision or update.available_time > decision:
            continue
        previous = known.get(update.asset)
        if previous is None or (update.effective_time, update.available_time) > (
            previous.effective_time, previous.available_time
        ):
            known[update.asset] = update
    return known


def eligible_assets(
    updates: Iterable[MembershipUpdate], *, decision_time: datetime, assets: Sequence[str]
) -> tuple[str, ...]:
    """Return known inclusions in requested order; unknown assets are excluded."""
    decision = _time(decision_time, "decision_time")
    requested = _order(assets, "assets")
    known = _membership_asof(updates, decision)
    return tuple(asset for asset in requested if asset in known and known[asset].included)


@dataclass(frozen=True)
class AsOfSnapshot:
    decision_time: datetime
    assets: tuple[str, ...]
    fields: tuple[str, ...]
    values: tuple[tuple[float | None, ...], ...]
    records: tuple[tuple[Observation | None, ...], ...]
    membership_records: tuple[MembershipUpdate, ...]
    excluded_assets: tuple[str, ...]
    max_age: timedelta | None


def build_asof_snapshot(
    observations: Iterable[Observation], *, decision_time: datetime,
    assets: Sequence[str], fields: Sequence[str],
    membership_updates: Iterable[MembershipUpdate], allow_missing: bool = False,
    max_age: timedelta | None = None,
) -> AsOfSnapshot:
    """Select latest event, then latest available vintage, without any filling.

    Excluded requested assets are recorded and omitted from the matrix. Missing
    fields raise only for eligible assets. With allow_missing=True they remain
    None in both values and records. max_age limits age from event to decision;
    None explicitly permits any historical age. Membership updates require both
    effective_time <= decision and available_time <= decision.
    """
    decision = _time(decision_time, "decision_time")
    requested, columns = _order(assets, "assets"), _order(fields, "fields")
    if not columns:
        raise ValueError("fields must not be empty")
    if type(allow_missing) is not bool:
        raise ValueError("allow_missing must be a bool")
    if max_age is not None and (not isinstance(max_age, timedelta) or max_age < timedelta(0)):
        raise ValueError("max_age must be a nonnegative timedelta or None")
    known = _membership_asof(membership_updates, decision)
    selected_assets = tuple(a for a in requested if a in known and known[a].included)
    selected_set, field_set = set(selected_assets), set(columns)
    selected = {}
    for record in _vintages(observations, Observation):
        if record.asset not in selected_set or record.field not in field_set:
            continue
        if record.event_time >= decision or record.available_time > decision:
            continue
        if max_age is not None and decision - record.event_time > max_age:
            continue
        key = (record.asset, record.field)
        previous = selected.get(key)
        if previous is None or (record.event_time, record.available_time) > (
            previous.event_time, previous.available_time
        ):
            selected[key] = record
    missing = [(a, f) for a in selected_assets for f in columns if (a, f) not in selected]
    if missing and not allow_missing:
        raise ValueError(f"missing as-of observations (unavailable or stale): {missing}")
    records = tuple(tuple(selected.get((a, f)) for f in columns) for a in selected_assets)
    return AsOfSnapshot(
        decision, selected_assets, columns,
        tuple(tuple(None if r is None else r.value for r in row) for row in records),
        records, tuple(known[a] for a in selected_assets),
        tuple(a for a in requested if a not in selected_set), max_age,
    )


@dataclass(frozen=True)
class DataManifest:
    """Required provenance declarations; declarations are not external certification."""

    source: str
    licence: str
    data_kind: str
    sha256: str
    availability_semantics: str
    corporate_action_policy: str
    missingness_policy: str
    universe_policy: str

    def __post_init__(self) -> None:
        unspecified = {"unknown", "unspecified", "todo", "tbd", "n/a", "na", "none", "null", "missing", ""}
        for name in (
            "source", "licence", "availability_semantics", "corporate_action_policy",
            "missingness_policy", "universe_policy",
        ):
            value = _text(getattr(self, name), name)
            if value.casefold() in unspecified:
                raise ValueError(f"{name} needs explicit provenance or a declared policy")
            object.__setattr__(self, name, value)
        if self.data_kind not in ("synthetic", "market"):
            raise ValueError("data_kind must be exactly 'synthetic' or 'market'")
        digest = _text(self.sha256, "sha256").lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True)
class AdmittedData:
    """The exact hashed snapshot, safe to parse without reopening a mutable file."""

    path: Path
    manifest: DataManifest
    raw: bytes
    sha256: str

    @property
    def market_eligible(self) -> bool:
        """Eligible under supplied declarations only; not a licence/source audit."""
        return self.manifest.data_kind == "market"


def admit_data_file(
    path: str | Path, manifest: DataManifest, *, for_market: bool = False,
) -> AdmittedData:
    """Read only this explicit path; fail on missing provenance or hash mismatch.

    There is no cache discovery, download, fallback, data rewriting, or promotion
    of synthetic observations to market evidence. Parse returned raw bytes so
    later changes to the source path cannot alter the admitted snapshot.
    """
    if not isinstance(manifest, DataManifest):
        raise ValueError("an explicit validated DataManifest is required")
    if type(for_market) is not bool:
        raise ValueError("for_market must be a bool")
    if for_market and manifest.data_kind != "market":
        raise ValueError("synthetic data cannot be admitted as market evidence")
    explicit_path = Path(path)
    raw = explicit_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest.sha256:
        raise ValueError("file SHA-256 mismatch: supplied provenance does not match exact bytes")
    return AdmittedData(explicit_path.resolve(), manifest, raw, digest)
