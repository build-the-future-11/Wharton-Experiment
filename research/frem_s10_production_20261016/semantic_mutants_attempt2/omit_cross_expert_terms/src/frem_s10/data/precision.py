"""Prepare a pre-2010 value view and bind projected counts to timestamps only."""

from dataclasses import dataclass
from datetime import date

import numpy as np

from frem_s10.data.admission import AdmissionResult, DataContract, admit_sources, calendar_identity
from frem_s10.data.calendar import CalendarSchedule, FoldPlan, build_calendar_schedule, plan_fold
from frem_s10.data.features import build_origin_records, fit_burnin_transform
from frem_s10.data.source_reader import ScopedSourceReader, SourceIdentity
from frem_s10.domain import OriginRecord, Transform
from frem_s10.errors import ChronologyError, ContractError, IntegrityError
from frem_s10.identities import scientific_digest
from frem_s10.metadata import midnight_us

PILOT_END = date(2010, 1, 1)


@dataclass(frozen=True, slots=True)
class PrecisionInputs:
    """Only pilot values; evaluation folds are represented solely by calendar metadata."""

    admission: AdmissionResult
    pilot_calendar: CalendarSchedule
    projected_calendar: CalendarSchedule
    pilot: FoldPlan
    projected_a: FoldPlan
    projected_b: FoldPlan
    records: tuple[OriginRecord, ...]
    transform: Transform
    evidence_scope: str = "SOURCE_VIEW_WITHOUT_SCIENTIFIC_ADMISSION"

    def __post_init__(self) -> None:
        """Bind source view and all three counts to exact independently derived dates."""
        if self.evidence_scope not in ("GENERATED_ENGINEERING", "SOURCE_VIEW_WITHOUT_SCIENTIFIC_ADMISSION"):
            raise ContractError("pilot inputs cannot self-attest scientific source authority")
        panel = self.admission.panel
        if any(day >= PILOT_END for day in panel.dates):
            raise ChronologyError("precision value view includes a post-2009 session")
        if self.pilot_calendar != build_calendar_schedule(panel.dates):
            raise IntegrityError("precision calendar differs from its source view")
        expected = tuple(day for day in self.projected_calendar.dates if day < PILOT_END)
        if panel.dates != expected:
            raise IntegrityError("pilot dates are not the exact projected-calendar prefix")
        if self.pilot != plan_fold(self.pilot_calendar, "PILOT"):
            raise IntegrityError("pilot block plan differs from fixed timestamp construction")
        if self.projected_a != plan_fold(self.projected_calendar, "A") or self.projected_b != plan_fold(
            self.projected_calendar, "B"
        ):
            raise IntegrityError("projected counts are not bound to fixed A/B timestamp plans")
        self._validate_records()

    def _validate_records(self) -> None:
        """Reconstruct target/transform identities only within the already bounded view."""
        if any(record.available_us >= midnight_us(PILOT_END) for record in self.records):
            raise ChronologyError("precision target is unavailable before 2010")
        expected_records = build_origin_records(self.admission.panel, calendar=self.pilot_calendar)
        if tuple(record.record_id for record in self.records) != tuple(r.record_id for r in expected_records):
            raise IntegrityError("precision origin membership differs from the admitted value view")
        # Target values are part of the identity even though origin identity excludes them.
        if any(
            any(
                not np.array_equal(getattr(record, name), getattr(expected, name))
                for name in (
                    "x",
                    "g",
                    "d",
                    "h1_g",
                    "h1_d",
                    "anchor_index",
                    "origin_us",
                    "endpoint_us",
                    "available_us",
                    "feature_available_us",
                )
            )
            for record, expected in zip(self.records, expected_records, strict=True)
        ):
            raise IntegrityError("precision target differs from admitted source construction")
        expected_transform = fit_burnin_transform(self.records)
        if (
            self.transform.record_id != expected_transform.record_id
            or self.transform.record_ids != expected_transform.record_ids
            or any(
                not np.array_equal(getattr(self.transform, name), getattr(expected_transform, name))
                for name in ("feature_mean", "feature_scale", "score_scale")
            )
        ):
            raise IntegrityError("precision transform differs from fixed pre-2008 burn-in")

    @property
    def record_id(self) -> str:
        """Identify source view, transforms and exact timestamp-derived projected counts."""
        return scientific_digest(
            {
                "schema": "s10-precision-inputs-v1",
                "evidence_scope": self.evidence_scope,
                "sources": list(self.admission.source_hashes),
                "calendar": self.pilot_calendar.record_id,
                "projected_calendar": self.projected_calendar.record_id,
                "pilot": self.pilot.record_id,
                "A": self.projected_a.record_id,
                "B": self.projected_b.record_id,
                "transform": self.transform.record_id,
                "values_end_exclusive": PILOT_END.isoformat(),
            }
        )


def prepare_precision_inputs(
    reader: ScopedSourceReader,
    identities: tuple[SourceIdentity, ...],
    *,
    pilot_contract: DataContract,
    projected_sessions: tuple[date, ...],
    projected_calendar_id: str,
    evidence_scope: str = "SOURCE_VIEW_WITHOUT_SCIENTIFIC_ADMISSION",
) -> PrecisionInputs:
    """Filter to the permitted value view before creating a single target.

    A reviewed, consumed precision worker calls this function after resource
    admission. Raw bytes are hash verified by the source reader; out-of-scope
    numeric fields are neither validated as observations nor passed to target
    construction. Projected sample sizes never need evaluation outcome values.
    """
    if evidence_scope not in ("GENERATED_ENGINEERING", "SOURCE_VIEW_WITHOUT_SCIENTIFIC_ADMISSION"):
        raise ContractError("pilot preparation cannot self-attest scientific source authority")
    if pilot_contract.end_exclusive != PILOT_END:
        raise ChronologyError("precision source declaration must end exactly at 2010-01-01")
    if projected_calendar_id != calendar_identity(projected_sessions):
        raise IntegrityError("projected calendar identity mismatch")
    if any(day >= date(2017, 1, 1) for day in projected_sessions):
        raise ChronologyError("projected development dates escape the pre-2017 boundary")
    if pilot_contract.expected_sessions != tuple(day for day in projected_sessions if day < PILOT_END):
        raise IntegrityError("pilot source calendar is not the exact full-calendar prefix")
    projected = build_calendar_schedule(projected_sessions)
    a, b = plan_fold(projected, "A"), plan_fold(projected, "B")
    admitted = admit_sources(reader, identities, contract=pilot_contract)
    pilot_calendar = build_calendar_schedule(admitted.panel.dates)
    pilot = plan_fold(pilot_calendar, "PILOT")
    records = build_origin_records(admitted.panel, calendar=pilot_calendar)
    return PrecisionInputs(
        admitted,
        pilot_calendar,
        projected,
        pilot,
        a,
        b,
        records,
        fit_burnin_transform(records),
        evidence_scope,
    )
