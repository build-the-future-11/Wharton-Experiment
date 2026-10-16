"""Pure complete-campaign adapter to fixed primary and secondary statistics."""

from dataclasses import dataclass, fields, replace

import numpy as np

from frem_s10.analysis.ablations import ABLATION_IDS, AblationFold, AblationResult, assess_ablations
from frem_s10.analysis.endpoints import FoldEvidence, StudyResult, assess_study
from frem_s10.arrays import IntArray
from frem_s10.errors import ContractError, IntegrityError

from .evidence import _axes, _fold
from .matrix import ABLATIONS, CellSpec, campaign_matrix
from .replay import CampaignResult, CellResult, ForecastScore


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    """Paired numerical results; source authority and scientific admission stay external."""

    forecast_tape_id: str
    cell_ids: tuple[str, ...]
    folds: tuple[FoldEvidence, FoldEvidence]
    ablation_folds: tuple[AblationFold, AblationFold]
    primary: StudyResult
    ablations: AblationResult
    bootstrap: tuple[tuple[str, int, IntArray], ...]
    evidence_scope: str
    source_authority_approved: bool = False
    authorizes_campaign: bool = False

    def __post_init__(self) -> None:
        """Do not let numerical predicate support become an empirical approval."""
        if (
            self.evidence_scope
            not in ("GENERATED_ENGINEERING_ANALYSIS", "COMPUTATIONAL_STATISTICS_SOURCE_AUTHORITY_UNVERIFIED")
            or self.source_authority_approved is not False
            or self.authorizes_campaign is not False
            or self.primary.evidence_scope != self.evidence_scope
            or self.ablations.evidence_scope != self.evidence_scope
        ):
            raise IntegrityError("analysis bundle cannot self-attest scientific source authority")


def _paired_cells(campaign: CampaignResult) -> dict[str, CellResult]:
    expected = tuple(spec.cell_id for spec in campaign_matrix())
    if not campaign.complete or campaign.planned_cell_ids != expected:
        raise IntegrityError("analysis requires the exact completed 192-cell matrix")
    if not isinstance(campaign.forecast_tape_id, str) or not campaign.forecast_tape_id:
        raise IntegrityError("analysis requires the retained forecast tape identity")
    cells = {result.spec.cell_id: result for result in campaign.results}
    laws: dict[tuple[str, str, str, int], str] = {}
    scores: dict[tuple[str, int, str], ForecastScore] = {}
    for fold in ("A", "B"):
        primary = cells[CellSpec(fold, "strategy", "contextual_stack", 10).cell_id]
        axes = _axes(primary)
        if not axes or tuple(index for _, _, index in axes) != tuple(range(axes[0][2], axes[-1][2] + 1, 4)):
            raise IntegrityError("complete four-anchor progression required")
        for result in campaign.results:
            if result.spec.fold_id == fold:
                if _axes(result) != axes:
                    raise IntegrityError("all campaign cells must share exact paired fold axes")
                _shared_laws(result, axes[0][0], axes[0][2], laws, scores)
    return cells


def _shared_laws(
    result: CellResult,
    first_decision: int,
    first_anchor: int,
    laws: dict[tuple[str, str, str, int], str],
    scores: dict[tuple[str, int, str], ForecastScore],
) -> None:
    spec = result.spec
    frozen = spec.kind == "ablation" and spec.arm == "frozen_state"
    for decision, score in zip(result.decisions, result.scores, strict=True):
        issued_us = first_decision if frozen else decision.decision_us
        if decision.forecast_decision_us != issued_us or not decision.forecast_id:
            raise IntegrityError("forecast issue clock differs from declared current/frozen law")
        key = (
            spec.fold_id,
            spec.variant,
            spec.forecast_arm,
            first_anchor if frozen else decision.anchor_index,
        )
        if key in laws and laws[key] != decision.forecast_id:
            raise IntegrityError("shared forecast law differs across policy, cost or fixed ablation cells")
        laws[key] = decision.forecast_id
        score_key = (spec.fold_id, decision.anchor_index, decision.forecast_id)
        previous = scores.get(score_key)
        if previous is not None and any(
            not np.array_equal(getattr(previous, field.name), getattr(score, field.name))
            for field in fields(ForecastScore)
        ):
            raise IntegrityError("the same forecast and outcome have inconsistent saved scores")
        scores[score_key] = score


def _ablation_fold(cells: dict[str, CellResult], fold: FoldEvidence) -> AblationFold:
    rows = tuple(cells[CellSpec(fold.fold_id, "ablation", name, 10).cell_id] for name in ABLATIONS)
    return AblationFold(
        fold.fold_id,
        fold.decision_us,
        fold.origin_ids,
        fold.energy[:, 0],
        tuple(
            (identity, np.asarray([score.energy for score in row.scores], dtype=np.float64))
            for identity, row in zip(ABLATION_IDS, rows, strict=True)
        ),
    )


def analyze_campaign(campaign: CampaignResult) -> AnalysisBundle:
    """Assess complete paired retained numbers without source approval or model execution.

    Generated full-size evidence is allowed only as explicitly generated
    analysis. Source/freeze authority, event reconstruction and supervised
    execution are separate gates; internal axis checks do not certify them.
    """
    if campaign.evidence_scope not in ("GENERATED_ENGINEERING", "KERNEL_REPLAY"):
        raise ContractError("only explicitly scoped complete replay evidence can be analyzed")
    cells = _paired_cells(campaign)
    folds = (_fold(cells, "A"), _fold(cells, "B"))
    ablation_folds = (_ablation_fold(cells, folds[0]), _ablation_fold(cells, folds[1]))
    primary = assess_study(folds, integrity_verified=True, complete_matrix=True)
    ablations = assess_ablations(ablation_folds)
    if len(primary.bootstrap) != len(ablations.bootstrap) or any(
        (left[0], left[1]) != (right[0], right[1]) or not np.array_equal(left[2], right[2])
        for left, right in zip(primary.bootstrap, ablations.bootstrap, strict=True)
    ):
        raise IntegrityError("primary and secondary inference must retain identical fixed bootstrap streams")
    scope = (
        "GENERATED_ENGINEERING_ANALYSIS"
        if campaign.evidence_scope == "GENERATED_ENGINEERING"
        else "COMPUTATIONAL_STATISTICS_SOURCE_AUTHORITY_UNVERIFIED"
    )
    return AnalysisBundle(
        campaign.forecast_tape_id,
        campaign.planned_cell_ids,
        folds,
        ablation_folds,
        replace(primary, evidence_scope=scope),
        replace(ablations, evidence_scope=scope),
        primary.bootstrap,
        scope,
    )
