"""Exact dated adapters from retained ledgers to the fixed endpoint statistics."""

import numpy as np

from frem_s10.analysis.endpoints import FoldEvidence
from frem_s10.errors import ContractError, IntegrityError

from .matrix import CellSpec, campaign_matrix
from .replay import CampaignResult, CellResult


def _axes(result: CellResult) -> tuple[tuple[int, str, int], ...]:
    if result.final_state.phase != "FINAL" or result.final_state.cell_id != result.spec.cell_id:
        raise IntegrityError("endpoint input is not its declared terminal ledger")
    if len(result.decisions) != len(result.scores) or len(result.log_increments) != len(result.decisions):
        raise IntegrityError("score and ledger block counts differ")
    for decision, score in zip(result.decisions, result.scores, strict=True):
        if (score.origin_id, score.anchor_index, score.forecast_id) != (
            decision.origin_id,
            decision.anchor_index,
            decision.forecast_id,
        ) or decision.selection.forecast_id != decision.forecast_id:
            raise IntegrityError("dated score is not linked to the law used for its decision")
    return tuple((item.decision_us, item.origin_id, item.anchor_index) for item in result.decisions)


def _fold(cells: dict[str, CellResult], fold_id: str) -> FoldEvidence:
    score_specs = tuple(
        CellSpec(fold_id, "strategy", arm, 10) for arm in ("contextual_stack", "global_stack", "equal")
    )
    wealth_specs = (
        score_specs[0],
        CellSpec(fold_id, "benchmark", "equal100", 10),
        CellSpec(fold_id, "benchmark", "buy_hold100", 10),
        CellSpec(fold_id, "benchmark", "equal80", 10),
        CellSpec(fold_id, "strategy", "contextual_stack", 10, "nominal"),
    )
    scores = tuple(cells[spec.cell_id] for spec in score_specs)
    wealth = tuple(cells[spec.cell_id] for spec in wealth_specs)
    expected = _axes(scores[0])
    if any(_axes(result) != expected for result in (*scores, *wealth)):
        raise IntegrityError("endpoint cells do not share exact decision timestamps and origin identities")
    if tuple(item.forecast_id for item in wealth[0].decisions) != tuple(
        item.forecast_id for item in wealth[-1].decisions
    ):
        raise IntegrityError("robust and nominal comparisons require the identical issued forecast laws")
    return FoldEvidence(
        fold_id,
        np.asarray([stamp for stamp, _, _ in expected], dtype=np.int64),
        tuple(identity for _, identity, _ in expected),
        np.column_stack([[score.energy for score in result.scores] for result in scores]),
        np.column_stack([result.log_increments for result in wealth]),
        np.asarray([result.final_state.nav for result in wealth], dtype=np.float64),
        np.asarray([result.maximum_drawdown for result in wealth], dtype=np.float64),
    )


def extract_fold_evidence(campaign: CampaignResult) -> tuple[FoldEvidence, FoldEvidence]:
    """Require the full kernel campaign and exact paired columns before assessment.

    Tiny generated matrix tests intentionally cannot be relabeled as a
    scientific endpoint result. The supervisor separately verifies source,
    execution, freeze and artifact integrity before interpreting these rows.
    """
    expected = tuple(spec.cell_id for spec in campaign_matrix())
    if not campaign.complete or campaign.planned_cell_ids != expected:
        raise IntegrityError("endpoint assessment requires the exact completed 192-cell matrix")
    if campaign.evidence_scope != "KERNEL_REPLAY":
        raise ContractError("generated engineering replay is not scientific endpoint evidence")
    cells = {result.spec.cell_id: result for result in campaign.results}
    return _fold(cells, "A"), _fold(cells, "B")
