"""Constructed retained-type fixtures for analysis axes, never a simulated market run."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from frem_s10.domain import ForecastDistribution
from frem_s10.errors import ContractError, IntegrityError
from frem_s10.experiment.analysis import analyze_campaign
from frem_s10.experiment.matrix import ABLATIONS, CellSpec, campaign_matrix
from frem_s10.experiment.replay import CampaignResult, CellResult, DecisionRecord, ForecastScore
from frem_s10.metadata import midnight_us
from frem_s10.portfolio.actions import Valuation
from frem_s10.portfolio.benchmarks import select_benchmark
from frem_s10.portfolio.ledger import LedgerEvent, MarkKind, NavMark, initial_ledger


def cell(spec: CellSpec) -> CellResult:
    """Construct explicit cash marks and labeled scores to test extraction alone."""
    count = 24 if spec.fold_id == "A" else 25
    day = date(2010, 1, 4) if spec.fold_id == "A" else date(2013, 1, 7)
    times = tuple(midnight_us(day) + index * 28 * 86_400_000_000 for index in range(count))
    state = replace(
        initial_ledger(cell_id=spec.cell_id, start_us=times[0] - 1),
        phase="FINAL",
        cursor_us=times[-1] + 2,
    )
    policy = select_benchmark(
        "cash",
        Valuation(np.zeros(3), 100000.0),
        ForecastDistribution("constructed-law", np.ones((1, 3)), np.zeros((1, 3)), np.ones(1)),
        one_way_cost=0.0,
        initial_entry=True,
    )
    decisions = []
    scores = []
    frozen = spec.kind == "ablation" and spec.arm == "frozen_state"
    for index, timestamp in enumerate(times):
        anchor = 100 + 4 * index
        law = f"{spec.fold_id}/{spec.variant}/{spec.forecast_arm}/{100 if frozen else anchor}"
        origin = f"{spec.fold_id}/origin/{anchor}"
        event = LedgerEvent(
            f"{spec.cell_id}/{index}",
            timestamp + 1,
            "REBALANCE",
            "constructed-before",
            "constructed-after",
            np.zeros(3),
            np.zeros(3),
            0.0,
            0.0,
            100000.0,
            100000.0,
            0.0,
            np.zeros(3),
            np.zeros(3),
            100000.0,
            100000.0,
        )
        decisions.append(
            DecisionRecord(
                anchor,
                origin,
                law,
                times[0] if frozen else timestamp,
                timestamp,
                "constructed-valuation",
                replace(policy, forecast_id=law),
                event,
            )
        )
        energy = 0.8 if spec.forecast_arm == "contextual_stack" else 1.0
        if (
            spec.kind == "ablation"
            and spec.arm not in ("force_liquidate", "no_exact_hold")
            and (not frozen or index > 0)
        ):
            energy += (ABLATIONS.index(spec.arm) + 1) / 100
        scores.append(
            ForecastScore(
                origin,
                law,
                anchor,
                energy,
                0.0,
                np.zeros(3),
                0.0,
                np.zeros(4),
                np.zeros(4),
                np.zeros(3),
                np.zeros(3),
                np.zeros((3, 3)),
                np.zeros((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
            )
        )
    initial = NavMark(times[0] - 1, MarkKind.INITIAL_CAPITAL, 100000.0, "constructed-initial")
    terminal = NavMark(times[-1] + 2, MarkKind.FINAL_LIQUIDATION, 100000.0, state.record_id)
    boundaries = (
        initial,
        *(NavMark(t, MarkKind.LIQUIDATABLE_BOUNDARY, 100000.0, "constructed-boundary") for t in times[1:]),
        terminal,
    )
    return CellResult(
        spec,
        state,
        tuple(item.fill for item in decisions),
        tuple(decisions),
        tuple(scores),
        (initial, terminal),
        boundaries,
    )


@pytest.fixture(scope="module")
def campaign() -> CampaignResult:
    matrix = campaign_matrix()
    return CampaignResult(
        tuple(cell(spec) for spec in matrix),
        None,
        tuple(spec.cell_id for spec in matrix),
        "constructed-tape",
        "GENERATED_ENGINEERING",
    )


def modify(campaign: CampaignResult, spec: CellSpec, changed: CellResult) -> CampaignResult:
    return replace(campaign, results=tuple(changed if row.spec == spec else row for row in campaign.results))


def test_analysis_bundle_has_exact_axes_ablation_order_and_shared_streams(campaign: CampaignResult) -> None:
    bundle = analyze_campaign(campaign)
    assert bundle.evidence_scope == "GENERATED_ENGINEERING_ANALYSIS"
    assert bundle.primary.evidence_scope == bundle.ablations.evidence_scope == bundle.evidence_scope
    assert not bundle.source_authority_approved and not bundle.authorizes_campaign
    assert bundle.primary.h1 is not None and bundle.primary.h1.status == "PASS"
    assert bundle.primary.h2 is not None and bundle.primary.h2.status == "FAIL"
    assert bundle.primary.disposition == "MIXED"
    for fold in bundle.folds:
        count = 24 if fold.fold_id == "A" else 25
        assert fold.origin_ids == tuple(f"{fold.fold_id}/origin/{100 + 4 * i}" for i in range(count))
        np.testing.assert_array_equal(fold.energy, np.repeat([[0.8, 1.0, 1.0]], count, axis=0))
        np.testing.assert_array_equal(fold.log_increments, np.zeros((count, 5)))
        np.testing.assert_array_equal(fold.terminal_wealth, np.full(5, 100000.0))
    expected = {
        "A1": 0.01,
        "A2": 0.02 * 47 / 49,
        "A3": 0.03,
        "A4": 0,
        "A5": 0,
        "A6": 0.06,
        "A7": 0.07,
        "A8": 0.08,
    }
    for row in bundle.ablations.comparisons:
        assert row.pooled_mean == pytest.approx(expected[row.ablation_id])
        assert row.raw_p == (1 if row.ablation_id in ("A4", "A5") else 1 / 5001)
    for primary, secondary in zip(bundle.primary.bootstrap, bundle.ablations.bootstrap, strict=True):
        assert primary[:2] == secondary[:2]
        np.testing.assert_array_equal(primary[2], secondary[2])
    with pytest.raises(IntegrityError):
        replace(bundle, source_authority_approved=True)


@pytest.mark.parametrize("defect", ["axes", "law", "score", "issued"])
def test_unreported_control_cell_cannot_hide_bad_pairing(campaign: CampaignResult, defect: str) -> None:
    spec = CellSpec("B", "strategy", "zero", 40, "nominal")
    original = next(row for row in campaign.results if row.spec == spec)
    decision, score = original.decisions[0], original.scores[0]
    if defect == "axes":
        decision = replace(decision, decision_us=decision.decision_us + 1)
    elif defect == "issued":
        decision = replace(decision, forecast_decision_us=decision.forecast_decision_us - 1)
    elif defect == "score":
        score = replace(score, energy=score.energy + 0.2)
    else:
        decision = replace(
            decision, forecast_id="wrong", selection=replace(decision.selection, forecast_id="wrong")
        )
        score = replace(score, forecast_id="wrong")
    changed = replace(
        original, decisions=(decision, *original.decisions[1:]), scores=(score, *original.scores[1:])
    )
    with pytest.raises(IntegrityError):
        analyze_campaign(modify(campaign, spec, changed))


def test_incomplete_or_unsupported_scope_never_produces_a_bundle(campaign: CampaignResult) -> None:
    with pytest.raises(IntegrityError, match="192"):
        analyze_campaign(replace(campaign, results=campaign.results[:-1]))
    with pytest.raises(ContractError, match="scoped"):
        analyze_campaign(replace(campaign, evidence_scope="REAL_ADMITTED"))
    unverified = analyze_campaign(replace(campaign, evidence_scope="KERNEL_REPLAY"))
    assert unverified.evidence_scope == "COMPUTATIONAL_STATISTICS_SOURCE_AUTHORITY_UNVERIFIED"
    assert not unverified.source_authority_approved and not unverified.authorizes_campaign
