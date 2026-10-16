"""Bounded generated-only orchestration, identity, chronology and accounting evidence."""

from dataclasses import replace
from datetime import date, timedelta
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from frem_s10.data.calendar import build_calendar_schedule
from frem_s10.data.features import build_origin_records, fit_burnin_transform
from frem_s10.domain import ForecastDistribution, OriginRecord, SessionPanel, Transform
from frem_s10.errors import ArtifactIOError, ContractError, IntegrityError, NumericalError
from frem_s10.experiment import replay
from frem_s10.experiment.artifacts import publish_campaign, verify_publication
from frem_s10.experiment.evidence import extract_fold_evidence
from frem_s10.experiment.forecasts import ForecastFrame, ForecastTape, build_forecast_tape
from frem_s10.experiment.matrix import ABLATIONS, FORECAST_ARMS, MODEL_VARIANTS, CellSpec, campaign_matrix
from frem_s10.experiment.replay import (
    CampaignResult,
    CellFailure,
    CellReplayError,
    CellResult,
    ForecastScore,
    ReplayInputs,
    ReplayWindow,
    ScoreCache,
    SessionTiming,
    prepare_campaign,
    replay_campaign,
    replay_cell,
)
from frem_s10.metadata import scheduled_open_us
from frem_s10.portfolio.actions import Valuation
from frem_s10.portfolio.policy import PolicyDecision


def generated_panel(count: int) -> tuple[SessionPanel, tuple[SessionTiming, ...]]:
    """One artificial Monday observation per week, never a real exchange calendar."""
    days = tuple(date(2006, 1, 2) + timedelta(weeks=index) for index in range(count))
    sequence = np.arange(count, dtype=np.float64)
    prices = np.column_stack((100 + sequence / 10, 100 + np.sin(sequence), 100 + np.cos(sequence)))
    dividends = np.zeros((count, 3))
    dividends[::4, 0] = 0.25
    aligned: list[tuple[str | None, str | None, str | None]] = []
    timing: list[SessionTiming] = []
    flat: list[tuple[str, ...]] = []
    for index, day in enumerate(days):
        identity = f"constructed-SPY-event-{index}" if dividends[index, 0] > 0 else None
        aligned.append((identity, None, None))
        group = () if identity is None else (identity,)
        flat.append(group)
        opening = scheduled_open_us(day)
        timing.append(SessionTiming(opening, opening + 6 * 3600 * 1_000_000, (group, (), ())))
    panel = SessionPanel(
        days,
        prices,
        prices * 1.0001,
        dividends,
        ("generated-SPY", "generated-TLT", "generated-GLD"),
        tuple(flat),
        asset_event_ids=tuple(aligned),
    )
    return panel, tuple(timing)


@pytest.fixture(scope="session")
def inputs() -> ReplayInputs:
    panel, timing = generated_panel(574)
    complete = prepare_campaign(panel, timing)
    windows = tuple(ReplayWindow(window.fold_id, window.origin_indices[:2]) for window in complete.windows)
    return replace(complete, windows=windows, evidence_scope="GENERATED_ENGINEERING")


@pytest.fixture(scope="session")
def tape(inputs: ReplayInputs) -> ForecastTape:
    by_index = {record.anchor_index: record for record in inputs.origins}
    frames: list[ForecastFrame] = []
    for index in sorted(index for window in inputs.windows for index in window.origin_indices):
        record = by_index[index]
        for variant in MODEL_VARIANTS:
            arms = FORECAST_ARMS if variant == "primary" else ("contextual_stack",)
            laws: list[tuple[str, ForecastDistribution]] = []
            for arm in arms:
                g = np.array([[1.02, 1.01, 1.005], [0.98, 1.03, 1.02], [1.04, 0.99, 1.01]])
                g += (index % 11) * 0.0001
                if arm == "zero":
                    g = np.ones((3, 3))
                law = ForecastDistribution(
                    f"generated/{variant}/{index}/{arm}", g, np.zeros_like(g), np.full(3, 1 / 3)
                )
                laws.append((arm, law))
            frames.append(
                ForecastFrame(
                    index,
                    record.record_id,
                    record.origin_us + 1,
                    variant,
                    tuple(laws),
                    f"generated-before-{index}-{variant}",
                    f"generated-after-{index}-{variant}",
                )
            )
    return ForecastTape(
        tuple(frames),
        inputs.transform.record_id,
        "GENERATED_ENGINEERING",
        4,
        tuple((variant, f"generated-final-{variant}") for variant in MODEL_VARIANTS),
    )


@pytest.fixture(scope="session")
def campaign(inputs: ReplayInputs, tape: ForecastTape) -> CampaignResult:
    return replay_campaign(inputs, tape)


def test_exact_192_registry_and_no_combined_or_cost_shifted_ablations() -> None:
    matrix = campaign_matrix()
    assert len(matrix) == len({cell.cell_id for cell in matrix}) == 192
    assert [sum(cell.kind == kind for cell in matrix) for kind in ("strategy", "benchmark", "ablation")] == [
        128,
        48,
        16,
    ]
    assert tuple(cell.arm for cell in matrix if cell.fold_id == "A" and cell.kind == "ablation") == ABLATIONS
    assert matrix[0].cell_id == "A/strategy/unconditional/nominal/0"
    for kwargs in ({"cost_bps": 20}, {"policy": "nominal"}):
        base = CellSpec("A", "ablation", "repeat_h1", 10)
        with pytest.raises(ContractError):
            replace(base, **kwargs)


def test_full_192_generated_matrix_completes_and_each_fold_starts_from_cash(campaign: CampaignResult) -> None:
    assert campaign.complete, campaign.failure
    assert len(campaign.results) == 192
    assert campaign.evidence_scope == "GENERATED_ENGINEERING"
    for result in campaign.results:
        assert result.final_state.phase == "FINAL"
        assert result.final_state.initial_cash == 100000.0
        assert result.events[0].cash_before == 100000.0
        np.testing.assert_array_equal(result.events[0].shares_before, np.zeros(3))
        assert len(result.decisions) == len(result.scores) == len(result.log_increments) == 2
        np.testing.assert_allclose(
            result.log_increments.sum(), np.log(result.final_state.nav / 100000), atol=2e-15
        )
        assert sum(event.kind == "FINAL_LIQUIDATION" for event in result.events) == 1
        assert 0 <= result.maximum_drawdown < 1
        assert result.events[-1].after_id == result.final_state.record_id
        with pytest.raises(ValueError):
            result.scores[0].loss_outcomes.setflags(write=True)


def result_named(
    campaign: CampaignResult, fold: str, kind: str, arm: str, policy: str = "robust"
) -> CellResult:
    return next(item for item in campaign.results if item.spec == CellSpec(fold, kind, arm, 10, policy))


def test_frozen_state_freezes_entire_law_separately_per_fold_and_controls_share_laws(
    campaign: CampaignResult, tape: ForecastTape
) -> None:
    frozen_ids: list[str] = []
    for fold in ("A", "B"):
        main = result_named(campaign, fold, "strategy", "contextual_stack")
        frozen = result_named(campaign, fold, "ablation", "frozen_state")
        nominal = result_named(campaign, fold, "strategy", "contextual_stack", "nominal")
        assert tuple(item.forecast_id for item in main.decisions) == tuple(
            item.forecast_id for item in nominal.decisions
        )
        assert main.decisions[0].forecast_id != main.decisions[1].forecast_id
        assert (
            frozen.decisions[0].forecast_id
            == frozen.decisions[1].forecast_id
            == main.decisions[0].forecast_id
        )
        assert frozen.decisions[1].forecast_decision_us < frozen.decisions[1].decision_us
        frozen_ids.append(frozen.decisions[0].forecast_id)
        for ablation in ("force_liquidate", "no_exact_hold"):
            control = result_named(campaign, fold, "ablation", ablation)
            assert [item.forecast_id for item in control.decisions] == [
                item.forecast_id for item in main.decisions
            ]
            assert [score.energy for score in control.scores] == [score.energy for score in main.scores]
        first, second = main.decisions
        assert not np.array_equal(
            tape.frame(first.anchor_index).get("contextual_stack").g,
            tape.frame(second.anchor_index).get("contextual_stack").g,
        )
    assert len(set(frozen_ids)) == 2


def test_cl15_sells_after_entering_dividend_before_reentry_and_final_exit_once(
    campaign: CampaignResult,
) -> None:
    result = result_named(campaign, "A", "ablation", "force_liquidate")
    sales = [event for event in result.events if event.kind == "FORCED_SALE"]
    assert len(sales) == 2
    for sale in sales:
        position = result.events.index(sale)
        opening, fill = result.events[position - 1], result.events[position + 1]
        assert opening.kind == "OPEN_AND_DIVIDENDS" and fill.kind == "REBALANCE"
        assert (opening.timestamp_us, opening.phase_order) < (sale.timestamp_us, sale.phase_order)
        assert sale.timestamp_us + 2 == fill.timestamp_us
        np.testing.assert_array_equal(sale.shares_after, np.zeros(3))
        np.testing.assert_array_equal(sale.dividends, np.zeros(3))
        np.testing.assert_array_equal(fill.dividends, np.zeros(3))
    assert result.final_state.fees == pytest.approx(sum(event.fee for event in result.events), abs=1e-8)


def test_full_kernel_clock_uses_direct_h4_maturity_and_future_changes_cannot_alter_current_forecasts() -> (
    None
):
    panel, _ = generated_panel(116)
    calendar = build_calendar_schedule(panel.dates)
    records = build_origin_records(panel, calendar=calendar)
    transform = fit_burnin_transform(records)
    selected = tuple(record.anchor_index for record in records if record.origin_us >= 1199145600000000)[:2]
    first = next(record for record in records if record.anchor_index == selected[0])
    original = build_forecast_tape(records, transform, retained_anchor_indices=selected)
    changed_records = tuple(
        replace(record, g=record.g * 1.2, d=record.d + 0.003)
        if record.available_us >= first.origin_us + 1
        else record
        for record in records
    )
    changed = build_forecast_tape(changed_records, transform, retained_anchor_indices=selected[:1])
    for variant in MODEL_VARIANTS:
        before = original.frame(selected[0], variant)
        after = changed.frame(selected[0], variant)
        assert [law.record_id for _, law in before.laws] == [law.record_id for _, law in after.laws]
        assert before.state_after_id == after.state_after_id
    h1 = original.frame(selected[0], "repeat_h1").get("contextual_stack")
    mature = tuple(record for record in records if record.available_us < first.origin_us + 1)
    assert len(h1.g) == len(mature)
    for index, record in enumerate(mature):
        assert record.h1_g is not None and record.h1_d is not None
        np.testing.assert_allclose(h1.g[index], record.h1_g**4)
        np.testing.assert_allclose(
            h1.d[index], record.h1_d * (1 + record.h1_g + record.h1_g**2 + record.h1_g**3)
        )


def test_replay_rejects_wrong_transform_missing_frame_wrong_source_ownership_and_shrunken_production_fold(
    inputs: ReplayInputs, tape: ForecastTape
) -> None:
    with pytest.raises(IntegrityError):
        replay_campaign(inputs, replace(tape, transform_id="wrong"))
    with pytest.raises(ContractError):
        replace(tape, frames=tape.frames[:-1])
    with pytest.raises(ContractError):
        replace(inputs, evidence_scope="KERNEL_REPLAY")
    with pytest.raises(ContractError):
        replace(inputs, panel=replace(inputs.panel, asset_event_ids=None))
    slot = next(index for index, item in enumerate(inputs.timing) if item.dividend_event_ids[0])
    changed = list(inputs.timing)
    ids = changed[slot].dividend_event_ids[0]
    changed[slot] = replace(changed[slot], dividend_event_ids=((), ids, ()))
    with pytest.raises(IntegrityError):
        replace(inputs, timing=tuple(changed))


def test_late_cell_failure_retains_all_previous_cell_evidence(
    inputs: ReplayInputs, tape: ForecastTape, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = replay.replay_cell
    calls = 0

    def interrupted(
        inputs: ReplayInputs, tape: ForecastTape, cell: CellSpec, *, _score_cache: ScoreCache | None = None
    ) -> CellResult:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise NumericalError("constructed late failure")
        return original(inputs, tape, cell, _score_cache=_score_cache)

    monkeypatch.setattr(replay, "replay_cell", interrupted)
    failed = replay_campaign(inputs, tape)
    assert not failed.complete and failed.failure is not None
    assert len(failed.results) == 3
    assert failed.failure.cell_id == campaign_matrix()[3].cell_id
    assert failed.failure.code == "NUMERICAL"
    assert failed.planned_cell_ids == tuple(cell.cell_id for cell in campaign_matrix())
    assert all(item.final_state.phase == "FINAL" for item in failed.results)


def test_one_cell_replay_is_bitwise_deterministic(inputs: ReplayInputs, tape: ForecastTape) -> None:
    spec = CellSpec("B", "benchmark", "buy_hold100", 10)
    first, second = replay_cell(inputs, tape, spec), replay_cell(inputs, tape, spec)
    assert first.final_state.record_id == second.final_state.record_id
    assert tuple(event.after_id for event in first.events) == tuple(event.after_id for event in second.events)
    np.testing.assert_array_equal(first.log_increments, second.log_increments)
    assert first.decisions[1].selection.action.is_hold


def test_real_kernel_to_policy_to_ledger_bridge_uses_generated_inputs_only() -> None:
    panel, timing = generated_panel(116)
    calendar = build_calendar_schedule(panel.dates)
    records = build_origin_records(panel, calendar=calendar)
    transform = fit_burnin_transform(records)
    selected = tuple(record.anchor_index for record in records if record.origin_us >= 1199145600000000)
    first, second = selected[0], selected[4]
    windows = (ReplayWindow("A", (first,)), ReplayWindow("B", (second,)))
    inputs = ReplayInputs(panel, calendar, timing, records, transform, windows, "GENERATED_ENGINEERING")
    kernel = build_forecast_tape(records, transform, retained_anchor_indices=(first, second))
    # Labels describe this deliberately tiny generated fixture, not a full study.
    tape = replace(kernel, evidence_scope="GENERATED_ENGINEERING")
    for fold in ("A", "B"):
        result = replay_cell(inputs, tape, CellSpec(fold, "strategy", "contextual_stack", 10))
        assert result.final_state.phase == "FINAL"
        assert len(result.decisions) == len(result.log_increments) == 1
        assert result.decisions[0].selection.certificate.candidate_count == 609
        assert (
            result.decisions[0].forecast_id
            == tape.frame(result.decisions[0].anchor_index).get("contextual_stack").record_id
        )
        assert np.isfinite(result.scores[0].energy)


def test_failed_a4_selection_preserves_completed_source_credit_and_forced_sale(
    inputs: ReplayInputs,
    tape: ForecastTape,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_policy(
        valuation: Valuation,
        forecast: ForecastDistribution,
        *,
        one_way_cost: float,
        robust: bool = True,
        include_hold: bool = True,
    ) -> PolicyDecision:
        raise NumericalError("constructed failure after precommitted sale")

    monkeypatch.setattr(replay, "select_policy", failed_policy)
    with pytest.raises(CellReplayError) as failure:
        replay_cell(inputs, tape, CellSpec("A", "ablation", "force_liquidate", 10))
    partial = failure.value.partial
    assert tuple(event.kind for event in partial.events) == ("OPEN_AND_DIVIDENDS", "FORCED_SALE")
    assert partial.final_state.phase == "OPEN"
    assert partial.final_state.forced_sale_us == partial.final_state.open_us
    assert not partial.decisions
    assert partial.events[-1].after_id == partial.final_state.record_id


def test_shared_scores_are_computed_once_and_match_uncached_replay(
    inputs: ReplayInputs,
    tape: ForecastTape,
    campaign: CampaignResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = replay._score
    calls = 0

    def counted(record: OriginRecord, law: ForecastDistribution, transform: Transform) -> ForecastScore:
        nonlocal calls
        calls += 1
        return original(record, law, transform)

    monkeypatch.setattr(replay, "_score", counted)
    steps = tuple(islice(replay.campaign_steps(inputs, tape), 8))
    assert len(steps) == 8 and all(isinstance(step, CellResult) for step in steps)
    assert calls == 8  # Four laws * two origins, reused by nominal and robust.
    for fold in ("A", "B"):
        cached = result_named(campaign, fold, "ablation", "frozen_state")
        uncached = replay_cell(inputs, tape, cached.spec)
        for left, right in zip(cached.scores, uncached.scores, strict=True):
            assert left.energy.hex() == right.energy.hex()
            np.testing.assert_array_equal(left.marginal_crps, right.marginal_crps)
        assert cached.scores[0].origin_id != cached.scores[1].origin_id
        assert cached.scores[0] is not cached.scores[1]


def test_endpoint_adapter_rejects_engineering_incomplete_small_and_unpaired_evidence(
    campaign: CampaignResult,
) -> None:
    with pytest.raises(ContractError, match="generated engineering"):
        extract_fold_evidence(campaign)
    with pytest.raises(IntegrityError, match="completed 192"):
        extract_fold_evidence(replace(campaign, results=campaign.results[:-1]))
    # Deliberate hostile relabeling still cannot bypass the full-N protocol.
    relabeled = replace(campaign, evidence_scope="KERNEL_REPLAY")
    with pytest.raises(ContractError, match=r"24\.\.1000"):
        extract_fold_evidence(relabeled)
    results = list(relabeled.results)
    slot = next(
        index
        for index, item in enumerate(results)
        if item.spec == CellSpec("A", "strategy", "global_stack", 10)
    )
    altered = results[slot]
    decisions = (
        replace(altered.decisions[0], decision_us=altered.decisions[0].decision_us + 1),
        *altered.decisions[1:],
    )
    results[slot] = replace(altered, decisions=decisions)
    with pytest.raises(IntegrityError, match="timestamps"):
        extract_fold_evidence(replace(relabeled, results=tuple(results)))
    results = list(relabeled.results)
    slot = next(
        index
        for index, item in enumerate(results)
        if item.spec == CellSpec("A", "strategy", "contextual_stack", 10, "nominal")
    )
    altered = results[slot]
    wrong = "different-law-consistently-relinked"
    first = altered.decisions[0]
    decision = replace(first, forecast_id=wrong, selection=replace(first.selection, forecast_id=wrong))
    results[slot] = replace(
        altered,
        decisions=(decision, *altered.decisions[1:]),
        scores=(replace(altered.scores[0], forecast_id=wrong), *altered.scores[1:]),
    )
    with pytest.raises(IntegrityError, match="identical issued"):
        extract_fold_evidence(replace(relabeled, results=tuple(results)))


def test_discrete_interval_endpoint_ties_and_raw_diagnostics_have_exact_units(inputs: ReplayInputs) -> None:
    law = ForecastDistribution(
        "constructed-singleton", np.array([[0.99, 1.02, 1.01]]), np.zeros((1, 3)), np.ones(1)
    )
    outcome = replace(inputs.origins[0], g=law.g[0], d=law.d[0])
    score = replay._score(outcome, law, inputs.transform)
    assert score.energy == score.variogram == score.mean_squared_error == 0
    np.testing.assert_array_equal(score.raw_marginal_crps, np.zeros(3))
    np.testing.assert_array_equal(score.raw_mean_squared_errors, np.zeros(3))
    np.testing.assert_array_equal(score.interval_lower_raw, np.broadcast_to(outcome.r, (3, 3)))
    np.testing.assert_array_equal(score.interval_upper_raw, score.interval_lower_raw)
    np.testing.assert_array_equal(score.interval_coverage, np.ones((3, 3)))
    np.testing.assert_array_equal(score.interval_outcome_boundary_ties, np.ones((3, 3)))


@pytest.mark.parametrize("corruption", ("changed_bytes", "missing_cell", "unexpected_cell"))
def test_streamed_artifacts_retain_failed_partial_typed_arrays_and_detect_inventory_corruption(
    inputs: ReplayInputs,
    tape: ForecastTape,
    campaign: CampaignResult,
    tmp_path: Path,
    corruption: str,
) -> None:
    failure = CellFailure(
        campaign.results[1].spec.cell_id,
        "CONSTRUCTED_FAILURE",
        "deliberate failure after generated cell computation",
        campaign.results[1],
    )
    receipt = publish_campaign(
        tmp_path, inputs, tape, experiment_id="generated-publication", steps=(campaign.results[0], failure)
    )
    assert not receipt.complete and receipt.completed_cells == 1
    assert receipt.evidence_scope == "GENERATED_ENGINEERING"
    manifest = verify_publication(tmp_path, receipt.manifest_sha256)
    assert manifest["scientific_support_claimed"] is False
    assert manifest["failure_cell_id"] == failure.cell_id
    saved = (tmp_path / "cell-001.json").read_text()
    assert '"record_type":"CellFailure"' in saved and '"partial":{"fields"' in saved
    assert '"dtype":"<f8"' in saved and '"shape":[3]' in saved and '"unit":"shares"' in saved
    assert '"evidence_scope":"GENERATED_ENGINEERING"' in saved
    with pytest.raises(ArtifactIOError):
        publish_campaign(tmp_path, inputs, tape, experiment_id="overwrite", steps=())
    if corruption == "changed_bytes":
        (tmp_path / "cell-000.json").write_bytes((tmp_path / "cell-000.json").read_bytes() + b" ")
    elif corruption == "missing_cell":
        (tmp_path / "cell-000.json").unlink()
    else:
        (tmp_path / "cell-999.json").write_text("{}")
    with pytest.raises(IntegrityError):
        verify_publication(tmp_path, receipt.manifest_sha256)


def test_truncated_artifact_stream_writes_explicit_failure_without_claiming_completion(
    inputs: ReplayInputs,
    tape: ForecastTape,
    campaign: CampaignResult,
    tmp_path: Path,
) -> None:
    receipt = publish_campaign(
        tmp_path, inputs, tape, experiment_id="generated-truncation", steps=(campaign.results[0],)
    )
    assert not receipt.complete and receipt.completed_cells == 1
    assert '"code":"INCOMPLETE_MATRIX"' in (tmp_path / "cell-001.json").read_text()
    assert verify_publication(tmp_path, receipt.manifest_sha256)["complete"] is False


def test_complete_generated_inventory_has_exact_192_hashes_and_no_empirical_claim(
    inputs: ReplayInputs,
    tape: ForecastTape,
    campaign: CampaignResult,
    tmp_path: Path,
) -> None:
    receipt = publish_campaign(
        tmp_path, inputs, tape, experiment_id="generated-complete-matrix", steps=campaign.results
    )
    assert receipt.complete and receipt.completed_cells == 192
    manifest = verify_publication(tmp_path, receipt.manifest_sha256)
    assert manifest["evidence_scope"] == "GENERATED_ENGINEERING"
    assert manifest["scientific_support_claimed"] is False
    assert manifest["failure_cell_id"] is None
    assert len(tuple(tmp_path.iterdir())) == 195 + len(tape.frames)
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) < 16 * 1024 * 1024


def test_tape_binary_component_identity_is_copy_stable_and_sensitive_to_one_bit(tape: ForecastTape) -> None:
    frame = tape.frames[0]
    arm, law = frame.laws[0]
    copied = replace(law, g=law.g.copy(), d=law.d.copy(), probabilities=law.probabilities.copy())
    equal_frame = replace(frame, laws=((arm, copied), *frame.laws[1:]))
    assert replace(tape, frames=(equal_frame, *tape.frames[1:])).record_id == tape.record_id
    g = law.g.copy()
    g[0, 0] = np.nextafter(g[0, 0], np.inf)
    changed_frame = replace(frame, laws=((arm, replace(law, g=g)), *frame.laws[1:]))
    assert replace(tape, frames=(changed_frame, *tape.frames[1:])).record_id != tape.record_id


def test_kernel_input_reconstructs_payloads_even_when_tampering_preserves_ids() -> None:
    panel, timing = generated_panel(574)
    complete = prepare_campaign(panel, timing)
    original = complete.origins[0]
    for field in ("x", "g", "d", "h1_g", "h1_d"):
        values = getattr(original, field)
        assert values is not None
        changed = replace(original, **{field: values + 0.0001})
        with pytest.raises(IntegrityError, match="numeric payload"):
            replace(complete, origins=(changed, *complete.origins[1:]))
    with pytest.raises(IntegrityError, match="burn-in reconstruction"):
        replace(
            complete,
            transform=replace(complete.transform, feature_scale=complete.transform.feature_scale * 2),
        )
