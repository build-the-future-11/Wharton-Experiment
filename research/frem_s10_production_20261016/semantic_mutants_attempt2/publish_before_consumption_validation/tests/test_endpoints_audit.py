"""Independent fixed-fold endpoint/claim checks on constructed evidence only."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from frem_s10.analysis.endpoints import FoldEvidence, assess_study
from frem_s10.errors import ContractError, NumericalError
from frem_s10.metadata import midnight_us


def fold(fold_id: str, count: int = 24) -> FoldEvidence:
    start = date(2010, 1, 4) if fold_id == "A" else date(2013, 1, 7)
    dates = midnight_us(start) + np.arange(count, dtype=np.int64) * (28 * 86_400_000_000)
    wealth = np.array([120000.0, 105000.0, 104000.0, 99000.0, 110000.0])
    increments = np.repeat((np.log(wealth / 100000) / count)[None, :], count, axis=0)
    energy = np.repeat(np.array([[0.8, 1.0, 1.0]]), count, axis=0)
    return FoldEvidence(
        fold_id,
        dates,
        tuple(f"{fold_id}/{i}" for i in range(count)),
        energy,
        increments,
        wealth,
        np.array([0.1, 0.2, 0.3, 0.2, 0.15]),
    )


def test_fixed_required_comparisons_and_retrospective_claim_scope() -> None:
    result = assess_study((fold("A"), fold("B", 25)), integrity_verified=True, complete_matrix=True)
    assert result.disposition == "SUPPORTED"
    assert result.h1 is not None and result.h2 is not None
    assert result.h1.status == result.h2.status == "PASS"
    assert {(x.control, x.block) for x in result.h1.comparisons} == {
        (control, block) for control in ("global_stack", "equal") for block in (2, 4)
    }
    assert {(x.control, x.block) for x in result.h2.comparisons} == {
        (control, block) for control in ("equal100", "buy_hold100") for block in (2, 4)
    }
    assert len(result.bootstrap) == 4
    assert result.evidence_scope == "COMPUTATIONAL_STATISTICS_SOURCE_AUTHORITY_UNVERIFIED"
    assert not result.confirmation_claim and not result.original_nll_hypothesis_satisfied
    assert not result.profitability_guarantee


def test_pooled_score_bootstrap_uses_original_date_weights_and_shared_indices() -> None:
    first, second = fold("A"), fold("B", 25)
    first = replace(first, energy=np.repeat(np.array([[1.0, 2.0, 4.0]]), 24, axis=0))
    second = replace(second, energy=np.repeat(np.array([[3.0, 4.0, 8.0]]), 25, axis=0))
    result = assess_study((first, second), integrity_verified=True, complete_matrix=True)
    assert result.h1 is not None
    rows = {(row.control, row.block): row for row in result.h1.comparisons}
    expected = 1 - (24 * 1 + 25 * 3) / (24 * 2 + 25 * 4)
    assert rows[("global_stack", 2)].pooled_point == pytest.approx(expected)
    assert expected != pytest.approx((0.5 + 0.25) / 2)
    arrays = {(name, block): values for name, block, values in result.bootstrap}
    for block in (2, 4):
        candidate = np.array([sum(first.energy[i, 0] for i in row) for row in arrays[("A", block)]])
        candidate += np.array([sum(second.energy[i, 0] for i in row) for row in arrays[("B", block)]])
        control = np.array([sum(first.energy[i, 1] for i in row) for row in arrays[("A", block)]])
        control += np.array([sum(second.energy[i, 1] for i in row) for row in arrays[("B", block)]])
        bounds = np.quantile(1 - candidate / control, [0.025, 0.975], method="linear")
        assert rows[("global_stack", block)].lower == pytest.approx(bounds[0])
        assert rows[("global_stack", block)].upper == pytest.approx(bounds[1])


def test_exact_five_percent_boundary_does_not_pass_from_subtraction_roundoff() -> None:
    first, second = fold("A"), fold("B")
    energy = np.repeat(np.array([[19.0, 20.0, 20.0]]), 24, axis=0)
    result = assess_study(
        (replace(first, energy=energy), replace(second, energy=energy)),
        integrity_verified=True,
        complete_matrix=True,
    )
    assert result.h1 is not None
    assert result.h1.status == "FAIL"
    assert all(not row.lower_clears_margin for row in result.h1.comparisons)
    assert result.disposition == "MIXED"


def test_one_failing_fold_or_control_cannot_be_replaced_by_favorable_pooled_result() -> None:
    first, second = fold("A"), fold("B")
    energy = second.energy.copy()
    energy[:, 1] = 0.7
    result = assess_study(
        (first, replace(second, energy=energy)), integrity_verified=True, complete_matrix=True
    )
    assert result.h1 is not None and result.h1.status == "FAIL"
    assert len(result.h1.comparisons) == 4


def test_undefined_required_score_endpoint_is_inconclusive_without_dropping_family() -> None:
    first, second = fold("A"), fold("B")
    energy = first.energy.copy()
    energy[:, 1] = 0
    result = assess_study(
        (replace(first, energy=energy), second), integrity_verified=True, complete_matrix=True
    )
    assert result.h1 is not None and result.h1.status == "INCONCLUSIVE"
    assert result.disposition == "INCONCLUSIVE"
    assert len(result.h1.comparisons) == 4
    assert all(row.reason for row in result.h1.comparisons if row.control == "global_stack")


def test_unfavorable_required_block_length_cannot_be_omitted() -> None:
    first, second = fold("A"), fold("B")
    energy = first.energy.copy()
    energy[:, 0] = np.concatenate((np.full(12, 0.73), np.full(12, 1.03)))
    result = assess_study(
        (replace(first, energy=energy), replace(second, energy=energy)),
        integrity_verified=True,
        complete_matrix=True,
    )
    assert result.h1 is not None
    for row in result.h1.comparisons:
        assert row.lower is not None
        assert row.lower_clears_margin is (row.block == 2)
        assert row.lower == pytest.approx(0.06375 if row.block == 2 else 0.045)
    assert result.h1.status == result.disposition == "INCONCLUSIVE"


def test_undefined_bootstrap_draw_invalidates_the_endpoint_even_with_positive_observed_mean() -> None:
    first, second = fold("A"), fold("B")
    energy = first.energy.copy()
    energy[:, 0] = 0.001
    energy[:, 1] = 0
    energy[0, 1] = 1
    result = assess_study(
        (replace(first, energy=energy), replace(second, energy=energy)),
        integrity_verified=True,
        complete_matrix=True,
    )
    assert result.h1 is not None
    required = tuple(row for row in result.h1.comparisons if row.control == "global_stack")
    assert len(required) == 2
    assert all(row.reason and "no draw dropped" in row.reason for row in required)
    assert result.h1.status == result.disposition == "INCONCLUSIVE"


@pytest.mark.parametrize("defect", ["wealth", "cash", "drawdown"])
def test_economic_point_gates_cannot_be_rescued_by_statistical_success(defect: str) -> None:
    first, second = fold("A"), fold("B")
    if defect == "drawdown":
        second = replace(second, maximum_drawdown=np.array([0.21, 0.2, 0.3, 0.2, 0.15]))
    else:
        wealth = second.terminal_wealth.copy()
        wealth[0] = 110000 if defect == "wealth" else 100000
        increments = np.repeat((np.log(wealth / 100000) / 24)[None, :], 24, axis=0)
        second = replace(second, terminal_wealth=wealth, log_increments=increments)
    result = assess_study((first, second), integrity_verified=True, complete_matrix=True)
    assert result.h2 is not None and result.h2.status == "FAIL"
    assert result.disposition == "MIXED"
    assert any(not passed for _, passed in result.h2.point_gates)


def test_h3_uses_equal_date_block_utility_and_separate_terminal_values() -> None:
    result = assess_study((fold("A"), fold("B", 25)), integrity_verified=True, complete_matrix=True)
    first, second, pooled = result.h3
    first_reference = (1.2 ** (-3 / 24) - 1) / -3
    second_reference = (1.2 ** (-3 / 25) - 1) / -3
    assert first.robust_mean_block_utility == pytest.approx(first_reference)
    assert second.robust_mean_block_utility == pytest.approx(second_reference)
    assert pooled.robust_mean_block_utility == pytest.approx(
        (24 * first_reference + 25 * second_reference) / 49
    )
    assert first.robust_terminal_utility == pytest.approx((1.2**-3 - 1) / -3)
    assert pooled.robust_terminal_utility is None


def test_nonfinite_secondary_utility_is_an_explicit_numerical_failure() -> None:
    first, second = fold("A"), fold("B")
    increments = first.log_increments.copy()
    increments[0, 0] += 300
    increments[1, 0] -= 300
    first = replace(first, log_increments=increments)
    with pytest.raises(NumericalError, match="CRRA utility arithmetic"):
        assess_study((first, second), integrity_verified=True, complete_matrix=True)


def test_missing_matrix_or_integrity_rejects_support_and_bad_pairing_fails() -> None:
    first, second = fold("A"), fold("B")
    assert (
        assess_study((first, second), integrity_verified=False, complete_matrix=True).disposition == "INVALID"
    )
    assert (
        assess_study((first, second), integrity_verified=True, complete_matrix=False).disposition == "INVALID"
    )
    with pytest.raises(ContractError):
        assess_study((second, first), integrity_verified=True, complete_matrix=True)
    with pytest.raises(ContractError, match="telescope"):
        replace(first, log_increments=np.zeros((24, 5)))
    with pytest.raises(ContractError, match="fold scope"):
        replace(first, decision_us=first.decision_us[::-1])
    with pytest.raises(ValueError):
        first.energy.setflags(write=True)
