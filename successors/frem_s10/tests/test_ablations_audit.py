"""Independent finite-data secondary inference references, never market evidence."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from frem_s10.analysis.ablations import ABLATION_IDS, AblationFold, assess_ablations
from frem_s10.analysis.bootstrap import campaign_indices
from frem_s10.errors import ContractError, IntegrityError
from frem_s10.metadata import midnight_us


def fold(name: str, count: int) -> AblationFold:
    day = date(2010, 1, 4) if name == "A" else date(2013, 1, 7)
    dates = midnight_us(day) + np.arange(count, dtype=np.int64) * (28 * 86_400_000_000)
    primary = np.ones(count)
    changes = (0.1 if name == "A" else 0.3, -0.1, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0)
    return AblationFold(
        name,
        dates,
        tuple(f"{name}/{i}" for i in range(count)),
        primary,
        tuple((identity, primary + change) for identity, change in zip(ABLATION_IDS, changes, strict=True)),
    )


def test_all_sixteen_centered_tests_and_structural_zeros_are_retained() -> None:
    result = assess_ablations((fold("A", 24), fold("B", 25)))
    assert len(result.comparisons) == 16
    assert {(row.ablation_id, row.block) for row in result.comparisons} == {
        (name, block) for name in ABLATION_IDS for block in (2, 4)
    }
    for row in result.comparisons:
        if row.ablation_id in ("A1", "A6"):
            assert row.raw_p == 1 / 5001
            assert row.rejects_zero_advantage
        else:
            assert row.raw_p == row.holm_p == 1
            assert not row.rejects_zero_advantage
        assert row.structural_zero is (row.ablation_id in ("A4", "A5"))
        if row.ablation_id == "A1":
            assert row.pooled_mean == pytest.approx((24 * 0.1 + 25 * 0.3) / 49)
            assert row.pooled_mean != pytest.approx(0.2)
    assert result.evidence_scope == "COMPUTATIONAL_STATISTICS_SOURCE_AUTHORITY_UNVERIFIED"
    assert not result.confirmation_claim and not result.changes_primary


def test_centering_direction_plus_one_and_global_holm_match_independent_reference() -> None:
    first, second = fold("A", 24), fold("B", 25)
    arrays = []
    for source in (first, second):
        changes = 0.02 + 0.3 * np.sin(np.arange(len(source.origin_ids)) * 0.7)
        scores = tuple((name, values + changes if name == "A8" else values) for name, values in source.scores)
        arrays.append(replace(source, scores=scores))
    result = assess_ablations((arrays[0], arrays[1]))
    saved = {(name, block): values for name, block, values in result.bootstrap}
    exact = campaign_indices(24, 25)
    for key in saved:
        np.testing.assert_array_equal(saved[key], exact[key])
        with pytest.raises(ValueError):
            saved[key].setflags(write=True)
    a = dict(arrays[0].scores)["A8"] - arrays[0].primary
    b = dict(arrays[1].scores)["A8"] - arrays[1].primary
    observed = (sum(float(x) for x in a) + sum(float(x) for x in b)) / 49
    for row in (item for item in result.comparisons if item.ablation_id == "A8"):
        count = 0
        uncentered_count = 0
        for a_indices, b_indices in zip(saved[("A", row.block)], saved[("B", row.block)], strict=True):
            resampled = (sum(float(a[i]) for i in a_indices) + sum(float(b[i]) for i in b_indices)) / 49
            count += resampled - observed >= observed
            uncentered_count += resampled >= observed
        assert row.pooled_mean == pytest.approx(observed)
        assert row.exceedances == count
        assert row.raw_p == (1 + count) / 5001
        assert count != uncentered_count
    previous = 0.0
    for rank, row in enumerate(
        sorted(result.comparisons, key=lambda item: (item.raw_p, item.ablation_id, item.block))
    ):
        previous = max(previous, min(1.0, (16 - rank) * row.raw_p))
        assert row.holm_p == previous
        assert row.rejects_zero_advantage is (previous <= 0.05)


def test_incomplete_family_wrong_pairing_and_policy_only_variation_fail() -> None:
    first, second = fold("A", 24), fold("B", 25)
    with pytest.raises(ContractError, match=r"A1\.\.A8"):
        replace(first, scores=first.scores[:-1])
    with pytest.raises(ContractError, match=r"A1\.\.A8"):
        replace(first, scores=first.scores[::-1])
    for identity in ("A4", "A5"):
        changed = tuple(
            (name, values + 1e-14 if name == identity else values) for name, values in first.scores
        )
        with pytest.raises(IntegrityError, match="exact zero"):
            replace(first, scores=changed)
    with pytest.raises(ContractError, match="ordered A/B"):
        assess_ablations((second, first))
    with pytest.raises(ContractError, match="fold scope"):
        replace(first, decision_us=first.decision_us[::-1])
    with pytest.raises(ValueError):
        first.scores[0][1].setflags(write=True)
