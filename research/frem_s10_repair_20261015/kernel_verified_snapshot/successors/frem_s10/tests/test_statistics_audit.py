"""Independent block construction and statistical edge cases on generated arrays."""

import numpy as np
import pytest
from frem_s10.analysis.bootstrap import (
    assess_precision,
    campaign_indices,
    holm16,
    pilot_indices,
    relative_gain,
    resampled_gains,
)
from frem_s10.errors import ContractError, NumericalError


def test_exact_campaign_stream_order_and_scalar_block_construction() -> None:
    actual = campaign_indices(7, 9)
    children = np.random.SeedSequence(1901).spawn(4)
    for (fold, block, count), child in zip(
        (("A", 2, 7), ("B", 2, 9), ("A", 4, 7), ("B", 4, 9)), children, strict=True
    ):
        rng = np.random.Generator(np.random.PCG64(child))
        for draw in range(3):
            starts = rng.integers(0, count, size=(count + block - 1) // block, dtype=np.int64)
            expected = [(int(start) + offset) % count for start in starts for offset in range(block)][:count]
            assert actual[(fold, block)][draw].tolist() == expected
        with pytest.raises(ValueError):
            actual[(fold, block)].setflags(write=True)
    assert not np.array_equal(actual[("A", 2)], pilot_indices(7)[2])


def test_relative_gain_is_ratio_of_means_and_paired_resampling() -> None:
    p = np.array([0.5, 3.0])
    c = np.array([1.0, 4.0])
    assert relative_gain(p, c) == pytest.approx(0.3)
    assert relative_gain(p, c) != pytest.approx(np.mean(1 - p / c))
    indices = np.tile(np.array([0, 1], dtype=np.int64), (5000, 1))
    assert np.allclose(resampled_gains(p, c, indices), 0.3)


def test_undefined_draw_is_not_silently_deleted() -> None:
    p = np.array([0.0, 1.0])
    c = np.array([0.0, 2.0])
    indices = np.tile(np.array([0, 1], dtype=np.int64), (5000, 1))
    indices[17] = [0, 0]
    with pytest.raises(NumericalError):
        resampled_gains(p, c, indices)


def test_perfectly_constant_apparent_gain_does_not_pass_precision() -> None:
    c = np.ones(32)
    assert assess_precision(c * 0.5, c, c, 38, 51).status == "PRECISION_UNIDENTIFIABLE"
    assert assess_precision(c, c * 2, c, 38, 51).status == "PRECISION_UNIDENTIFIABLE"
    assert assess_precision(c[:23], c[:23], c[:23], 38, 51).status == "PRECISION_UNIDENTIFIABLE"


def test_precision_requires_all_four_results_and_uses_prospective_counts() -> None:
    x = np.linspace(0, 2 * np.pi, 32)
    candidate = 1 + 0.02 * np.sin(x)
    equal = 1.1 + 0.01 * np.cos(x)
    global_stack = 1.05 + 0.02 * np.cos(x)
    result = assess_precision(candidate, equal, global_stack, 38, 51)
    assert result.status == "PRECISION_PASS"
    assert len(result.comparisons) == 4
    for _, _, pilot_sd, projected, passes in result.comparisons:
        assert projected == pytest.approx(pilot_sd * (32 / 89) ** 0.5)
        assert passes == (2.8 * projected <= 0.05)


def test_complete_holm_family_matches_hand_calculation() -> None:
    family = {(f"A{i}", block): 1.0 for i in range(1, 9) for block in (2, 4)}
    family[("A1", 2)] = 0.001
    family[("A2", 2)] = 0.002
    family[("A3", 2)] = 0.002
    result = holm16(family)
    assert result[("A1", 2)] == 0.016
    assert result[("A2", 2)] == 0.03
    assert result[("A3", 2)] == 0.03
    assert result[("A4", 2)] == 1
    with pytest.raises(ContractError):
        holm16({("A1", 2): 0.001})
    family[("A5", 4)] = 0.0
    with pytest.raises(ContractError):
        holm16(family)
