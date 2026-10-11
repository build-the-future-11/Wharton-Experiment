import pytest
import torch

from reng.memory import EngramBank, EngramRecord, SpectralEngramRetriever


def record(index: int, *, available: int | None = None) -> EngramRecord:
    return EngramRecord(
        asset_id="A",
        horizon=21,
        branch=0,
        origin_ns=index,
        available_ns=index + 2 if available is None else available,
        context=torch.full((168,), float(index)).detach(),
        spectrum=torch.complex(torch.ones(32), torch.zeros(32)).detach(),
        value=torch.full((35,), float(index)).detach(),
        record_id=f"r{index}",
    )


def test_memory_visibility_requires_full_maturity() -> None:
    bank = EngramBank()
    bank.insert(record(1, available=10))
    assert bank.eligible("A", 21, 0, 9) == ()
    assert len(bank.eligible("A", 21, 0, 10)) == 1


def test_frozen_bank_rejects_insertions() -> None:
    bank = EngramBank()
    bank.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        bank.insert(record(1))


def test_too_few_records_returns_exact_zero() -> None:
    retriever = SpectralEngramRetriever(0)
    result = retriever(
        torch.zeros(168), torch.complex(torch.ones(32), torch.zeros(32)), (record(1),), query_ns=20
    )
    assert result.missing
    assert torch.equal(result.value, torch.zeros(32))


def test_permutation_preserves_value_multiset() -> None:
    bank = EngramBank()
    for index in range(8):
        bank.insert(record(index))
    original = [float(item.value[0]) for item in bank.eligible("A", 21, 0, 20)]
    permuted = [float(item[0]) for item in bank.permuted_values("A", 21, 0, 20, 11)]
    assert sorted(original) == sorted(permuted)
    assert original != permuted
