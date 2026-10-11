import pytest
import torch

from rejepa.engram import SpectralEngramMemory


def test_unmatured_engram_is_invisible_and_matured_is_retrievable() -> None:
    memory = SpectralEngramMemory(3, 4, capacity=2)
    memory.write(torch.tensor([1.0, 0.0, 0.0]), torch.arange(4.0), 10, 20, 20, 1.0)
    query = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    before, before_ids = memory.retrieve(query, as_of_ns=19)
    assert before_ids == ()
    assert torch.equal(before, torch.zeros_like(before))
    after, after_ids = memory.retrieve(query, as_of_ns=20)
    assert after_ids == (0,)
    assert torch.equal(after, torch.arange(4.0).unsqueeze(0))
    after.sum().backward()
    assert query.grad is not None
    assert all(record.key.requires_grad is False for record in memory.records)


def test_same_time_or_future_credit_cannot_be_written() -> None:
    memory = SpectralEngramMemory(2, 2)
    with pytest.raises(ValueError):
        memory.write(torch.ones(2), torch.ones(2), 10, 10, 10, 0.0)
    with pytest.raises(ValueError):
        memory.write(torch.ones(2), torch.ones(2), 10, 20, 19, 0.0)


def test_capacity_retains_high_utility_records() -> None:
    memory = SpectralEngramMemory(2, 2, capacity=2)
    for index, utility in enumerate([0.1, 2.0, 1.0]):
        memory.write(
            torch.tensor([1.0, float(index)]),
            torch.ones(2) * index,
            index,
            index + 1,
            index + 1,
            utility,
        )
    assert {record.utility for record in memory.records} == {1.0, 2.0}
