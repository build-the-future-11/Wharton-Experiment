import numpy as np
import torch

from rejepa.engram import SpectralEngramMemory
from rejepa.memory_diagnostics import memory_diagnostics, retrieval_precision, utility_forgetting_candidates
from rejepa.simulator import (
    CounterfactualLatentSimulator,
    StressCondition,
    conditional_coverage,
    synthetic_real_distinguishability,
    tail_event_diagnostics,
)


def test_stress_conditioned_simulator_shapes_and_reproducibility() -> None:
    torch.manual_seed(1)
    simulator = CounterfactualLatentSimulator(6, worlds=4, horizon=3, interpreters=2)
    latent = torch.randn(5, 6)
    condition = StressCondition(volatility=1.0, currency=-0.5)
    first = simulator(latent, condition, generator=torch.Generator().manual_seed(9))
    second = simulator(latent, condition, generator=torch.Generator().manual_seed(9))
    assert first.trajectories.shape == (5, 4, 3, 6)
    assert torch.equal(first.trajectories, second.trajectories)


def test_simulator_coverage_tail_and_distinguishability_diagnostics() -> None:
    rng = np.random.default_rng(4)
    observed = rng.normal(size=20)
    draws = observed[:, None] + rng.normal(scale=0.5, size=(20, 50))
    coverage = conditional_coverage(observed, draws)
    tail = tail_event_diagnostics(observed, draws, threshold=-1.0)
    auc = synthetic_real_distinguishability(
        rng.normal(size=(30, 4)), rng.normal(loc=0.2, size=(30, 4)), seed=3
    )
    assert 0 <= coverage["overall"] <= 1
    assert 0 <= tail["tail_brier"] <= 1
    assert 0 <= auc <= 1


def test_memory_diagnostics_respect_maturity() -> None:
    memory = SpectralEngramMemory(3, 3, capacity=4)
    first = memory.write(torch.tensor([1.0, 0, 0]), torch.ones(3), 1, 2, 2, 1.0)
    second = memory.write(torch.tensor([0.9, 0.1, 0]), torch.ones(3), 3, 5, 5, 0.1)
    result = memory_diagnostics(memory, torch.tensor([[1.0, 0, 0]]), as_of_ns=2, rare_identifiers={first})
    assert result["eligible_records"] == 1
    assert result["rare_regime_recall"] == 1
    assert utility_forgetting_candidates(memory, 1) == (second,)


def test_retrieval_precision() -> None:
    assert retrieval_precision([1, 2, 3], {2, 3}) == 2 / 3
    assert retrieval_precision([], {1}) == 0
