import numpy as np
import torch

from rejepa.graph_memory import EngramKnowledgeGraph, TemporalRelation
from rejepa.representation import downstream_ridge_probe, latent_diagnostics
from rejepa.spectral_diagnostics import amplitude_phase, causal_time_frequency_energy


def test_latent_diagnostics_and_probe_are_executable() -> None:
    latent = torch.randn(20, 5)
    diagnostics = latent_diagnostics(latent)
    assert 1 <= diagnostics["effective_rank"] <= 5
    x = np.arange(60.0).reshape(20, 3)
    y = x[:, 0] * 0.2
    probe = downstream_ridge_probe(x[:15], y[:15], x[15:], y[15:])
    assert probe["mse"] >= 0


def test_graph_relations_are_causal() -> None:
    graph = EngramKnowledgeGraph(
        ("equity", "fx"),
        (TemporalRelation("fx", "equity", 0.5, 10, "currency_transmission"),),
    )
    values = torch.tensor([[1.0], [2.0]])
    before = graph.propagate(values, 9)
    after = graph.propagate(values, 10)
    assert torch.equal(before, values)
    assert not torch.equal(after, values)


def test_spectral_amplitude_phase_and_time_frequency_shapes() -> None:
    sequence = torch.randn(2, 16, 3)
    amplitude, phase = amplitude_phase(sequence)
    energy = causal_time_frequency_energy(sequence, window=8, step=4)
    assert amplitude.shape == phase.shape == (2, 8, 3)
    assert energy.shape == (2, 3, 3)
    assert (amplitude >= 0).all()
