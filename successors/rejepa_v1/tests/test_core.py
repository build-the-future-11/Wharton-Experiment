import torch

from rejepa.reservoir import StableReservoir
from rejepa.spectral import CausalFourierBank


def test_reservoir_is_fixed_and_contractive() -> None:
    reservoir = StableReservoir(3, 12, spectral_norm=0.8, leak=0.4)
    assert list(reservoir.parameters()) == []
    assert reservoir.contraction_bound < 1.0
    assert torch.linalg.matrix_norm(reservoir.recurrent, ord=2) <= 0.80001
    output = reservoir(torch.randn(2, 8, 3))
    assert output.shape == (2, 8, 12)


def test_fourier_bank_is_prefix_causal() -> None:
    bank = CausalFourierBank(features=2, bands=3)
    prefix = torch.randn(2, 12, 2)
    future_a = torch.randn(2, 5, 2)
    future_b = torch.randn(2, 5, 2) * 100
    features_a = bank(torch.cat([prefix, future_a], dim=1)[:, :12])
    features_b = bank(torch.cat([prefix, future_b], dim=1)[:, :12])
    assert torch.equal(features_a, features_b)
    assert features_a.shape == (2, 6)


def test_fourier_bank_supports_short_forecast_targets() -> None:
    output = CausalFourierBank(features=2, bands=3)(torch.randn(4, 2, 2))
    assert output.shape == (4, 6)
    assert torch.isfinite(output).all()
