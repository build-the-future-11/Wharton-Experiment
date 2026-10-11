import numpy as np
import torch

from reng.graph import signed_correlation_adjacency
from reng.probability import (
    dense_gaussian_log_prob,
    low_rank_gaussian_log_prob,
    marginal_mixture_quantile,
)
from reng.reservoir import FixedReservoir
from reng.spectral import CausalSpectralChannels, finite_difference_gradient_error


def test_fourier_gradient_matches_float64_finite_difference() -> None:
    module = CausalSpectralChannels()
    values = torch.linspace(-1, 1, 128) + 0.1 * torch.sin(torch.arange(128.0))
    assert finite_difference_gradient_error(module, values) < 1e-5


def test_reservoir_singular_value_and_contraction() -> None:
    reservoir = FixedReservoir(18, 32, leak=0.2, seed=11)
    assert float(torch.linalg.svdvals(reservoir.recurrent).max()) <= 0.9 + 1e-8
    sequence = torch.randn(3, 20, 18)
    initial_a = torch.randn(3, 32)
    initial_b = torch.randn(3, 32)
    one_step = sequence[:, :1]
    out_a = reservoir(one_step, initial_a)
    out_b = reservoir(one_step, initial_b)
    assert torch.linalg.vector_norm(out_a - out_b, dim=-1).max() <= (
        reservoir.contraction_bound * torch.linalg.vector_norm(initial_a - initial_b, dim=-1).max()
        + 1e-6
    )


def test_signed_graph_is_row_abs_normalized() -> None:
    base = torch.linspace(-1, 1, 63)
    returns = torch.stack([base, base, -base]).unsqueeze(0)
    observed = torch.ones_like(returns, dtype=torch.bool)
    adjacency = signed_correlation_adjacency(returns, observed)
    assert torch.allclose(adjacency.abs().sum(-1), torch.ones(1, 3))
    assert adjacency[0, 0, 2] < 0


def test_signed_graph_respects_pairwise_missingness() -> None:
    returns = torch.tensor([[[1.0, 2.0, 3.0, 99.0], [2.0, 4.0, 6.0, -99.0], [3.0, 1.0, 2.0, 4.0]]])
    observed = torch.ones_like(returns, dtype=torch.bool)
    observed[0, :2, -1] = False
    adjacency = signed_correlation_adjacency(
        returns, observed, min_pairs=3, threshold=0.9, scale=0.8
    )
    assert adjacency[0, 0, 1] > 0
    assert torch.allclose(adjacency, adjacency.transpose(-1, -2), atol=0.5)


def test_low_rank_log_probability_matches_dense() -> None:
    torch.manual_seed(11)
    value = torch.randn(8, dtype=torch.float64)
    mean = torch.randn(8, dtype=torch.float64)
    diagonal = torch.rand(8, dtype=torch.float64) + 0.2
    factors = torch.randn(8, 4, dtype=torch.float64) * 0.1
    low_rank, _ = low_rank_gaussian_log_prob(value, mean, diagonal, factors, jitters=(0.0,))
    dense = dense_gaussian_log_prob(value, mean, diagonal, factors)
    assert abs(float(low_rank - dense)) < 1e-7


def test_analytic_mixture_quantile_matches_large_draw() -> None:
    weights = np.array([0.3, 0.7])
    means = np.array([-0.5, 0.7])
    variances = np.array([0.4, 0.2])
    median = marginal_mixture_quantile(0.5, weights, means, variances)
    rng = np.random.default_rng(11)
    branches = rng.choice(2, size=300_000, p=weights)
    draws = rng.normal(means[branches], np.sqrt(variances[branches]))
    assert abs(median - np.quantile(draws, 0.5)) < 0.01
