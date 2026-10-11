"""Bounded local benchmark for engineering-equivalent cleanup optimizations."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from reng.funding import funding_frontier, simulate_funding_path
from reng.graph import signed_correlation_adjacency


def _timed(callable_, repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples))


def _reference_graph(
    returns: torch.Tensor,
    observed: torch.Tensor,
    *,
    min_pairs: int,
    threshold: float,
    scale: float,
) -> torch.Tensor:
    batch, assets, _ = returns.shape
    adjacency = torch.zeros(batch, assets, assets, dtype=returns.dtype)
    for batch_index in range(batch):
        for left_index in range(assets):
            adjacency[batch_index, left_index, left_index] = 1.0
            for right_index in range(left_index + 1, assets):
                mask = observed[batch_index, left_index] & observed[batch_index, right_index]
                if int(mask.sum()) < min_pairs:
                    continue
                left = returns[batch_index, left_index, mask]
                right = returns[batch_index, right_index, mask]
                left = left - left.mean()
                right = right - right.mean()
                denominator = torch.sqrt(left.square().sum() * right.square().sum()).clamp_min(1e-12)
                correlation = (left * right).sum() / denominator
                if correlation.abs() >= threshold:
                    adjacency[batch_index, left_index, right_index] = scale * correlation
                    adjacency[batch_index, right_index, left_index] = scale * correlation
    return adjacency / adjacency.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)


def graph_benchmark() -> dict[str, float]:
    generator = torch.Generator().manual_seed(11)
    returns = torch.randn(4, 32, 128, generator=generator, dtype=torch.float64)
    observed = torch.rand(4, 32, 128, generator=generator) > 0.08
    arguments = {"min_pairs": 50, "threshold": 0.2, "scale": 0.8}
    reference = _reference_graph(returns, observed, **arguments)
    optimized = signed_correlation_adjacency(returns, observed, **arguments)
    maximum_error = float((reference - optimized).abs().max())
    if maximum_error > 1e-10:
        raise AssertionError(f"Graph optimization changed output: {maximum_error}")
    reference_seconds = _timed(lambda: _reference_graph(returns, observed, **arguments), 3)
    optimized_seconds = _timed(lambda: signed_correlation_adjacency(returns, observed, **arguments), 20)
    return {
        "reference_median_seconds": reference_seconds,
        "optimized_median_seconds": optimized_seconds,
        "speedup": reference_seconds / optimized_seconds,
        "maximum_absolute_error": maximum_error,
    }


def funding_benchmark() -> dict[str, float]:
    rng = np.random.default_rng(11)
    scenarios = rng.normal(0.003, 0.02, (500, 192, 4))
    weights = np.full(4, 0.25)
    grid = np.linspace(0.0, 100_000.0, 5)
    portfolio = scenarios @ weights

    def reference():
        return [
            [
                simulate_funding_path(
                    path,
                    facility_month=48,
                    facility_contribution=float(contribution),
                    terminal_reserve=25_000.0,
                )
                for path in portfolio
            ]
            for contribution in grid
        ]

    expected = reference()
    optimized = funding_frontier(
        scenarios,
        weights,
        grid,
        facility_month=48,
        terminal_reserve=25_000.0,
    )
    expected_success = np.asarray([np.mean([result.success for result in group]) for group in expected])
    observed_success = np.asarray([summary.success_frequency for summary in optimized])
    maximum_error = float(np.max(np.abs(expected_success - observed_success)))
    if maximum_error > 0:
        raise AssertionError(f"Funding optimization changed success rates: {maximum_error}")
    reference_seconds = _timed(reference, 3)
    optimized_seconds = _timed(
        lambda: funding_frontier(
            scenarios,
            weights,
            grid,
            facility_month=48,
            terminal_reserve=25_000.0,
        ),
        10,
    )
    return {
        "reference_median_seconds": reference_seconds,
        "optimized_median_seconds": optimized_seconds,
        "speedup": reference_seconds / optimized_seconds,
        "maximum_success_frequency_error": maximum_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    receipt = {
        "status": "PASS",
        "scope": "Local synthetic engineering benchmark; not market or scientific evidence",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "graph_adjacency": graph_benchmark(),
        "funding_frontier": funding_benchmark(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
