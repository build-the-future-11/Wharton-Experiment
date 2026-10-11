"""Manuscript endpoints, diagnostics, and dependence-aware inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_QUANTILES = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def pinball_loss(outcome: np.ndarray, prediction: np.ndarray, quantile: float) -> np.ndarray:
    residual = np.asarray(outcome) - np.asarray(prediction)
    return residual * (quantile - (residual < 0))


def normalized_quantile_scores(
    outcomes: np.ndarray,
    quantile_predictions: np.ndarray,
    scales: np.ndarray,
    dates: np.ndarray,
    *,
    quantiles: np.ndarray = DEFAULT_QUANTILES,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique dates and per-date scores; assets never inflate N."""
    y = np.asarray(outcomes, dtype=float)
    qhat = np.asarray(quantile_predictions, dtype=float)
    scale = np.asarray(scales, dtype=float)
    date = np.asarray(dates)
    if qhat.shape != y.shape + (len(quantiles),) or scale.shape != y.shape:
        raise ValueError("Outcome, quantile, and scale shapes do not align")
    if (scale <= 0).any():
        raise ValueError("Normalization scales must be positive")
    if valid is None:
        valid = np.ones_like(y, dtype=bool)
    losses = np.stack(
        [pinball_loss(y / scale, qhat[..., index] / scale, q) for index, q in enumerate(quantiles)],
        axis=-1,
    )
    unique = np.unique(date)
    scores = np.empty(len(unique), dtype=float)
    for index, value in enumerate(unique):
        selection = (date == value)[:, None] & valid
        if not selection.any():
            raise ValueError("A declared date has no valid assets")
        scores[index] = losses[selection].mean()
    return unique, scores


def relative_improvement(candidate_score: np.ndarray, control_score: np.ndarray) -> float:
    candidate = float(np.mean(candidate_score))
    control = float(np.mean(control_score))
    if control <= 0:
        raise ValueError("Control score must be positive")
    return 1.0 - candidate / control


def _stationary_indices(length: int, expected_block: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.empty(length, dtype=int)
    indices[0] = rng.integers(length)
    restart_probability = 1.0 / expected_block
    for position in range(1, length):
        if rng.random() < restart_probability:
            indices[position] = rng.integers(length)
        else:
            indices[position] = (indices[position - 1] + 1) % length
    return indices


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lower: float
    upper: float
    draws: np.ndarray
    expected_block: int


def paired_stationary_bootstrap(
    candidate: np.ndarray,
    control: np.ndarray,
    folds: np.ndarray,
    *,
    repetitions: int = 5000,
    expected_block: int = 63,
    seed: int = 11,
) -> BootstrapResult:
    candidate = np.asarray(candidate, dtype=float)
    control = np.asarray(control, dtype=float)
    folds = np.asarray(folds)
    if candidate.shape != control.shape or candidate.shape != folds.shape or candidate.ndim != 1:
        raise ValueError("Candidate, control, and folds must be aligned vectors")
    if len(candidate) < 2 or expected_block <= 0 or repetitions <= 0:
        raise ValueError("Invalid bootstrap dimensions")
    if not np.isfinite(candidate).all() or not np.isfinite(control).all() or (control <= 0).any():
        raise ValueError("Scores must be finite and controls positive")
    rng = np.random.default_rng(seed)
    fold_values = np.unique(folds)
    fold_indices = [np.flatnonzero(folds == fold) for fold in fold_values]
    draws = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sample = np.concatenate(
            [
                indices[_stationary_indices(len(indices), expected_block, rng)]
                for indices in fold_indices
            ]
        )
        draws[repetition] = relative_improvement(candidate[sample], control[sample])
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return BootstrapResult(
        relative_improvement(candidate, control), float(lower), float(upper), draws, expected_block
    )


def classify_primary(
    result: BootstrapResult, *, threshold: float = 0.02, valid: bool = True
) -> str:
    if not valid:
        return "INVALID"
    if result.point >= threshold and result.lower > 0:
        return "SUPPORTED"
    if result.upper < threshold:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def holm_adjust(
    p_values: dict[str, float], alpha: float = 0.05
) -> dict[str, dict[str, float | bool]]:
    if not p_values or any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must be a nonempty mapping in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    rejected_so_far = True
    output: dict[str, dict[str, float | bool]] = {}
    running_adjusted = 0.0
    for rank, (name, value) in enumerate(ordered):
        multiplier = count - rank
        running_adjusted = max(running_adjusted, min(1.0, value * multiplier))
        critical = alpha / multiplier
        reject = rejected_so_far and value <= critical
        rejected_so_far = reject
        output[name] = {
            "raw": value,
            "adjusted": running_adjusted,
            "critical": critical,
            "reject": reject,
        }
    return output


def representation_diagnostics(latent: np.ndarray) -> dict[str, object]:
    values = np.asarray(latent, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Expected at least two embedding rows")
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / (len(values) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)
    probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
    nonzero = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    coordinate_std = values.std(axis=0)
    return {
        "coordinate_variance": values.var(axis=0).tolist(),
        "mean_norm": float(np.linalg.norm(values, axis=1).mean()),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "effective_rank": effective_rank,
        "collapsed": bool(np.median(coordinate_std) < 0.05 or effective_rank < 4),
    }


def interval_coverage(
    outcomes: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    y, lo, hi = map(lambda value: np.asarray(value, dtype=float), (outcomes, lower, upper))
    if y.shape != lo.shape or y.shape != hi.shape or np.any(lo > hi):
        raise ValueError("Invalid interval arrays")
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
    }


def energy_score(scenarios: np.ndarray, outcome: np.ndarray, *, seed: int = 11) -> float:
    draws = np.asarray(scenarios, dtype=float)
    value = np.asarray(outcome, dtype=float)
    if draws.ndim != 2 or value.shape != (draws.shape[1],):
        raise ValueError("Scenario/outcome dimensions differ")
    first = np.linalg.norm(draws - value, axis=1).mean()
    rng = np.random.default_rng(seed)
    paired = draws[rng.permutation(len(draws))]
    second = 0.5 * np.linalg.norm(draws - paired, axis=1).mean()
    return float(first - second)
