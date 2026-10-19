"""Complete no-overwrite fold and campaign artifact bundles."""

from __future__ import annotations

import io
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from frem_s12.analysis import PrimaryDecision
from frem_s12.evidence import atomic_json, atomic_write, package_identity, read_json, sha256_file
from frem_s12.experiment import ARMS, FoldResult
from frem_s12.policy import realize
from frem_s12.scores import Law, energy_score, variogram_score


def write_fold(output: Path, result: FoldResult, *, provenance: dict[str, object]) -> dict[str, object]:
    """Write all raw fold arrays and a hash-bound manifest."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("fold output already exists")
    output.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        months=result.months,
        outcomes=result.outcomes,
        validation_months=result.validation_months,
        validation_outcomes=result.validation_outcomes,
        validation_scores=result.validation_scores,
        validation_forecast_atoms=result.validation_forecast_atoms,
        validation_forecast_probabilities=result.validation_forecast_probabilities,
        validation_forecast_lengths=result.validation_forecast_lengths,
        scores=result.scores,
        variograms=result.variograms,
        expert_scores=result.expert_scores,
        router_weights=result.router_weights,
        forecast_atoms=result.forecast_atoms,
        forecast_probabilities=result.forecast_probabilities,
        forecast_lengths=result.forecast_lengths,
        wealth=result.wealth,
        targets=result.targets,
        holdings=result.holdings,
    )
    atomic_write(output / "arrays.npz", buffer.getvalue())
    summary = {
        "schema": "frem-s12-fold-summary-v3",
        "family": result.family,
        "fold": result.fold,
        "seed": result.seed,
        "arms": ARMS,
        "rows": len(result.months),
        "selected_non_frem": result.selected_non_frem,
        "mean_energy": dict(zip(ARMS, result.scores.mean(axis=0).tolist(), strict=True)),
        "mean_variogram": dict(zip(ARMS, result.variograms.mean(axis=0).tolist(), strict=True)),
        "terminal_wealth": dict(zip(ARMS, result.wealth[-1].tolist(), strict=True)),
        "provenance": provenance,
        "source_identity": package_identity()["sha256"],
    }
    atomic_json(output / "summary.json", summary)
    manifest = {
        "schema": "frem-s12-fold-manifest-v3",
        "complete": True,
        "files": {
            "arrays.npz": sha256_file(output / "arrays.npz"),
            "summary.json": sha256_file(output / "summary.json"),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def _read_fold_arrays(output: Path, summary: dict[str, Any]) -> FoldResult:
    """Load one validated fold result from its retained arrays."""
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        return FoldResult(
            str(summary["family"]),
            str(summary["fold"]),
            int(summary["seed"]),
            arrays["months"],
            arrays["outcomes"],
            arrays["validation_months"],
            arrays["validation_outcomes"],
            arrays["validation_scores"],
            arrays["validation_forecast_atoms"],
            arrays["validation_forecast_probabilities"],
            arrays["validation_forecast_lengths"],
            arrays["scores"],
            arrays["variograms"],
            arrays["expert_scores"],
            arrays["router_weights"],
            arrays["forecast_atoms"],
            arrays["forecast_probabilities"],
            arrays["forecast_lengths"],
            arrays["wealth"],
            arrays["targets"],
            arrays["holdings"],
            str(summary["selected_non_frem"]),
        )


def _verify_fold_reconstruction(result: FoldResult) -> None:
    """Reconstruct primary scores and policy accounting from raw evidence."""
    reconstructed_energy = np.empty_like(result.scores)
    reconstructed_variogram = np.empty_like(result.variograms)
    reconstructed_wealth = np.empty_like(result.wealth)
    prior_holdings = np.tile(
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (len(ARMS), 1),
    )
    cumulative_wealth = np.ones(len(ARMS), dtype=np.float64)
    reconstructed_validation = np.empty_like(result.validation_scores)
    for row in range(len(result.validation_months)):
        for arm in range(len(ARMS)):
            length = int(result.validation_forecast_lengths[row, arm])
            law = Law(
                result.validation_forecast_atoms[row, arm, :length],
                result.validation_forecast_probabilities[row, arm, :length],
            )
            reconstructed_validation[row, arm] = energy_score(law, result.validation_outcomes[row])
    for row in range(len(result.months)):
        for arm in range(len(ARMS)):
            length = int(result.forecast_lengths[row, arm])
            law = Law(
                result.forecast_atoms[row, arm, :length],
                result.forecast_probabilities[row, arm, :length],
            )
            reconstructed_energy[row, arm] = energy_score(law, result.outcomes[row])
            reconstructed_variogram[row, arm] = variogram_score(law, result.outcomes[row])
            factor, next_holdings = realize(
                result.targets[row, arm],
                result.outcomes[row],
                prior_holdings[arm],
                cost_bps=10.0,
            )
            cumulative_wealth[arm] *= factor
            reconstructed_wealth[row, arm] = cumulative_wealth[arm]
            if not np.allclose(next_holdings, result.holdings[row, arm], rtol=0, atol=1e-12):
                raise ValueError("fold artifact holdings do not reconstruct")
            prior_holdings[arm] = next_holdings
    if not np.allclose(reconstructed_energy, result.scores, rtol=1e-12, atol=1e-15):
        raise ValueError("fold artifact energy scores do not reconstruct")
    if not np.allclose(reconstructed_validation, result.validation_scores, rtol=1e-12, atol=1e-15):
        raise ValueError("fold artifact validation scores do not reconstruct")
    if not np.allclose(reconstructed_variogram, result.variograms, rtol=1e-12, atol=1e-15):
        raise ValueError("fold artifact variogram scores do not reconstruct")
    if not np.allclose(reconstructed_wealth, result.wealth, rtol=1e-12, atol=1e-15):
        raise ValueError("fold artifact wealth does not reconstruct")
    non_frem_indices = [ARMS.index(name) for name in ("gaussian", "student_t", "var", "causal_mlp")]
    expected_control = ("gaussian", "student_t", "var", "causal_mlp")[
        int(np.argmin(result.validation_scores[:, non_frem_indices].mean(axis=0)))
    ]
    if result.selected_non_frem != expected_control:
        raise ValueError("fold artifact selected control does not reconstruct")


def _month_range(start: int, end: int) -> np.ndarray:
    """Return the exact inclusive YYYYMM calendar range."""
    values: list[int] = []
    current = start
    while current <= end:
        values.append(current)
        year, month = divmod(current, 100)
        current = (year + 1) * 100 + 1 if month == 12 else year * 100 + month + 1
    return np.asarray(values, dtype=np.int64)


def _verify_fold_identity(result: FoldResult, *, engineering_only: bool) -> None:
    """Bind embedded identity and calendars to the frozen registry."""
    from frem_s12.campaign import FAMILIES, FOLDS, SEEDS

    matches = [fold for fold in FOLDS if fold.identifier == result.fold]
    family_ok = result.family in FAMILIES or (engineering_only and result.family == "generated")
    if not family_ok or result.seed not in SEEDS or len(matches) != 1:
        raise ValueError("fold artifact identity is outside the frozen registry")
    fold = matches[0]
    if not np.array_equal(result.validation_months, _month_range(fold.validation_start, fold.validation_end)):
        raise ValueError("fold artifact validation calendar differs from the frozen registry")
    if not np.array_equal(result.months, _month_range(fold.test_start, fold.test_end)):
        raise ValueError("fold artifact test calendar differs from the frozen registry")


def _verify_fold_summary(summary: dict[str, Any], result: FoldResult) -> None:
    """Bind producer summary values to independently reconstructed arrays."""
    if int(summary.get("rows", -1)) != len(result.months):
        raise ValueError("fold artifact row count differs from raw arrays")
    expected_energy = dict(zip(ARMS, result.scores.mean(axis=0).tolist(), strict=True))
    expected_variogram = dict(zip(ARMS, result.variograms.mean(axis=0).tolist(), strict=True))
    expected_wealth = dict(zip(ARMS, result.wealth[-1].tolist(), strict=True))
    if summary.get("mean_energy") != expected_energy:
        raise ValueError("fold summary energy differs from raw arrays")
    if summary.get("mean_variogram") != expected_variogram:
        raise ValueError("fold summary variogram differs from raw arrays")
    if summary.get("terminal_wealth") != expected_wealth:
        raise ValueError("fold summary wealth differs from raw arrays")


def read_fold(output: Path) -> FoldResult:
    """Read and independently reconstruct a complete fold evidence bundle."""
    if not output.is_dir() or output.is_symlink():
        raise ValueError("fold artifact root is not a regular directory")
    entries = set(output.iterdir())
    expected_entries = {output / "arrays.npz", output / "summary.json", output / "manifest.json"}
    if entries != expected_entries or any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("fold artifact directory inventory is not exact")
    if (output / "arrays.npz").stat().st_size > 512 * 2**20:
        raise ValueError("fold artifact array archive exceeds the safety limit")
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "frem-s12-fold-manifest-v3" or manifest.get("complete") is not True:
        raise ValueError("fold artifact manifest is incomplete or unsupported")
    if set(manifest.get("files", {})) != {"arrays.npz", "summary.json"}:
        raise ValueError("fold artifact manifest inventory is incomplete or foreign")
    for name, expected in manifest["files"].items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"fold artifact hash mismatch: {name}")
    summary = read_json(output / "summary.json")
    if summary.get("schema") != "frem-s12-fold-summary-v3":
        raise ValueError("fold artifact schema is unsupported")
    if summary.get("source_identity") != package_identity()["sha256"]:
        raise ValueError("fold artifact source identity is stale")
    if summary.get("arms") != list(ARMS):
        raise ValueError("fold artifact arm registry is unsupported")
    result = _read_fold_arrays(output, summary)
    provenance = summary.get("provenance")
    engineering_only = isinstance(provenance, dict) and (
        provenance.get("engineering_only") is True or provenance.get("generated") is True
    )
    _verify_fold_identity(result, engineering_only=engineering_only)
    _verify_fold_reconstruction(result)
    _verify_fold_summary(summary, result)
    return result


def decision_json(value: PrimaryDecision) -> dict[str, object]:
    """Convert a dataclass decision to JSON-compatible content."""
    return {"schema": "frem-s12-primary-decision-v1", **asdict(value)}
