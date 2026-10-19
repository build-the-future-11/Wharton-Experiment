"""Chronology-safe Fold-A-only configuration search and selection."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from frem_s12.campaign import FAMILIES, FOLDS, search_space
from frem_s12.data import ReturnPanel
from frem_s12.errors import ContractError
from frem_s12.evidence import (
    atomic_json,
    atomic_write,
    package_identity,
    read_json,
    sha256_file,
    validate_cell_provenance,
)
from frem_s12.model import EXPERTS, CausalRouter, ModelConfig, issue_experts, routed_law, router_features
from frem_s12.scores import Law, energy_score

SEARCH_SEEDS = (11, 29)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Retained validation-only evidence for one search cell."""

    trial: int
    family: str
    seed: int
    months: NDArray[np.int64]
    outcomes: NDArray[np.float64]
    scores: NDArray[np.float64]
    weights: NDArray[np.float64]
    forecast_atoms: NDArray[np.float64]
    forecast_probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and own search evidence."""
        months = np.array(self.months, dtype=np.int64, copy=True)
        outcomes = np.array(self.outcomes, dtype=np.float64, copy=True)
        scores = np.array(self.scores, dtype=np.float64, copy=True)
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        atoms = np.array(self.forecast_atoms, dtype=np.float64, copy=True)
        probabilities = np.array(self.forecast_probabilities, dtype=np.float64, copy=True)
        if self.trial not in range(len(search_space())) or self.family not in FAMILIES:
            raise ContractError("search identity is invalid")
        if self.seed not in SEARCH_SEEDS or scores.shape != (len(months),):
            raise ContractError("search seed or score shape is invalid")
        if weights.shape != (len(months), len(EXPERTS)) or outcomes.shape != (len(months), 6):
            raise ContractError("search weight shape is invalid")
        if atoms.ndim != 3 or atoms.shape[:1] != (len(months),) or atoms.shape[2] != 6:
            raise ContractError("search forecast atoms are invalid")
        if probabilities.shape != atoms.shape[:2]:
            raise ContractError("search forecast probabilities are invalid")
        if not all(np.isfinite(value).all() for value in (outcomes, scores, weights, atoms, probabilities)):
            raise ContractError("search result contains nonfinite values")
        if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
            raise ContractError("search forecast probability simplex is invalid")
        reconstructed = np.asarray(
            [energy_score(Law(atoms[row], probabilities[row]), outcomes[row]) for row in range(len(months))]
        )
        if not np.allclose(reconstructed, scores, rtol=1e-12, atol=1e-15):
            raise ContractError("search scores do not reconstruct from retained laws")
        for value in (months, outcomes, scores, weights, atoms, probabilities):
            value.setflags(write=False)
        object.__setattr__(self, "months", months)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "forecast_atoms", atoms)
        object.__setattr__(self, "forecast_probabilities", probabilities)


def config_for_trial(trial: int) -> ModelConfig:
    """Materialize one immutable search configuration."""
    try:
        values = search_space()[trial]
    except IndexError as exc:
        raise ContractError("search trial is outside the registry") from exc
    return ModelConfig(
        reservoir_width=int(values["reservoir_width"]),
        router_hidden=int(values["router_hidden"]),
        router_learning_rate=float(values["router_learning_rate"]),
        engram_capacity=int(values["engram_capacity"]),
    )


def run_search_cell(panel: ReturnPanel, *, trial: int, seed: int) -> SearchResult:
    """Score one configuration on Fold A validation without reading test rows."""
    if panel.family not in FAMILIES or seed not in SEARCH_SEEDS:
        raise ContractError("search cell is outside the frozen registry")
    fold = FOLDS[0]
    config = config_for_trial(trial)
    validation_start = int(np.searchsorted(panel.months, fold.validation_start))
    validation_stop = int(np.searchsorted(panel.months, fold.validation_end)) + 1
    if (
        validation_start == len(panel.months)
        or int(panel.months[validation_start]) != fold.validation_start
        or validation_stop > len(panel.months)
        or int(panel.months[validation_stop - 1]) != fold.validation_end
    ):
        raise ContractError("Fold A validation months are absent")
    initial = config.engram_capacity + 12
    if validation_start < initial:
        raise ContractError("search panel lacks training history")
    router = CausalRouter(config.reservoir_width + 2 * len(EXPERTS) + 2, config, seed)
    losses: list[list[float]] = []
    outcomes: list[NDArray[np.float64]] = []
    scores: list[float] = []
    weights_rows: list[NDArray[np.float64]] = []
    forecast_laws: list[Law] = []
    for origin in range(initial, validation_stop):
        history = panel.returns[:origin]
        bundle = issue_experts(history, config=config, seed=seed)
        loss_array = np.asarray(losses, dtype=np.float64).reshape(-1, len(EXPERTS))
        features = router_features(bundle.state, loss_array)
        weights = router.weights(features, len(losses))
        outcome = panel.returns[origin]
        if origin >= validation_start:
            law = routed_law(bundle.laws, weights)
            outcomes.append(outcome.copy())
            scores.append(energy_score(law, outcome))
            weights_rows.append(weights.copy())
            forecast_laws.append(law)
        router.update(
            features,
            bundle.laws,
            weights,
            outcome,
            enabled=len(losses) >= config.minimum_feedback,
        )
        losses.append([energy_score(law, outcome) for law in bundle.laws])
    return SearchResult(
        trial,
        panel.family,
        seed,
        panel.months[validation_start:validation_stop],
        np.asarray(outcomes),
        np.asarray(scores),
        np.asarray(weights_rows),
        np.asarray([law.atoms for law in forecast_laws]),
        np.asarray([law.probabilities for law in forecast_laws]),
    )


def write_search_result(
    output: Path, result: SearchResult, *, provenance: dict[str, object]
) -> dict[str, object]:
    """Write one immutable validation-only search bundle."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("search output already exists")
    output.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        months=result.months,
        outcomes=result.outcomes,
        scores=result.scores,
        weights=result.weights,
        forecast_atoms=result.forecast_atoms,
        forecast_probabilities=result.forecast_probabilities,
    )
    atomic_write(output / "arrays.npz", buffer.getvalue())
    summary = {
        "schema": "frem-s12-search-summary-v3",
        "trial": result.trial,
        "config": search_space()[result.trial],
        "family": result.family,
        "seed": result.seed,
        "fold": "A-validation-only",
        "rows": len(result.months),
        "mean_energy": float(result.scores.mean()),
        "provenance": provenance,
        "source_identity": package_identity()["sha256"],
    }
    atomic_json(output / "summary.json", summary)
    manifest = {
        "schema": "frem-s12-search-manifest-v3",
        "complete": True,
        "files": {
            "arrays.npz": sha256_file(output / "arrays.npz"),
            "summary.json": sha256_file(output / "summary.json"),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def _validate_search_directory(output: Path) -> None:
    """Require the exact safe three-file cell bundle."""
    if not output.is_dir() or output.is_symlink():
        raise ContractError("search artifact root is not a regular directory")
    entries = set(output.iterdir())
    expected_entries = {output / "arrays.npz", output / "summary.json", output / "manifest.json"}
    if entries != expected_entries or any(path.is_symlink() or not path.is_file() for path in entries):
        raise ContractError("search artifact directory inventory is not exact")
    if (output / "arrays.npz").stat().st_size > 512 * 2**20:
        raise ContractError("search artifact array archive exceeds the safety limit")


def _fold_a_validation_months() -> NDArray[np.int64]:
    """Return the exact frozen Fold-A validation calendar."""
    expected_months = []
    current = FOLDS[0].validation_start
    while current <= FOLDS[0].validation_end:
        expected_months.append(current)
        year, month = divmod(current, 100)
        current = (year + 1) * 100 + 1 if month == 12 else year * 100 + month + 1
    return np.asarray(expected_months, dtype=np.int64)


def read_search_result(output: Path) -> SearchResult:
    """Read one search bundle and reconstruct its score from raw finite laws."""
    _validate_search_directory(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "frem-s12-search-manifest-v3" or manifest.get("complete") is not True:
        raise ContractError("search manifest is incomplete")
    if set(manifest.get("files", {})) != {"arrays.npz", "summary.json"}:
        raise ContractError("search manifest inventory is incomplete or foreign")
    for name, digest in manifest["files"].items():
        if sha256_file(output / name) != digest:
            raise ContractError("search artifact hash mismatch")
    summary = read_json(output / "summary.json")
    if summary.get("schema") != "frem-s12-search-summary-v3":
        raise ContractError("search summary schema is unsupported")
    if summary.get("source_identity") != package_identity()["sha256"]:
        raise ContractError("search result is source-stale")
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        result = SearchResult(
            int(summary["trial"]),
            str(summary["family"]),
            int(summary["seed"]),
            arrays["months"],
            arrays["outcomes"],
            arrays["scores"],
            arrays["weights"],
            arrays["forecast_atoms"],
            arrays["forecast_probabilities"],
        )
    if summary.get("config") != search_space()[result.trial]:
        raise ContractError("search summary config differs from the registry")
    if summary.get("fold") != "A-validation-only" or int(summary.get("rows", -1)) != len(result.months):
        raise ContractError("search summary fold or row count is invalid")
    if not np.array_equal(result.months, _fold_a_validation_months()):
        raise ContractError("search calendar differs from Fold A validation")
    if not np.isclose(float(summary["mean_energy"]), result.scores.mean(), rtol=0, atol=1e-15):
        raise ContractError("search summary energy differs from reconstructed raw evidence")
    return result


def select_search(root: Path) -> dict[str, object]:
    """Select one configuration from the complete 144-cell Fold-A registry."""
    expected = {
        (trial, family, seed)
        for trial in range(len(search_space()))
        for family in FAMILIES
        for seed in SEARCH_SEEDS
    }
    if not root.is_dir() or root.is_symlink():
        raise ContractError("search campaign root is not a regular directory")
    expected_files = {
        (Path(f"trial={trial}") / f"family={family}" / f"seed={seed}" / name).as_posix()
        for trial, family, seed in expected
        for name in ("arrays.npz", "summary.json", "manifest.json")
    }
    observed_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ContractError("search campaign contains a symlink or special file")
        if path.is_file():
            observed_files.add(path.relative_to(root).as_posix())
    if observed_files != expected_files:
        raise ContractError("search campaign file inventory is incomplete or foreign")
    values: dict[tuple[int, str, int], float] = {}
    manifest_hashes: dict[str, str] = {}
    environment_hashes: set[str] = set()
    family_gates: dict[str, set[tuple[str, str, str]]] = {family: set() for family in FAMILIES}
    consumption_hashes: set[str] = set()
    identity = package_identity()["sha256"]
    for manifest_path in sorted(root.glob("trial=*/family=*/seed=*/manifest.json")):
        result = read_search_result(manifest_path.parent)
        key = (result.trial, result.family, result.seed)
        expected_path = root / f"trial={result.trial}" / f"family={result.family}" / f"seed={result.seed}"
        if manifest_path.parent != expected_path or key in values:
            raise ContractError("search result is misplaced or duplicate")
        summary = read_json(manifest_path.parent / "summary.json")
        provenance = validate_cell_provenance(
            summary.get("provenance"),
            phase="search",
            identity={"trial": result.trial, "family": result.family, "seed": result.seed},
        )
        environment_hashes.add(provenance["environment_receipt_sha256"])
        family_gates[result.family].add(
            (
                provenance["data_receipt_sha256"],
                provenance["acquisition_receipt_sha256"],
                provenance["archive_sha256"],
            )
        )
        consumption_hashes.add(provenance["consumption_sha256"])
        values[key] = float(result.scores.mean())
        manifest_hashes[manifest_path.relative_to(root).as_posix()] = sha256_file(manifest_path)
    if set(values) != expected:
        raise ContractError(f"search is incomplete: observed={len(values)} expected={len(expected)}")
    if len(environment_hashes) != 1 or any(len(values) != 1 for values in family_gates.values()):
        raise ContractError("search cells do not share one environment and one data gate per family")
    if len(consumption_hashes) != len(expected):
        raise ContractError("search cells do not have unique consumption claims")
    trial_means = {
        trial: float(np.mean([values[(trial, family, seed)] for family in FAMILIES for seed in SEARCH_SEEDS]))
        for trial in range(len(search_space()))
    }
    selected = min(trial_means, key=lambda trial: (trial_means[trial], trial))
    return {
        "schema": "frem-s12-search-selection-v2",
        "complete": True,
        "source_identity": identity,
        "selection_fold": "A-validation-only",
        "cells": len(values),
        "selected_trial": selected,
        "selected_config": search_space()[selected],
        "trial_mean_energy": {str(key): value for key, value in trial_means.items()},
        "cell_manifest_sha256": manifest_hashes,
    }


def config_from_selection(path: Path) -> ModelConfig:
    """Load a complete source-current search selection."""
    value = read_json(path)
    if (
        value.get("schema") != "frem-s12-search-selection-v2"
        or value.get("complete") is not True
        or value.get("source_identity") != package_identity()["sha256"]
        or int(value.get("cells", -1)) != 144
    ):
        raise ContractError("search selection is incomplete or source-stale")
    selected = int(value["selected_trial"])
    if value.get("selected_config") != search_space()[selected]:
        raise ContractError("search selection config differs from the registry")
    manifests = value.get("cell_manifest_sha256")
    expected_manifests = {
        (Path(f"trial={trial}") / f"family={family}" / f"seed={seed}" / "manifest.json").as_posix()
        for trial in range(len(search_space()))
        for family in FAMILIES
        for seed in SEARCH_SEEDS
    }
    if not isinstance(manifests, dict) or set(manifests) != expected_manifests:
        raise ContractError("search selection does not bind every cell manifest")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in manifests.items()
    ):
        raise ContractError("search selection manifest identities are invalid")
    return config_for_trial(selected)
