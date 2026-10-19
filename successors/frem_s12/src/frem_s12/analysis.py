"""Frozen, seed-aware hierarchical P0 decision reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from frem_s12.errors import ContractError
from frem_s12.experiment import ARMS, Fold, FoldResult


@dataclass(frozen=True, slots=True)
class PrimaryDecision:
    """Auditable P0 primary and dependence disposition."""

    status: str
    comparisons: dict[str, dict[str, float | list[float]]]
    positive_fold_fraction: dict[str, float]
    family_mean_differences: dict[str, dict[str, float]]
    complete_results: int
    aggregation: dict[str, object]


Series = dict[str, dict[str, NDArray[np.float64]]]
GroupedSeries = dict[str, dict[str, list[NDArray[np.float64]]]]
LabelGroupedSeries = dict[str, GroupedSeries]


def _hierarchical_draws(series: Series, *, block: int, draws: int, seed: int) -> NDArray[np.float64]:
    """Resample folds and time blocks after averaging algorithmic seeds.

    Families remain fixed and equally weighted because the three portfolios
    share the same market months and are not independent population draws.
    Fold identities and circular month blocks are sampled jointly across
    families so contemporaneous cross-family dependence is retained.
    """
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    families = sorted(series)
    folds = sorted(series[families[0]])
    if any(sorted(series[family]) != folds for family in families):
        raise ContractError("bootstrap families do not share identical folds")
    if any(
        len(series[family][fold]) != len(series[families[0]][fold]) for family in families for fold in folds
    ):
        raise ContractError("bootstrap families do not share identical month counts")
    for draw in range(draws):
        selected = rng.integers(0, len(folds), len(folds))
        fold_means: list[float] = []
        for selected_index in selected:
            fold = folds[int(selected_index)]
            length = len(series[families[0]][fold])
            count = int(np.ceil(length / block))
            starts = rng.integers(0, length, count)
            indices = (starts[:, None] + np.arange(block)) % length
            flat = indices.reshape(-1)[:length]
            fold_means.append(float(np.mean([series[family][fold][flat].mean() for family in families])))
        output[draw] = float(np.mean(fold_means))
    return output


def _aggregate(series: Series) -> float:
    return float(
        np.mean([np.mean([values.mean() for values in folds.values()]) for folds in series.values()])
    )


def _seed_averaged(
    grouped: dict[str, dict[str, list[NDArray[np.float64]]]],
) -> Series:
    return {
        family: {fold: np.mean(np.stack(values), axis=0) for fold, values in folds.items()}
        for family, folds in grouped.items()
    }


def _validate_seed_outcomes(
    results: tuple[FoldResult, ...], families: tuple[str, ...], folds: tuple[Fold, ...]
) -> None:
    """Require every algorithmic seed to share identical observed fold data."""
    for family in families:
        for fold in folds:
            identifier = fold.identifier
            matched = [result for result in results if result.family == family and result.fold == identifier]
            reference = matched[0]
            if any(
                not np.array_equal(result.months, reference.months)
                or not np.array_equal(result.outcomes, reference.outcomes)
                or not np.array_equal(result.validation_months, reference.validation_months)
                or not np.array_equal(result.validation_outcomes, reference.validation_outcomes)
                for result in matched[1:]
            ):
                raise ContractError("algorithmic seeds do not share identical fold outcomes")


def decide(results: tuple[FoldResult, ...], *, draws: int = 10_000, seed: int = 1912) -> PrimaryDecision:
    """Apply P0 using equal families/folds and non-independent seed averaging."""
    if not results or draws < 1000:
        raise ContractError("primary decision requires complete results and sufficient draws")
    identities = {(result.family, result.fold, result.seed) for result in results}
    if len(identities) != len(results):
        raise ContractError("duplicate fixed-matrix result identity")
    from frem_s12.campaign import FAMILIES, FOLDS, SEEDS

    expected = {
        (family, fold.identifier, algorithm_seed)
        for family in FAMILIES
        for fold in FOLDS
        for algorithm_seed in SEEDS
    }
    if identities != expected:
        raise ContractError("primary decision requires the exact frozen 120-cell matrix")
    _validate_seed_outcomes(results, FAMILIES, FOLDS)

    candidate_index = ARMS.index("s12_router")
    labels = ("s11_stacker", "non_frem")
    differences: LabelGroupedSeries = {label: {} for label in labels}
    candidate_energy: LabelGroupedSeries = {label: {} for label in labels}
    control_energy: LabelGroupedSeries = {label: {} for label in labels}
    candidate_variogram: LabelGroupedSeries = {label: {} for label in labels}
    control_variogram: LabelGroupedSeries = {label: {} for label in labels}

    for result in results:
        controls = {
            "s11_stacker": ARMS.index("s11_stacker"),
            "non_frem": ARMS.index(result.selected_non_frem),
        }
        for label, control_index in controls.items():
            registries = (
                (differences, result.scores[:, control_index] - result.scores[:, candidate_index]),
                (candidate_energy, result.scores[:, candidate_index]),
                (control_energy, result.scores[:, control_index]),
                (candidate_variogram, result.variograms[:, candidate_index]),
                (control_variogram, result.variograms[:, control_index]),
            )
            for registry, values in registries:
                registry[label].setdefault(result.family, {}).setdefault(result.fold, []).append(values)

    reports: dict[str, dict[str, float | list[float]]] = {}
    p_values: dict[str, float] = {}
    primary_pass: dict[str, bool] = {}
    positive_fraction: dict[str, float] = {}
    family_values: dict[str, dict[str, float]] = {family: {} for family in FAMILIES}
    for offset, label in enumerate(labels):
        difference_series = _seed_averaged(differences[label])
        candidate_series = _seed_averaged(candidate_energy[label])
        control_series = _seed_averaged(control_energy[label])
        candidate_v_series = _seed_averaged(candidate_variogram[label])
        control_v_series = _seed_averaged(control_variogram[label])
        bootstrap = _hierarchical_draws(difference_series, block=12, draws=draws, seed=seed + offset)
        candidate = _aggregate(candidate_series)
        control = _aggregate(control_series)
        if control <= 0:
            raise ContractError("control energy must be positive")
        relative = (control - candidate) / control
        low, high = np.quantile(bootstrap, (0.025, 0.975))
        p_value = float((1 + np.sum(bootstrap <= 0)) / (draws + 1))
        control_variogram_mean = _aggregate(control_v_series)
        if control_variogram_mean <= 0:
            raise ContractError("control variogram must be positive")
        variogram_relative = (
            _aggregate(candidate_v_series) - control_variogram_mean
        ) / control_variogram_mean
        fold_means = [
            float(np.mean([difference_series[family][fold.identifier].mean() for family in FAMILIES]))
            for fold in FOLDS
        ]
        fraction = float(np.mean(np.asarray(fold_means) > 0))
        for family in FAMILIES:
            family_values[family][label] = float(
                np.mean([values.mean() for values in difference_series[family].values()])
            )
        family_ok = all(family_values[family][label] >= 0 for family in FAMILIES)
        primary_pass[label] = bool(
            relative >= 0.05 and low > 0 and fraction >= 0.75 and family_ok and variogram_relative <= 0.01
        )
        positive_fraction[label] = fraction
        p_values[label] = p_value
        reports[label] = {
            "candidate_mean_energy": candidate,
            "control_mean_energy": control,
            "mean_energy_difference": control - candidate,
            "relative_improvement": relative,
            "interval": [float(low), float(high)],
            "one_sided_bootstrap_p": p_value,
            "variogram_relative_degradation": float(variogram_relative),
        }

    ordered = sorted((p_values[label], label) for label in labels)
    holm_pass: dict[str, bool] = {}
    prior_passed = True
    for index, (value, label) in enumerate(ordered):
        threshold = 0.05 / (len(ordered) - index)
        prior_passed = prior_passed and value <= threshold
        holm_pass[label] = prior_passed
        reports[label]["holm_threshold"] = threshold
        reports[label]["holm_pass"] = float(prior_passed)
    status = (
        "SUPPORTED_DEVELOPMENT_ONLY"
        if all(primary_pass.values()) and all(holm_pass.values())
        else "NOT_SUPPORTED_OR_INCONCLUSIVE_CLOSE_P0"
    )
    return PrimaryDecision(
        status,
        reports,
        positive_fraction,
        family_values,
        len(results),
        {
            "unit": "family_equal_then_fold_equal_then_month_equal",
            "algorithm_seeds": "averaged_within_family_fold_before_inference",
            "bootstrap": "joint_cross_family_fold_resample_plus_shared_circular_12_month_blocks",
            "positive_fraction_unit": "eight_folds_after_equal_family_and_seed_averaging",
        },
    )
