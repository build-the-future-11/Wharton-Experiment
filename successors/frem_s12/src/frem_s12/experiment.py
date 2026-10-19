"""Causal fold execution with complete forecast and policy retention."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frem_s12.baselines import CausalMLP, gaussian_law, student_t_law, var_law
from frem_s12.data import ReturnPanel
from frem_s12.errors import ContractError
from frem_s12.model import (
    EXPERTS,
    CausalRouter,
    ModelConfig,
    issue_experts,
    routed_law,
    router_features,
    s11_weights,
)
from frem_s12.policy import choose_action, realize
from frem_s12.scores import Law, energy_score, variogram_score

ARMS = (
    "s12_router",
    "expert_equal",
    "s11_stacker",
    "recent",
    "gaussian",
    "student_t",
    "var",
    "causal_mlp",
    "router_off",
)
NON_FREM = ("gaussian", "student_t", "var", "causal_mlp")


@dataclass(frozen=True, slots=True)
class Fold:
    """One nested chronological validation/test fold."""

    identifier: str
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    def __post_init__(self) -> None:
        """Require strict validation-before-test chronology."""
        if (
            not self.identifier
            or not self.validation_start <= self.validation_end < self.test_start <= self.test_end
        ):
            raise ContractError("fold chronology is invalid")


def _readonly_float(value: object, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError(f"fold result {name} is invalid")
    array.setflags(write=False)
    return array


def _validated_forecasts(
    atoms_value: object,
    probabilities_value: object,
    lengths_value: object,
    rows: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    atoms = np.array(atoms_value, dtype=np.float64, copy=True)
    probabilities = np.array(probabilities_value, dtype=np.float64, copy=True)
    lengths = np.array(lengths_value, dtype=np.int64, copy=True)
    if atoms.ndim != 4 or atoms.shape[:2] != (rows, len(ARMS)) or atoms.shape[3] != 6:
        raise ContractError("fold result forecast atoms are invalid")
    if probabilities.shape != atoms.shape[:3] or lengths.shape != (rows, len(ARMS)):
        raise ContractError("fold result forecast probability shapes are invalid")
    if not np.isfinite(atoms).all() or not np.isfinite(probabilities).all():
        raise ContractError("fold result forecast law is nonfinite")
    if np.any(lengths < 1) or np.any(lengths > atoms.shape[2]):
        raise ContractError("fold result forecast lengths are invalid")
    for row, arm in np.ndindex(lengths.shape):
        length = int(lengths[row, arm])
        active = probabilities[row, arm, :length]
        if np.any(active < 0) or not np.isclose(active.sum(), 1.0, atol=1e-12):
            raise ContractError("fold result forecast probabilities are invalid")
        if np.any(probabilities[row, arm, length:] != 0):
            raise ContractError("fold result forecast padding is invalid")
    for value in (atoms, probabilities, lengths):
        value.setflags(write=False)
    return atoms, probabilities, lengths


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Complete retained forecast and secondary-policy evidence for one fold."""

    family: str
    fold: str
    seed: int
    months: NDArray[np.int64]
    outcomes: NDArray[np.float64]
    validation_months: NDArray[np.int64]
    validation_outcomes: NDArray[np.float64]
    validation_scores: NDArray[np.float64]
    validation_forecast_atoms: NDArray[np.float64]
    validation_forecast_probabilities: NDArray[np.float64]
    validation_forecast_lengths: NDArray[np.int64]
    scores: NDArray[np.float64]
    variograms: NDArray[np.float64]
    expert_scores: NDArray[np.float64]
    router_weights: NDArray[np.float64]
    forecast_atoms: NDArray[np.float64]
    forecast_probabilities: NDArray[np.float64]
    forecast_lengths: NDArray[np.int64]
    wealth: NDArray[np.float64]
    targets: NDArray[np.float64]
    holdings: NDArray[np.float64]
    selected_non_frem: str

    def __post_init__(self) -> None:
        """Own every retained array and validate aligned row counts."""
        rows = len(self.months)
        expected = {
            "outcomes": (rows, 6),
            "scores": (rows, len(ARMS)),
            "variograms": (rows, len(ARMS)),
            "expert_scores": (rows, len(EXPERTS)),
            "router_weights": (rows, len(EXPERTS)),
            "wealth": (rows, len(ARMS)),
            "targets": (rows, len(ARMS), 7),
            "holdings": (rows, len(ARMS), 7),
        }
        for name, shape in expected.items():
            object.__setattr__(self, name, _readonly_float(getattr(self, name), shape, name))
        months = np.array(self.months, dtype=np.int64, copy=True)
        validation_months = np.array(self.validation_months, dtype=np.int64, copy=True)
        validation_outcomes = _readonly_float(
            self.validation_outcomes, (len(validation_months), 6), "validation_outcomes"
        )
        validation_scores = _readonly_float(
            self.validation_scores, (len(validation_months), len(ARMS)), "validation_scores"
        )
        if rows == 0 or np.any(np.diff(months) <= 0) or np.any(np.diff(validation_months) <= 0):
            raise ContractError("fold result months are invalid")
        atoms, probabilities, lengths = _validated_forecasts(
            self.forecast_atoms, self.forecast_probabilities, self.forecast_lengths, rows
        )
        validation_atoms, validation_probabilities, validation_lengths = _validated_forecasts(
            self.validation_forecast_atoms,
            self.validation_forecast_probabilities,
            self.validation_forecast_lengths,
            len(validation_months),
        )
        for name in ("targets", "holdings"):
            values = getattr(self, name)
            if np.any(values < -1e-12) or not np.allclose(values.sum(axis=2), 1.0, atol=1e-10):
                raise ContractError(f"fold result {name} violate the simplex")
        months.setflags(write=False)
        validation_months.setflags(write=False)
        object.__setattr__(self, "months", months)
        object.__setattr__(self, "validation_months", validation_months)
        object.__setattr__(self, "validation_outcomes", validation_outcomes)
        object.__setattr__(self, "validation_scores", validation_scores)
        object.__setattr__(self, "validation_forecast_atoms", validation_atoms)
        object.__setattr__(self, "validation_forecast_probabilities", validation_probabilities)
        object.__setattr__(self, "validation_forecast_lengths", validation_lengths)
        object.__setattr__(self, "forecast_atoms", atoms)
        object.__setattr__(self, "forecast_probabilities", probabilities)
        object.__setattr__(self, "forecast_lengths", lengths)
        if self.selected_non_frem not in NON_FREM:
            raise ContractError("selected non-FREM control is invalid")


def _pack_forecasts(
    laws_by_origin: list[tuple[Law, ...]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Pack ragged finite laws without discarding any issued atom or mass."""
    if not laws_by_origin or any(len(row) != len(ARMS) for row in laws_by_origin):
        raise ContractError("forecast registry is empty or incomplete")
    maximum = max(len(law.atoms) for row in laws_by_origin for law in row)
    atoms = np.zeros((len(laws_by_origin), len(ARMS), maximum, 6), dtype=np.float64)
    probabilities = np.zeros((len(laws_by_origin), len(ARMS), maximum), dtype=np.float64)
    lengths = np.empty((len(laws_by_origin), len(ARMS)), dtype=np.int64)
    for row_index, row in enumerate(laws_by_origin):
        for arm_index, law in enumerate(row):
            length = len(law.atoms)
            atoms[row_index, arm_index, :length] = law.atoms
            probabilities[row_index, arm_index, :length] = law.probabilities
            lengths[row_index, arm_index] = length
    return atoms, probabilities, lengths


def _index(panel: ReturnPanel, month: int) -> int:
    index = int(np.searchsorted(panel.months, month))
    if index == len(panel.months) or int(panel.months[index]) != month:
        raise ContractError(f"fold month is absent: {month}")
    return index


def _laws(
    history: NDArray[np.float64],
    expert_laws: tuple[Law, ...],
    router_weights: NDArray[np.float64],
    losses: NDArray[np.float64],
    mlp_law: Law,
    *,
    seed: int,
    origin: int,
) -> tuple[Law, ...]:
    equal = np.full(len(expert_laws), 1 / len(expert_laws))
    return (
        routed_law(expert_laws, router_weights),
        routed_law(expert_laws, equal),
        routed_law(expert_laws, s11_weights(losses)),
        expert_laws[1],
        gaussian_law(history, seed=seed * 100_003 + origin),
        student_t_law(history, seed=seed * 200_003 + origin),
        var_law(history),
        mlp_law,
        routed_law(expert_laws, equal),
    )


def run_fold(
    panel: ReturnPanel,
    fold: Fold,
    *,
    seed: int,
    config: ModelConfig | None = None,
    cost_bps: float = 10.0,
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], object] | None = None,
    checkpoint_every: int = 12,
) -> FoldResult:
    """Run one fold with online updates only after each outcome matures."""
    cfg = config or ModelConfig()
    validation_start = _index(panel, fold.validation_start)
    validation_stop = _index(panel, fold.validation_end) + 1
    test_start = _index(panel, fold.test_start)
    test_stop = _index(panel, fold.test_end) + 1
    if validation_start < cfg.engram_capacity + 12 or validation_stop != test_start:
        raise ContractError("fold lacks history or validation/test are not contiguous")
    if checkpoint_every < 1:
        raise ContractError("checkpoint interval must be positive")
    input_width = cfg.reservoir_width + 2 * len(EXPERTS) + 2
    router = CausalRouter(input_width, cfg, seed)
    mlp = CausalMLP(6, 16, 0.001, seed + 10_000)
    losses: list[list[float]] = []
    validation_months: list[int] = []
    validation_outcomes: list[NDArray[np.float64]] = []
    validation_scores: list[NDArray[np.float64]] = []
    validation_forecast_rows: list[tuple[Law, ...]] = []
    outcome_rows: list[NDArray[np.float64]] = []
    score_rows: list[NDArray[np.float64]] = []
    variogram_rows: list[NDArray[np.float64]] = []
    expert_score_rows: list[NDArray[np.float64]] = []
    router_rows: list[NDArray[np.float64]] = []
    forecast_rows: list[tuple[Law, ...]] = []
    wealth = np.ones(len(ARMS), dtype=np.float64)
    holdings = np.tile(np.asarray([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64), (len(ARMS), 1))
    wealth_rows: list[NDArray[np.float64]] = []
    target_rows: list[NDArray[np.float64]] = []
    holding_rows: list[NDArray[np.float64]] = []
    initial_origin = cfg.engram_capacity + 12
    first_origin = initial_origin
    if resume_state is not None:
        expected_identity = {"family": panel.family, "fold": fold.identifier, "seed": seed}
        if resume_state.get("identity") != expected_identity:
            raise ContractError("checkpoint identity differs from the requested cell")
        first_origin = int(resume_state["next_origin"])
        if not initial_origin <= first_origin <= test_stop:
            raise ContractError("checkpoint origin is outside the cell")
        router.load_state_dict(resume_state["router"])
        mlp.load_state_dict(resume_state["mlp"])
        losses = [[float(item) for item in row] for row in resume_state["losses"]]
        validation_months = [int(value) for value in resume_state["validation_months"]]
        validation_outcomes = [
            np.asarray(row, dtype=np.float64) for row in resume_state["validation_outcomes"]
        ]
        validation_scores = [np.asarray(row, dtype=np.float64) for row in resume_state["validation_scores"]]
        validation_forecast_rows = [
            tuple(
                Law(
                    np.asarray(law["atoms"], dtype=np.float64),
                    np.asarray(law["probabilities"], dtype=np.float64),
                )
                for law in row
            )
            for row in resume_state["validation_forecast_rows"]
        ]
        outcome_rows = [np.asarray(row, dtype=np.float64) for row in resume_state["outcome_rows"]]
        score_rows = [np.asarray(row, dtype=np.float64) for row in resume_state["score_rows"]]
        variogram_rows = [np.asarray(row, dtype=np.float64) for row in resume_state["variogram_rows"]]
        expert_score_rows = [np.asarray(row, dtype=np.float64) for row in resume_state["expert_score_rows"]]
        router_rows = [np.asarray(row, dtype=np.float64) for row in resume_state["router_rows"]]
        forecast_rows = [
            tuple(
                Law(
                    np.asarray(law["atoms"], dtype=np.float64),
                    np.asarray(law["probabilities"], dtype=np.float64),
                )
                for law in row
            )
            for row in resume_state["forecast_rows"]
        ]
        wealth = np.asarray(resume_state["wealth"], dtype=np.float64)
        holdings = np.asarray(resume_state["holdings"], dtype=np.float64)
        wealth_rows = [np.asarray(row, dtype=np.float64) for row in resume_state["wealth_rows"]]
        target_rows = [np.asarray(row, dtype=np.float64) for row in resume_state["target_rows"]]
        holding_rows = [np.asarray(row, dtype=np.float64) for row in resume_state["holding_rows"]]
        if len(losses) != first_origin - initial_origin:
            raise ContractError("checkpoint feedback count is inconsistent")
    for origin in range(first_origin, test_stop):
        history = panel.returns[:origin]
        bundle = issue_experts(history, config=cfg, seed=seed)
        loss_array = np.asarray(losses, dtype=np.float64).reshape(-1, len(EXPERTS))
        features = router_features(bundle.state, loss_array)
        weights = router.weights(features, len(losses))
        mlp_law, mlp_features, mlp_hidden = mlp.predict(history)
        laws = _laws(history, bundle.laws, weights, loss_array, mlp_law, seed=seed, origin=origin)
        outcome = panel.returns[origin]
        scores = np.asarray([energy_score(law, outcome) for law in laws])
        variograms = np.asarray([variogram_score(law, outcome) for law in laws])
        expert_scores = [energy_score(law, outcome) for law in bundle.laws]
        if validation_start <= origin < validation_stop:
            validation_months.append(int(panel.months[origin]))
            validation_outcomes.append(outcome.copy())
            validation_scores.append(scores.copy())
            validation_forecast_rows.append(laws)
        if test_start <= origin < test_stop:
            outcome_rows.append(outcome.copy())
            score_rows.append(scores)
            variogram_rows.append(variograms)
            expert_score_rows.append(np.asarray(expert_scores))
            router_rows.append(weights.copy())
            forecast_rows.append(laws)
            targets = np.empty_like(holdings)
            for index, law in enumerate(laws):
                action = choose_action(law, holdings[index], cost_bps=cost_bps)
                targets[index] = action
                factor, holdings[index] = realize(action, outcome, holdings[index], cost_bps=cost_bps)
                wealth[index] *= factor
            wealth_rows.append(wealth.copy())
            target_rows.append(targets)
            holding_rows.append(holdings.copy())
        router.update(
            features,
            bundle.laws,
            weights,
            outcome,
            enabled=len(losses) >= cfg.minimum_feedback,
        )
        mlp.update(mlp_features, mlp_hidden, outcome)
        losses.append(expert_scores)
        next_origin = origin + 1
        if checkpoint_callback is not None and (
            next_origin == test_stop or (next_origin - initial_origin) % checkpoint_every == 0
        ):
            checkpoint_callback(
                {
                    "schema": "frem-s12-fold-checkpoint-v3",
                    "identity": {"family": panel.family, "fold": fold.identifier, "seed": seed},
                    "next_origin": next_origin,
                    "router": router.state_dict(),
                    "mlp": mlp.state_dict(),
                    "losses": losses,
                    "validation_months": validation_months,
                    "validation_outcomes": [value.tolist() for value in validation_outcomes],
                    "validation_scores": [value.tolist() for value in validation_scores],
                    "validation_forecast_rows": [
                        [
                            {"atoms": law.atoms.tolist(), "probabilities": law.probabilities.tolist()}
                            for law in row
                        ]
                        for row in validation_forecast_rows
                    ],
                    "outcome_rows": [value.tolist() for value in outcome_rows],
                    "score_rows": [value.tolist() for value in score_rows],
                    "variogram_rows": [value.tolist() for value in variogram_rows],
                    "expert_score_rows": [value.tolist() for value in expert_score_rows],
                    "router_rows": [value.tolist() for value in router_rows],
                    "forecast_rows": [
                        [
                            {"atoms": law.atoms.tolist(), "probabilities": law.probabilities.tolist()}
                            for law in row
                        ]
                        for row in forecast_rows
                    ],
                    "wealth": wealth.tolist(),
                    "holdings": holdings.tolist(),
                    "wealth_rows": [value.tolist() for value in wealth_rows],
                    "target_rows": [value.tolist() for value in target_rows],
                    "holding_rows": [value.tolist() for value in holding_rows],
                    "rng": "counter_addressed_by_seed_and_origin",
                }
            )
    validation = np.asarray(validation_scores)
    selected = NON_FREM[int(np.argmin(validation[:, [ARMS.index(name) for name in NON_FREM]].mean(axis=0)))]
    validation_atoms, validation_probabilities, validation_lengths = _pack_forecasts(validation_forecast_rows)
    forecast_atoms, forecast_probabilities, forecast_lengths = _pack_forecasts(forecast_rows)
    return FoldResult(
        panel.family,
        fold.identifier,
        seed,
        panel.months[test_start:test_stop],
        np.asarray(outcome_rows),
        np.asarray(validation_months, dtype=np.int64),
        np.asarray(validation_outcomes),
        validation,
        validation_atoms,
        validation_probabilities,
        validation_lengths,
        np.asarray(score_rows),
        np.asarray(variogram_rows),
        np.asarray(expert_score_rows),
        np.asarray(router_rows),
        forecast_atoms,
        forecast_probabilities,
        forecast_lengths,
        np.asarray(wealth_rows),
        np.asarray(target_rows),
        np.asarray(holding_rows),
        selected,
    )
