"""Bounded feedback rules; this is not learned backpropagation or an RL agent."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ControllerConfig:
    enabled: bool = True
    base_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    lr_factor: float = 0.5
    patience: int = 2
    improvement_tolerance: float = 1e-5
    base_jepa_weight: float = 0.1
    minimum_jepa_fraction: float = 0.25
    expert_rate: float = 0.03
    expert_logit_limit: float = 4.0

    def __post_init__(self) -> None:
        if not 0 < self.min_learning_rate <= self.base_learning_rate:
            raise ValueError("Learning-rate bounds must be positive and ordered")
        if not 0 < self.lr_factor < 1 or self.patience < 1:
            raise ValueError("Controller requires lr_factor in (0,1) and positive patience")
        if self.base_jepa_weight < 0 or not 0 <= self.minimum_jepa_fraction <= 1:
            raise ValueError("Auxiliary-loss weights must be nonnegative and bounded")
        if self.expert_rate < 0 or self.expert_logit_limit <= 0:
            raise ValueError("Expert update parameters must be nonnegative/positive")


class AdaptiveController:
    """Validation-only LR/auxiliary schedules and delayed expert loss weighting.

    Expert weighting uses bounded differences of realized squared error. Loss
    weights and learning rates change only at explicit validation boundaries.
    Online mode updates the expert weight after outcome availability; it never
    silently trains the neural network on the evaluation set.
    """

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.learning_rate = config.base_learning_rate
        self.jepa_weight = config.base_jepa_weight
        self.memory_logit = 0.0
        self.best_validation = math.inf
        self.bad_validations = 0
        self.validation_count = 0
        self.observation_count = 0
        self.last_outcome_availability = -(1 << 63)
        self.history: list[dict[str, Any]] = []

    @property
    def memory_weight(self) -> float:
        if not self.config.enabled:
            return 1.0
        return 1.0 / (1.0 + math.exp(-self.memory_logit))

    def observe_matured(
        self,
        *,
        actual: float,
        base_prediction: float,
        memory_prediction: float,
        scale: float,
        label_time: int,
        as_of_time: int,
        had_memory: bool,
    ) -> None:
        if label_time >= as_of_time:
            raise ValueError("Expert feedback must have matured strictly before decision time")
        if not all(math.isfinite(v) for v in (actual, base_prediction, memory_prediction, scale)):
            raise ValueError("Matured feedback must be finite")
        if scale <= 0:
            raise ValueError("Feedback scale must be positive")
        self.last_outcome_availability = max(self.last_outcome_availability, int(label_time))
        self.observation_count += 1
        if not self.config.enabled or not had_memory:
            return
        denominator = scale * scale + (actual - base_prediction) ** 2 + 1e-12
        improvement = ((actual - base_prediction) ** 2 - (actual - memory_prediction) ** 2) / denominator
        improvement = max(-1.0, min(1.0, improvement))
        limit = self.config.expert_logit_limit
        self.memory_logit = max(-limit, min(limit, self.memory_logit + self.config.expert_rate * improvement))

    def observe_validation(self, score: float) -> dict[str, float]:
        if not math.isfinite(score):
            raise ValueError("A non-finite validation score cannot drive the learning controller")
        self.validation_count += 1
        if score < self.best_validation - self.config.improvement_tolerance:
            self.best_validation = score
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        if self.config.enabled and self.bad_validations >= self.config.patience:
            self.learning_rate = max(
                self.config.min_learning_rate, self.learning_rate * self.config.lr_factor
            )
            self.jepa_weight = max(
                self.config.base_jepa_weight * self.config.minimum_jepa_fraction,
                self.jepa_weight * self.config.lr_factor,
            )
            self.bad_validations = 0
        event = {
            "validation_score": score,
            "learning_rate": self.learning_rate,
            "jepa_weight": self.jepa_weight,
            "memory_weight": self.memory_weight,
        }
        self.history.append({"validation_number": self.validation_count, **event})
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "learning_rate": self.learning_rate,
            "jepa_weight": self.jepa_weight,
            "memory_logit": self.memory_logit,
            "best_validation": None if math.isinf(self.best_validation) else self.best_validation,
            "bad_validations": self.bad_validations,
            "validation_count": self.validation_count,
            "observation_count": self.observation_count,
            "last_outcome_availability": self.last_outcome_availability,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> AdaptiveController:
        controller = cls(ControllerConfig(**state["config"]))
        for key in (
            "learning_rate",
            "jepa_weight",
            "memory_logit",
            "bad_validations",
            "validation_count",
            "observation_count",
            "last_outcome_availability",
            "history",
        ):
            setattr(controller, key, state[key])
        controller.best_validation = (
            math.inf if state["best_validation"] is None else state["best_validation"]
        )
        return controller
