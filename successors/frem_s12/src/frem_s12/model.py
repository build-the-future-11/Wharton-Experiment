"""Causal multiscale reservoir, engram experts, and learned delayed router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frem_s12.errors import NumericalError
from frem_s12.scores import Law, combine_laws, energy_weight_gradient

EXPERTS = ("short", "medium", "long", "context", "tail_regime")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Frozen P0 model settings."""

    fourier_width: int = 32
    reservoir_width: int = 32
    spectral_radius: float = 0.90
    input_scale: float = 0.50
    engram_capacity: int = 120
    context_neighbors: int = 24
    medium_half_life: float = 24.0
    router_hidden: int = 16
    router_learning_rate: float = 0.003
    router_weight_cap: float = 0.60
    router_entropy_penalty: float = 0.002
    minimum_feedback: int = 24
    gradient_clip: float = 5.0

    def __post_init__(self) -> None:
        """Reject settings outside the bounded implementation contract."""
        if not 8 <= self.fourier_width <= 256 or not 8 <= self.reservoir_width <= 256:
            raise NumericalError("feature or reservoir width is invalid")
        if not 0 < self.spectral_radius < 1 or not 0 < self.input_scale <= 2:
            raise NumericalError("reservoir scaling is invalid")
        if not 60 <= self.engram_capacity <= 240 or not 8 <= self.context_neighbors < self.engram_capacity:
            raise NumericalError("engram settings are invalid")
        if not 4 <= self.router_hidden <= 128 or self.router_learning_rate <= 0:
            raise NumericalError("router settings are invalid")
        if not 1 / len(EXPERTS) <= self.router_weight_cap <= 1:
            raise NumericalError("router cap is invalid")
        if self.minimum_feedback < 12 or self.gradient_clip <= 0:
            raise NumericalError("feedback or gradient settings are invalid")


@dataclass(frozen=True, slots=True)
class ExpertBundle:
    """Five causal expert laws and their decision state."""

    laws: tuple[Law, ...]
    state: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own state and enforce expert count."""
        state = np.array(self.state, dtype=np.float64, copy=True)
        if len(self.laws) != len(EXPERTS) or state.ndim != 1 or not np.isfinite(state).all():
            raise NumericalError("expert bundle is invalid")
        state.setflags(write=False)
        object.__setattr__(self, "state", state)


class FourierReservoir:
    """Fixed random Fourier recurrent state with spectral normalization."""

    def __init__(self, dimensions: int, config: ModelConfig, seed: int) -> None:
        """Initialize deterministic fixed matrices."""
        rng = np.random.default_rng(seed)
        self.config = config
        self.fourier_weights = rng.normal(0, 1, (config.fourier_width, dimensions))
        self.fourier_bias = rng.uniform(0, 2 * np.pi, config.fourier_width)
        recurrent = rng.normal(0, 1 / np.sqrt(config.reservoir_width), (config.reservoir_width,) * 2)
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        if not np.isfinite(radius) or radius <= 0:
            raise NumericalError("reservoir spectral normalization failed")
        self.recurrent = recurrent * (config.spectral_radius / radius)
        self.input_weights = rng.normal(
            0,
            1 / np.sqrt(config.fourier_width),
            (config.reservoir_width, config.fourier_width),
        )

    def states(self, returns: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return states before each outcome and after the complete prefix."""
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise NumericalError("reservoir inputs are invalid")
        states = np.zeros((len(values) + 1, self.config.reservoir_width), dtype=np.float64)
        for index, row in enumerate(values):
            prior = values[max(0, index - 120) : index]
            if len(prior) < 12:
                center = np.zeros(values.shape[1])
                scale = np.ones(values.shape[1])
            else:
                center = np.median(prior, axis=0)
                scale = np.median(np.abs(prior - center), axis=0)
                scale = np.where(scale > 1e-6, scale, 1.0)
            standardized = np.clip((row - center) / scale, -10, 10)
            features = np.sqrt(2 / self.config.fourier_width) * np.cos(
                self.fourier_weights @ standardized + self.fourier_bias
            )
            states[index + 1] = np.tanh(
                self.recurrent @ states[index] + self.config.input_scale * (self.input_weights @ features)
            )
        if not np.isfinite(states).all():
            raise NumericalError("reservoir state is nonfinite")
        return states


def _uniform(atoms: NDArray[np.float64]) -> Law:
    return Law(atoms, np.full(len(atoms), 1 / len(atoms)))


def _recency(atoms: NDArray[np.float64], half_life: float) -> Law:
    age = np.arange(len(atoms) - 1, -1, -1, dtype=np.float64)
    weights = np.exp2(-age / half_life)
    return Law(atoms, weights / weights.sum())


def issue_experts(history: NDArray[np.float64], *, config: ModelConfig, seed: int) -> ExpertBundle:
    """Issue five laws using only the supplied matured prefix."""
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) < config.engram_capacity + 12:
        raise NumericalError("expert history is insufficient or malformed")
    reservoir = FourierReservoir(6, config, seed)
    states = reservoir.states(values)
    start = len(values) - config.engram_capacity
    atoms = values[start:]
    contexts = states[start : len(values)]
    current = states[-1]
    center = np.median(contexts, axis=0)
    scale = np.median(np.abs(contexts - center), axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    distance = np.linalg.norm((contexts - current) / scale, axis=1)
    nearest = np.argsort(distance, kind="stable")[: config.context_neighbors]
    context_weights = np.zeros(len(atoms), dtype=np.float64)
    context_weights[nearest] = 1 / len(nearest)
    market = atoms.mean(axis=1)
    volatility = atoms.std(axis=1, ddof=1)
    latest_market = float(values[-12:].mean(axis=1).mean())
    latest_volatility = float(values[-12:].std(axis=1, ddof=1).mean())
    same = (market < 0) == (latest_market < 0)
    same &= (volatility >= np.median(volatility)) == (latest_volatility >= np.median(volatility))
    if same.sum() < 12:
        same = (market < 0) == (latest_market < 0)
    tail_weights = np.full(len(atoms), 0.10 / len(atoms), dtype=np.float64)
    tail_weights[same] += 0.90 / int(same.sum())
    laws = (
        _uniform(atoms[-24:]),
        _recency(atoms[-60:], config.medium_half_life),
        _uniform(atoms),
        Law(atoms, context_weights),
        Law(atoms, tail_weights / tail_weights.sum()),
    )
    return ExpertBundle(laws, current)


def s11_weights(loss_history: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the frozen delayed capped S11 score stacker to five matched experts."""
    losses = np.asarray(loss_history, dtype=np.float64)
    equal = np.full(len(EXPERTS), 1 / len(EXPERTS))
    if len(losses) < 24:
        return equal
    history = losses[-120:]
    age = np.arange(len(history) - 1, -1, -1, dtype=np.float64)
    recency = np.exp2(-age / 36.0)
    recency /= recency.sum()
    centered = history - history.mean(axis=1, keepdims=True)
    scale = float(np.median(np.abs(centered)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return equal
    logits = -(recency @ centered) / scale
    fitted = np.exp(logits - logits.max())
    fitted /= fitted.sum()
    fitted = 0.75 * fitted + 0.25 * equal
    cap = 0.60
    while fitted.max() > cap + 1e-12:
        index = int(fitted.argmax())
        excess = float(fitted[index] - cap)
        fitted[index] = cap
        mask = np.arange(len(fitted)) != index
        fitted[mask] += excess * fitted[mask] / fitted[mask].sum()
    return fitted / fitted.sum()


def router_features(state: NDArray[np.float64], loss_history: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build causal router inputs from state and matured expert scores."""
    losses = np.asarray(loss_history, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[1] != len(EXPERTS):
        raise NumericalError("router loss history shape is invalid")
    if len(losses):
        history = losses[-60:]
        age = np.arange(len(history) - 1, -1, -1, dtype=np.float64)
        weights = np.exp2(-age / 24.0)
        weights /= weights.sum()
        means = weights @ history
        regrets = means - float(means.mean())
        dispersion = np.sqrt(weights @ (history - means) ** 2)
        latest = float(history[-1].std())
    else:
        regrets = np.zeros(len(EXPERTS))
        dispersion = np.zeros(len(EXPERTS))
        latest = 0.0
    result = np.concatenate([state, regrets, dispersion, [float(np.std(state)), latest]])
    if not np.isfinite(result).all():
        raise NumericalError("router features are nonfinite")
    return result


class CausalRouter:
    """One-hidden-layer online neural router with exact energy gradient."""

    def __init__(self, input_width: int, config: ModelConfig, seed: int) -> None:
        """Initialize parameters and Adam state deterministically."""
        self.config = config
        self.steps = 0
        rng = np.random.default_rng(seed)
        self.a = rng.normal(0, 0.05, (config.router_hidden, input_width))
        self.c = np.zeros(config.router_hidden)
        self.b = rng.normal(0, 0.05, (len(EXPERTS), config.router_hidden))
        self.d = np.zeros(len(EXPERTS))
        self.m = {name: np.zeros_like(value) for name, value in self._parameters().items()}
        self.v = {name: np.zeros_like(value) for name, value in self._parameters().items()}

    def _parameters(self) -> dict[str, NDArray[np.float64]]:
        return {"a": self.a, "c": self.c, "b": self.b, "d": self.d}

    def _forward(
        self, features: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
        hidden = np.tanh(self.a @ features + self.c)
        logits = self.b @ hidden + self.d
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        count = len(EXPERTS)
        alpha = (self.config.router_weight_cap - 1 / count) / (1 - 1 / count)
        weights = alpha * probabilities + (1 - alpha) / count
        return hidden, probabilities, weights, alpha

    def weights(self, features: NDArray[np.float64], feedback_count: int) -> NDArray[np.float64]:
        """Return constrained weights, using equal weights during burn-in."""
        if feedback_count < self.config.minimum_feedback:
            return np.full(len(EXPERTS), 1 / len(EXPERTS))
        _, _, weights, _ = self._forward(features)
        return weights

    def update(
        self,
        features: NDArray[np.float64],
        laws: tuple[Law, ...],
        weights: NDArray[np.float64],
        outcome: NDArray[np.float64],
        *,
        enabled: bool,
    ) -> None:
        """Update after outcome maturity using exact mixture-score derivative."""
        if not enabled:
            return
        hidden, probabilities, expected, alpha = self._forward(features)
        if not np.allclose(expected, weights, atol=1e-12, rtol=0):
            raise NumericalError("router update weights differ from issued forecast")
        gradient_weights = energy_weight_gradient(laws, weights, outcome)
        gradient_weights += self.config.router_entropy_penalty * (np.log(weights) + 1)
        gradient_probabilities = alpha * gradient_weights
        gradient_logits = probabilities * (
            gradient_probabilities - float(gradient_probabilities @ probabilities)
        )
        gradients = {
            "b": np.outer(gradient_logits, hidden),
            "d": gradient_logits,
        }
        hidden_gradient = self.b.T @ gradient_logits * (1 - hidden**2)
        gradients["a"] = np.outer(hidden_gradient, features)
        gradients["c"] = hidden_gradient
        norm = float(np.sqrt(sum(float(np.sum(value**2)) for value in gradients.values())))
        if not np.isfinite(norm):
            raise NumericalError("router gradient is nonfinite")
        if norm > self.config.gradient_clip:
            gradients = {
                name: value * (self.config.gradient_clip / norm) for name, value in gradients.items()
            }
        self.steps += 1
        learning_rate = self.config.router_learning_rate
        for name, parameter in self._parameters().items():
            gradient = gradients[name]
            self.m[name] = 0.9 * self.m[name] + 0.1 * gradient
            self.v[name] = 0.999 * self.v[name] + 0.001 * gradient**2
            corrected_m = self.m[name] / (1 - 0.9**self.steps)
            corrected_v = self.v[name] / (1 - 0.999**self.steps)
            parameter -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
            if not np.isfinite(parameter).all():
                raise NumericalError("router parameter became nonfinite")

    def state_dict(self) -> dict[str, Any]:
        """Return complete serializable optimizer and parameter state."""
        return {
            "steps": self.steps,
            **{name: value.tolist() for name, value in self._parameters().items()},
            "m": {name: value.tolist() for name, value in self.m.items()},
            "v": {name: value.tolist() for name, value in self.v.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact state after validating every array shape."""
        self.steps = int(state["steps"])
        for name, parameter in self._parameters().items():
            restored = np.asarray(state[name], dtype=np.float64)
            if restored.shape != parameter.shape or not np.isfinite(restored).all():
                raise NumericalError("router checkpoint parameter is invalid")
            parameter[:] = restored
            for slot in (self.m, self.v):
                moment = np.asarray(state["m" if slot is self.m else "v"][name], dtype=np.float64)
                if moment.shape != parameter.shape or not np.isfinite(moment).all():
                    raise NumericalError("router checkpoint moment is invalid")
                slot[name] = moment.copy()


def routed_law(laws: tuple[Law, ...], weights: NDArray[np.float64]) -> Law:
    """Return the exact routed finite mixture."""
    return combine_laws(laws, weights)
