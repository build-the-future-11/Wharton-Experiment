"""Strong finite-law controls and a causal parameter-matched MLP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frem_s12.errors import NumericalError
from frem_s12.scores import Law


def _draws(seed: int, count: int, dimensions: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    half = (count + 1) // 2
    values = rng.normal(size=(half, dimensions))
    return np.concatenate([values, -values], axis=0)[:count]


def gaussian_law(history: NDArray[np.float64], *, seed: int, draws: int = 96) -> Law:
    """Shrinkage multivariate Gaussian represented by deterministic atoms."""
    values = np.asarray(history[-120:], dtype=np.float64)
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.8 * covariance + 0.2 * diagonal + np.eye(values.shape[1]) * 1e-8
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12)))
    atoms = mean + _draws(seed, draws, values.shape[1]) @ root.T
    return Law(atoms, np.full(draws, 1 / draws))


def student_t_law(history: NDArray[np.float64], *, seed: int, draws: int = 96, df: float = 7.0) -> Law:
    """Shrinkage multivariate Student-t finite law."""
    values = np.asarray(history[-120:], dtype=np.float64)
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.8 * covariance + 0.2 * diagonal + np.eye(values.shape[1]) * 1e-8
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12)))
    rng = np.random.default_rng(seed)
    half = (draws + 1) // 2
    normal = rng.normal(size=(half, values.shape[1]))
    scale = np.sqrt(rng.chisquare(df, size=(half, 1)) / df)
    standardized = normal / scale
    standardized = np.concatenate([standardized, -standardized], axis=0)[:draws]
    return Law(mean + standardized @ root.T, np.full(draws, 1 / draws))


def var_law(history: NDArray[np.float64], *, ridge: float = 0.01) -> Law:
    """Ridge VAR(1) mean with joint residual bootstrap law."""
    values = np.asarray(history[-121:], dtype=np.float64)
    x, y = values[:-1], values[1:]
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    residuals = y - design @ coefficients
    mean = np.concatenate([[1.0], values[-1]]) @ coefficients
    atoms = mean + residuals
    return Law(atoms, np.full(len(atoms), 1 / len(atoms)))


@dataclass(slots=True)
class CausalMLP:
    """Online autoregressive MLP control with joint residual scenarios."""

    dimensions: int
    hidden: int
    learning_rate: float
    seed: int
    w1: NDArray[np.float64] = field(init=False, repr=False)
    b1: NDArray[np.float64] = field(init=False, repr=False)
    w2: NDArray[np.float64] = field(init=False, repr=False)
    b2: NDArray[np.float64] = field(init=False, repr=False)
    residuals: list[NDArray[np.float64]] = field(init=False, repr=False)
    steps: int = field(init=False, default=0)
    m: dict[str, NDArray[np.float64]] = field(init=False, repr=False)
    v: dict[str, NDArray[np.float64]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize deterministic parameters and optimizer-free SGD state."""
        rng = np.random.default_rng(self.seed)
        width = self.dimensions * 12
        self.w1 = rng.normal(0, 0.02, (self.hidden, width))
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(0, 0.02, (self.dimensions, self.hidden))
        self.b2 = np.zeros(self.dimensions)
        self.residuals = []
        self.m = {name: np.zeros_like(value) for name, value in self._parameters().items()}
        self.v = {name: np.zeros_like(value) for name, value in self._parameters().items()}

    def _parameters(self) -> dict[str, NDArray[np.float64]]:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def _features(self, history: NDArray[np.float64]) -> NDArray[np.float64]:
        window = np.asarray(history[-12:], dtype=np.float64)
        center = np.median(history[-120:], axis=0)
        scale = np.median(np.abs(history[-120:] - center), axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return np.asarray(np.clip((window - center) / scale, -10, 10).reshape(-1), dtype=np.float64)

    def predict(self, history: NDArray[np.float64]) -> tuple[Law, NDArray[np.float64], NDArray[np.float64]]:
        """Issue a residual-bootstrap law before seeing the next outcome."""
        features = self._features(history)
        hidden = np.tanh(self.w1 @ features + self.b1)
        mean = self.w2 @ hidden + self.b2
        if len(self.residuals) >= 24:
            residuals = np.asarray(self.residuals[-120:])
        else:
            residuals = np.asarray(history[-60:] - history[-60:].mean(axis=0))
        law = Law(mean + residuals, np.full(len(residuals), 1 / len(residuals)))
        return law, features, hidden

    def update(
        self,
        features: NDArray[np.float64],
        hidden: NDArray[np.float64],
        outcome: NDArray[np.float64],
    ) -> None:
        """Update only after the issued outcome matures."""
        # The forecast law is the raw network mean plus an empirical residual
        # distribution.  Its probability-weighted mean can therefore include a
        # finite-sample residual offset.  Optimizing against that shifted law
        # mean would leak the residual-library bias into both the gradient and
        # the next residual.  Train and record residuals against the causal
        # network mean that was actually issued before the outcome matured.
        mean = self.w2 @ hidden + self.b2
        error = mean - outcome
        gradient_w2 = np.outer(error, hidden)
        gradient_b2 = error
        hidden_gradient = self.w2.T @ error * (1 - hidden**2)
        gradients = {
            "w2": gradient_w2,
            "b2": gradient_b2,
            "w1": np.outer(hidden_gradient, features),
            "b1": hidden_gradient,
        }
        norm = float(np.sqrt(sum(float(np.sum(value**2)) for value in gradients.values())))
        if not np.isfinite(norm):
            raise NumericalError("MLP gradient became nonfinite")
        if norm > 5.0:
            gradients = {name: value * (5.0 / norm) for name, value in gradients.items()}
        self.steps += 1
        for name, parameter in self._parameters().items():
            gradient = gradients[name]
            self.m[name] = 0.9 * self.m[name] + 0.1 * gradient
            self.v[name] = 0.999 * self.v[name] + 0.001 * gradient**2
            corrected_m = self.m[name] / (1 - 0.9**self.steps)
            corrected_v = self.v[name] / (1 - 0.999**self.steps)
            parameter -= self.learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        self.residuals.append(np.asarray(outcome - mean, dtype=np.float64))
        self.residuals = self.residuals[-120:]
        if any(not np.isfinite(value).all() for value in (self.w1, self.b1, self.w2, self.b2)):
            raise NumericalError("MLP parameter became nonfinite")

    def state_dict(self) -> dict[str, Any]:
        """Return complete online model, optimizer, and residual state."""
        return {
            "steps": self.steps,
            **{name: value.tolist() for name, value in self._parameters().items()},
            "m": {name: value.tolist() for name, value in self.m.items()},
            "v": {name: value.tolist() for name, value in self.v.items()},
            "residuals": [value.tolist() for value in self.residuals],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a complete state after strict shape and finite checks."""
        steps = int(state["steps"])
        if steps < 0:
            raise NumericalError("MLP checkpoint steps are invalid")
        self.steps = steps
        for name, parameter in self._parameters().items():
            restored = np.asarray(state[name], dtype=np.float64)
            if restored.shape != parameter.shape or not np.isfinite(restored).all():
                raise NumericalError("MLP checkpoint parameter is invalid")
            parameter[:] = restored
            for key, slot in (("m", self.m), ("v", self.v)):
                moment = np.asarray(state[key][name], dtype=np.float64)
                if moment.shape != parameter.shape or not np.isfinite(moment).all():
                    raise NumericalError("MLP checkpoint moment is invalid")
                slot[name] = moment.copy()
        residuals = [np.asarray(value, dtype=np.float64) for value in state["residuals"]]
        if any(value.shape != (self.dimensions,) or not np.isfinite(value).all() for value in residuals):
            raise NumericalError("MLP checkpoint residual is invalid")
        self.residuals = [value.copy() for value in residuals]
