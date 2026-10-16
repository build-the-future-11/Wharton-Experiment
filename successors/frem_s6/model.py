"""Numpy S6 distribution forecaster with a trained predictive latent objective.

All times are finite increasing numeric UTC ordinals supplied by the data adapter.
Feature zero must be a historical log return. Future windows and targets are used
only in fit; their availability must strictly precede the declared fit cutoff.
This is an operational JEPA-style objective, not the image JEPA architecture.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .kernel import (
    DelayedEngramMemory,
    QuantizedRippleUpdater,
    ReservoirConfig,
    ReservoirEngramKernel,
    TrainOnlyScaler,
)


def finite(value, name):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def date_weights(times):
    times = finite(times, "dates")
    if times.ndim != 1 or not len(times):
        raise ValueError("dates must be a nonempty vector")
    _, inverse, counts = np.unique(times, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def student_logpdf(y, mean, scale, df):
    y, mean, scale, df = np.broadcast_arrays(*[finite(v, "distribution") for v in (y, mean, scale, df)])
    if (scale <= 0).any() or (df <= 2).any():
        raise ValueError("Student scales >0 and degrees of freedom >2 required")
    lgamma = np.vectorize(math.lgamma, otypes=[float])
    # logaddexp avoids squaring arbitrarily large residuals.
    with np.errstate(divide="ignore"):
        tail = np.logaddexp(0, 2 * np.log(np.abs(y - mean)) - 2 * np.log(scale) - np.log(df))
    return lgamma((df + 1) / 2) - lgamma(df / 2) - .5 * np.log(df * np.pi) - np.log(scale) - (df + 1) / 2 * tail


@dataclass
class MixtureForecast:
    mean: np.ndarray
    scale: np.ndarray
    df: np.ndarray
    weights: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self):
        self.mean, self.scale, self.df, self.weights = [finite(x, "mixture") for x in
                                                     (self.mean, self.scale, self.df, self.weights)]
        if self.mean.ndim != 2 or not all(x.shape == self.mean.shape for x in
                                          (self.scale, self.df, self.weights)):
            raise ValueError("mixture arrays must have identical [row,expert] shape")
        if len(self.names) != self.mean.shape[1] or len(set(self.names)) != len(self.names):
            raise ValueError("expert names must be unique and complete")
        if (self.scale <= 0).any() or (self.df <= 2).any() or (self.weights < 0).any():
            raise ValueError("invalid Student mixture parameters")
        if not np.allclose(self.weights.sum(1), 1, atol=1e-12, rtol=0):
            raise ValueError("expert weights must sum to one")

    def nll(self, y):
        y = finite(y, "target")
        if y.shape != (len(self.mean),):
            raise ValueError("target row mismatch")
        with np.errstate(divide="ignore"):
            logp = student_logpdf(y[:, None], self.mean, self.scale, self.df) + np.log(self.weights)
        top = logp.max(1)
        return -(top + np.log(np.exp(logp - top[:, None]).sum(1)))

    def log_return_moments(self):
        mean = np.sum(self.weights * self.mean, axis=1)
        variance = np.sum(self.weights * (self.scale**2 * self.df / (self.df - 2)
                                         + (self.mean - mean[:, None])**2), axis=1)
        return mean, variance


def latent_loss_grad(features, future, targets, weights, encoder, predictor, head,
                     target_encoder, *, jepa_weight=.1, variance_weight=.01, variance_floor=.2):
    """Exact manual derivatives of supervised + stopped-target JEPA + variance loss.

    Target encoder is fixed during differentiation. E[weights]=a normalized sum.
    The weighted variance regularizer discourages collapse but does not prove
    non-collapse; the achieved latent variances are retained in training history.
    """
    z = np.tanh(features @ encoder)
    target = np.tanh(future @ target_encoder)
    prediction = z @ predictor
    residual = z @ head - targets
    jerr = prediction - target
    dim = z.shape[1]
    mean = np.sum(weights[:, None] * z, axis=0)
    centered = z - mean
    std = np.sqrt(np.sum(weights[:, None] * centered**2, axis=0) + 1e-8)
    deficit = np.maximum(variance_floor - std, 0)
    loss = (.5 * np.sum(weights * residual**2)
            + .5 * jepa_weight * np.sum(weights[:, None] * jerr**2) / dim
            + .5 * variance_weight * np.mean(deficit**2))
    dz = weights[:, None] * residual[:, None] * head
    dz += jepa_weight / dim * weights[:, None] * (jerr @ predictor.T)
    dz -= variance_weight / dim * weights[:, None] * centered * (deficit / std)
    ge = features.T @ (dz * (1 - z**2))
    gp = jepa_weight / dim * z.T @ (weights[:, None] * jerr)
    gh = z.T @ (weights * residual)
    return float(loss), (ge, gp, gh), std


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    state_dim: int = 8
    latent_dim: int = 8
    rff_dim: int = 8
    epochs: int = 30
    learning_rate: float = .02
    jepa_weight: float = .1
    variance_weight: float = .01
    ema: float = .95
    ridge: float = 1.0
    df: float = 5.0
    seed: int = 11
    ripple_delay: int = 1
    ripple_steps: int = 3
    quantization_step: float = 1e-4
    use_xgboost: bool = False
    context: str = "full"

    def __post_init__(self):
        for name in ("input_dim", "state_dim", "latent_dim", "rff_dim", "epochs", "ripple_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.ripple_delay) is not int or self.ripple_delay < 0:
            raise ValueError("ripple delay must be nonnegative")
        vals = finite([self.learning_rate, self.jepa_weight, self.variance_weight, self.ema,
                       self.ridge, self.df, self.quantization_step], "model config")
        if (vals[[0, 4]] <= 0).any() or (vals[[1, 2, 6]] < 0).any() or not 0 <= self.ema < 1 or self.df <= 2:
            raise ValueError("invalid model hyperparameters")
        if self.context not in ("full", "half", "zero"):
            raise ValueError("context must be full, half, or zero")


class S6Forecaster:
    """Fit -> calibrate on later data -> issue -> delayed observe, by whole date.

    The training budget/configuration is fixed before fit, with no outer-result
    selection. Optional XGBoost is a real histogram-tree expert and fails loudly
    if unavailable. XGBoost is never imported for the default low-memory path.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.kernel = ReservoirEngramKernel(ReservoirConfig(config.input_dim, state_dim=config.state_dim,
                                                           rff_dim=config.rff_dim, seed=config.seed))
        self.scaler = TrainOnlyScaler()
        self.names = ("zero", "ridge", "trend", "adaptive_volatility", "reservoir_jepa")
        if config.use_xgboost:
            self.names += ("xgboost_hist",)
        self.fitted = False
        self.calibrated = False
        self.issued = {}
        self.observed = set()
        self.history = []
        self.latest_issue = None

    def _windows(self, windows):
        x = finite(windows, "windows")
        if x.ndim != 3 or len(x) == 0 or x.shape[2] != self.config.input_dim or x.shape[1] < 2:
            raise ValueError("nonempty [row,time>=2,feature] windows required")
        return x

    def _features(self, windows):
        x = self._windows(windows)
        z = self.scaler.transform(x)
        context = x.shape[1] // 2 if self.config.context == "half" else self.config.context
        features = self.kernel.encode(z, context=context)
        keep = x.shape[1] // 2
        effective = x[:, -keep:] if self.config.context == "half" else x
        if self.config.context == "zero":
            effective = np.zeros_like(x)
            z = np.zeros_like(z)
        elif self.config.context == "half":
            z = z[:, -keep:]
        simple = np.concatenate([z[:, -1], z.mean(1), np.ones((len(x), 1))], axis=1)
        return features, simple, effective

    def fit(self, windows, future_windows, y, *, origin_times, label_available,
            future_available, fit_cutoff):
        if self.fitted:
            raise ValueError("model already fitted; use a new instance for a new fit")
        x, future = self._windows(windows), self._windows(future_windows)
        y, origins, labels, future_times = [finite(v, "training metadata") for v in
                                           (y, origin_times, label_available, future_available)]
        n = len(x)
        if any(v.shape != (n,) for v in (y, origins, labels, future_times)) or future.shape != x.shape:
            raise ValueError("training rows must match")
        if not math.isfinite(fit_cutoff) or n < 4 or (origins >= fit_cutoff).any():
            raise ValueError("invalid fit cutoff or too few training rows")
        if (labels < origins).any() or (future_times < origins).any():
            raise ValueError("future targets cannot precede their origin")
        if max(labels.max(), future_times.max()) >= fit_cutoff:
            raise ValueError("training target availability must precede fit cutoff")
        self.scaler.fit(x, row_times=origins, train_end=fit_cutoff)
        features, simple, _ = self._features(x)
        fnext, _, _ = self._features(future)
        weights = date_weights(origins)
        self.target_center = float(weights @ y)
        self.target_scale = max(float(np.sqrt(weights @ (y - self.target_center)**2)), 1e-6)
        target = (y - self.target_center) / self.target_scale
        # Regularized least squares for a distinct simple context expert.
        self.ridge_coef = np.linalg.solve(simple.T @ (weights[:, None] * simple)
                                         + self.config.ridge * np.eye(simple.shape[1]),
                                         simple.T @ (weights * target))
        rng = np.random.default_rng(self.config.seed)
        self.encoder = rng.normal(size=(features.shape[1], self.config.latent_dim)) / np.sqrt(features.shape[1])
        self.predictor = np.eye(self.config.latent_dim)
        self.head = np.zeros(self.config.latent_dim)
        self.target_encoder = self.encoder.copy()
        params = [self.encoder, self.predictor, self.head]
        self.updaters = [QuantizedRippleUpdater(p.shape, learning_rate=self.config.learning_rate,
                                               quantization_step=self.config.quantization_step,
                                               delay_steps=self.config.ripple_delay,
                                               ripple_steps=self.config.ripple_steps) for p in params]
        for step in range(self.config.epochs):
            loss, gradients, std = latent_loss_grad(features, fnext, target, weights, *params,
                                                   self.target_encoder, jepa_weight=self.config.jepa_weight,
                                                   variance_weight=self.config.variance_weight)
            if not math.isfinite(loss):
                raise ValueError("nonfinite training loss")
            for i, (update, grad) in enumerate(zip(self.updaters, gradients, strict=True)):
                update.observe(grad, step=step)
                params[i] = update.apply(params[i], step=step)
            self.target_encoder = self.config.ema * self.target_encoder + (1 - self.config.ema) * params[0]
            self.history.append({"step": step, "loss": loss, "min_latent_std": float(std.min())})
        self.encoder, self.predictor, self.head = params
        self.pending_update_policy = "Retain queued update state; do not flush gradients after declared epochs"
        if self.config.use_xgboost:
            from xgboost import XGBRegressor
            self.tree = XGBRegressor(n_estimators=32, max_depth=2, learning_rate=.05,
                                     objective="reg:squarederror", tree_method="hist", max_bin=32,
                                     n_jobs=1, random_state=self.config.seed)
            self.tree.fit(simple, target, sample_weight=weights * len(weights))
        self.fit_cutoff = float(fit_cutoff)
        self.scales = np.full(len(self.names), self.target_scale)
        self.memory = DelayedEngramMemory(features.shape[1], len(self.names), capacity=60, bandwidth=1.0)
        self.fitted = True
        return self

    def _experts(self, windows):
        if not self.fitted:
            raise ValueError("fit required")
        features, simple, effective = self._features(windows)
        neural = np.tanh(features @ self.encoder) @ self.head
        # Trend is an explicit historical feature; no inference of contemporaneous availability.
        trend = effective[:, :, 0].mean(1)
        means = [np.zeros(len(features)), self.target_center + self.target_scale * (simple @ self.ridge_coef),
                 trend, np.zeros(len(features)), self.target_center + self.target_scale * neural]
        if self.config.use_xgboost:
            means.append(self.target_center + self.target_scale * self.tree.predict(simple))
        means = np.column_stack(means)
        scales = np.broadcast_to(self.scales, means.shape).copy()
        # EWMA conditional volatility uses only the visible historical return sequence.
        decay = .94 ** np.arange(effective.shape[1] - 1, -1, -1)
        decay /= decay.sum()
        rms = np.sqrt(np.sum(effective[:, :, 0]**2 * decay, axis=1))
        scales[:, 3] = np.maximum(rms * np.sqrt((self.config.df - 2) / self.config.df), self.scales[3] * .25)
        return features, means, scales

    def calibrate(self, windows, y, *, origin_times, maturity_times, available_times):
        if not self.fitted or self.calibrated:
            raise ValueError("exactly one calibration after fit required")
        x = self._windows(windows)
        y, origin, maturity, available = [finite(v, "calibration metadata") for v in
                                         (y, origin_times, maturity_times, available_times)]
        if any(v.shape != (len(x),) for v in (y, origin, maturity, available)):
            raise ValueError("calibration row mismatch")
        if (origin <= self.fit_cutoff).any() or (maturity <= origin).any() or (available < maturity).any():
            raise ValueError("calibration chronology violated")
        _, means, _ = self._experts(x)
        w = date_weights(origin)
        rms = np.sqrt(np.sum(w[:, None] * (y[:, None] - means)**2, axis=0))
        self.scales = np.maximum(rms * np.sqrt((self.config.df - 2) / self.config.df), 1e-6)
        features, means, scales = self._experts(x)
        losses = -student_logpdf(y[:, None], means, scales, self.config.df)
        self.recent_losses = []
        dates = sorted(np.unique(origin), key=lambda date: float(available[origin == date].max()))
        for date in dates:
            idx = origin == date
            self.memory.add(features[idx].mean(0), losses[idx].mean(0), origin_time=float(date),
                            maturity_time=float(maturity[idx].max()), available_time=float(available[idx].max()))
            self.recent_losses.append((float(available[idx].max()), losses[idx].mean(0).tolist()))
        self.calibration_cutoff = float(available.max())
        self.calibrated = True
        return self

    def forecast(self, windows, *, issue_time, asset_ids, router="memory"):
        if not self.calibrated or not math.isfinite(issue_time) or issue_time <= self.calibration_cutoff:
            raise ValueError("forecast must follow complete calibration availability")
        if router not in ("memory", "equal", "recent"):
            raise ValueError("unknown router")
        if issue_time in self.observed or (self.latest_issue is not None and issue_time < self.latest_issue):
            raise ValueError("forecast issue times must advance; observed dates are closed")
        x = self._windows(windows)
        assets = tuple(asset_ids)
        if len(assets) != len(x) or len(set(assets)) != len(assets) or not all(isinstance(a, str) and a for a in assets):
            raise ValueError("each date requires unique nonempty asset IDs")
        features, means, scales = self._experts(x)
        key = features.mean(0)
        e = len(self.names)
        weights = np.ones(e) / e
        if router == "memory":
            retrieved = self.memory.retrieve(key[None, :], query_time=issue_time)
            ess = float(retrieved.effective_count[0])
            loss = retrieved.values[0]
        else:
            rows = [v for t, v in sorted(self.recent_losses, key=lambda item: item[0]) if t < issue_time][-24:]
            ess = float(len(rows)) if router == "recent" else 0.0
            loss = np.mean(rows, axis=0) if rows else np.zeros(e)
        if ess > 0:
            scores = np.exp(-(loss - np.min(loss)))
            adaptive = scores / scores.sum()
            strength = ess / (ess + 12)
            weights = (1 - strength) * weights + strength * adaptive
        result = MixtureForecast(means, scales, np.full_like(means, self.config.df),
                                 np.broadcast_to(weights, means.shape).copy(), self.names)
        # One registered issue per date. Repeating a control forecast must not replace issued experts.
        if issue_time not in self.issued and issue_time not in self.observed:
            self.issued[float(issue_time)] = {"assets": assets, "key": key.tolist(),
                                            "means": means.tolist(), "scales": scales.tolist()}
        elif issue_time in self.issued:
            old = self.issued[issue_time]
            if old["assets"] != assets or not np.array_equal(old["means"], means) or not np.array_equal(old["scales"], scales):
                raise ValueError("a forecast date cannot be reissued with changed inputs/assets")
        self.latest_issue = float(issue_time)
        mature = sorted((r for r in self.recent_losses if r[0] < issue_time), key=lambda r: r[0])[-60:]
        self.recent_losses = mature + [r for r in self.recent_losses if r[0] >= issue_time]
        return result

    def observe(self, *, issue_time, y, asset_ids, maturity_times, available_times):
        if issue_time not in self.issued or issue_time in self.observed:
            raise ValueError("only a complete previously issued date can be observed once")
        record = self.issued[issue_time]
        if tuple(asset_ids) != tuple(record["assets"]):
            raise ValueError("complete original asset order required")
        y, maturity, available = [finite(v, "feedback") for v in (y, maturity_times, available_times)]
        if any(v.shape != (len(record["assets"]),) for v in (y, maturity, available)):
            raise ValueError("feedback row mismatch")
        if (maturity <= issue_time).any() or (available < maturity).any():
            raise ValueError("feedback chronology violated")
        loss = -student_logpdf(y[:, None], record["means"], record["scales"], self.config.df).mean(0)
        self.memory.add(record["key"], loss, origin_time=issue_time, maturity_time=float(maturity.max()),
                        available_time=float(available.max()))
        self.recent_losses.append((float(available.max()), loss.tolist()))
        self.observed.add(issue_time)
        del self.issued[issue_time]

    def save(self, path):
        if not self.calibrated:
            raise ValueError("only a fitted/calibrated forecaster can be exported")
        state = {"version": 1, "config": asdict(self.config), "scaler": self.scaler.state_dict(),
                 "kernel": self.kernel.state_dict(), "memory": self.memory.state_dict(),
                 "updaters": [u.state_dict() for u in self.updaters], "history": self.history,
                 "target_center": self.target_center, "target_scale": self.target_scale,
                 "fit_cutoff": self.fit_cutoff, "calibration_cutoff": self.calibration_cutoff,
                 "recent_losses": self.recent_losses, "observed": sorted(self.observed),
                 "issued": self.issued, "latest_issue": self.latest_issue,
                 "pending_update_policy": self.pending_update_policy}
        for name in ("ridge_coef", "encoder", "predictor", "head", "target_encoder", "scales"):
            state[name] = getattr(self, name).tolist()
        if self.config.use_xgboost:
            state["tree"] = base64.b64encode(self.tree.get_booster().save_raw(raw_format="json")).decode()
        with Path(path).open("x") as stream:
            json.dump(state, stream, allow_nan=False)

    @classmethod
    def load(cls, path):
        s = json.loads(Path(path).read_text())
        if s.get("version") != 1:
            raise ValueError("unsupported checkpoint")
        result = cls(ModelConfig(**s["config"]))
        result.scaler = TrainOnlyScaler.from_state_dict(s["scaler"])
        result.kernel = ReservoirEngramKernel.from_state_dict(s["kernel"])
        result.memory = DelayedEngramMemory.from_state_dict(s["memory"])
        result.updaters = [QuantizedRippleUpdater.from_state_dict(u) for u in s["updaters"]]
        for name in ("ridge_coef", "encoder", "predictor", "head", "target_encoder", "scales"):
            setattr(result, name, finite(s[name], name))
        for name in ("history", "target_center", "target_scale", "fit_cutoff", "calibration_cutoff",
                     "recent_losses", "pending_update_policy", "latest_issue"):
            setattr(result, name, s[name])
        result.observed = set(s["observed"])
        result.issued = {float(k): {**v, "assets": tuple(v["assets"])} for k, v in s["issued"].items()}
        if result.config.use_xgboost:
            from xgboost import XGBRegressor
            result.tree = XGBRegressor()
            result.tree.load_model(bytearray(base64.b64decode(s["tree"])))
        result.fitted = result.calibrated = True
        result._validate_checkpoint()
        return result

    def _validate_checkpoint(self):
        c = self.config
        d = self.kernel.output_dim
        expected = {"ridge_coef": (2 * c.input_dim + 1,), "encoder": (d, c.latent_dim),
                    "predictor": (c.latent_dim, c.latent_dim), "head": (c.latent_dim,),
                    "target_encoder": (d, c.latent_dim), "scales": (len(self.names),)}
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"checkpoint {name} shape mismatch")
        finite([self.target_center, self.target_scale, self.fit_cutoff, self.calibration_cutoff], "checkpoint scalars")
        if self.target_scale <= 0 or (self.scales <= 0).any() or self.calibration_cutoff <= self.fit_cutoff:
            raise ValueError("invalid checkpoint scale or chronology")
        if self.scaler.mean_ is None or self.scaler.mean_.shape != (c.input_dim,):
            raise ValueError("checkpoint scaler shape mismatch")
        if self.scaler.train_end_ != self.fit_cutoff:
            raise ValueError("checkpoint scaler cutoff mismatch")
        kc = self.kernel.config
        if (kc.input_dim, kc.state_dim, kc.rff_dim, kc.seed) != (c.input_dim, c.state_dim, c.rff_dim, c.seed):
            raise ValueError("checkpoint kernel configuration mismatch")
        if self.memory.key_dim != d or self.memory.value_dim != len(self.names):
            raise ValueError("checkpoint memory dimensions mismatch")
        if self.latest_issue is not None and (not math.isfinite(self.latest_issue)
                                             or self.latest_issue <= self.calibration_cutoff):
            raise ValueError("invalid checkpoint issue clock")
        if len(self.updaters) != 3 or any(u.shape != s for u, s in zip(
                self.updaters, (self.encoder.shape, self.predictor.shape, self.head.shape), strict=True)):
            raise ValueError("checkpoint optimizer shape mismatch")
        for time, losses in self.recent_losses:
            if not math.isfinite(time) or finite(losses, "saved losses").shape != (len(self.names),):
                raise ValueError("invalid saved dated losses")
        for time in self.observed:
            if not math.isfinite(time) or time <= self.calibration_cutoff:
                raise ValueError("invalid observed date")
        for time, record in self.issued.items():
            n = len(record["assets"])
            if not math.isfinite(time) or time <= self.calibration_cutoff or n == 0 or time in self.observed:
                raise ValueError("invalid issued date")
            if len(set(record["assets"])) != n or any(not isinstance(a, str) or not a for a in record["assets"]):
                raise ValueError("invalid issued assets")
            if finite(record["key"], "issued key").shape != (d,):
                raise ValueError("invalid issued key shape")
            means, scales = finite(record["means"], "issued means"), finite(record["scales"], "issued scales")
            if means.shape != (n, len(self.names)) or scales.shape != means.shape or (scales <= 0).any():
                raise ValueError("invalid issued expert distribution")
