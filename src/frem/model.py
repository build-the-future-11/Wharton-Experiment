"""Causal Fourier--Reservoir Engram predictor.

This is a prospective implementation, not a claim of financial predictability.
The online predictor accepts context only. Future observations are accepted by
``encode_target`` exclusively in the training objective.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MEMORY_FEATURES = 7


@dataclass
class ModelConfig:
    input_dim: int
    context_length: int = 64
    horizon: int = 5
    hidden_dim: int = 64
    reservoir_dim: int = 64
    reservoir_leak: float = 0.3
    reservoir_contraction: float = 0.9
    reservoir_seed: int = 0
    spectral_bands: int = 8
    memory_capacity: int = 512
    memory_neighbors: int = 8
    memory_ngram: int = 3
    memory_same_asset: bool = True
    memory_exact_ngram: bool = False
    memory_retention: str = "utility"
    memory_surprise_threshold: float = 0.0
    ngram_feature_index: int = 0
    dropout: float = 0.1
    scale_floor: float = 1e-4
    initial_step_scale: float = 0.01
    use_reservoir: bool = True
    use_spectral: bool = True
    use_memory: bool = True
    use_ngram: bool = True
    use_jepa: bool = True
    use_feature_gate: bool = True
    use_stochastic: bool = True
    use_adaptive_controller: bool = True
    residual_xgboost: bool = False
    xgboost_residual_weight: float = 0.25
    xgboost_residual_clip: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "reservoir_leak",
            "reservoir_contraction",
            "memory_surprise_threshold",
            "dropout",
            "scale_floor",
            "initial_step_scale",
            "xgboost_residual_weight",
            "xgboost_residual_clip",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        integer_fields = (
            "input_dim",
            "context_length",
            "horizon",
            "hidden_dim",
            "reservoir_dim",
            "spectral_bands",
            "memory_capacity",
            "memory_neighbors",
            "memory_ngram",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.context_length < 4:
            raise ValueError("context_length must be at least four observations")
        if self.memory_ngram > min(self.context_length, 8):
            raise ValueError("memory_ngram must be <= context_length and <= 8")
        if not 0 <= self.ngram_feature_index < self.input_dim:
            raise ValueError("ngram_feature_index is outside the feature array")
        if not 0 < self.reservoir_leak <= 1:
            raise ValueError("reservoir_leak must lie in (0, 1]")
        if not 0 < self.reservoir_contraction < 1:
            raise ValueError("reservoir_contraction must lie strictly in (0, 1)")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.scale_floor <= 0 or self.initial_step_scale <= 0:
            raise ValueError("Predictive scales must be strictly positive")
        if self.memory_retention not in {"utility", "fifo", "random", "uniform"}:
            raise ValueError("memory_retention must be utility, fifo, random, or uniform")
        if self.memory_surprise_threshold < 0:
            raise ValueError("memory_surprise_threshold must be nonnegative")
        if not self.use_spectral and self.memory_surprise_threshold > 0:
            raise ValueError("A positive spectral-surprise write threshold requires use_spectral=True")
        if not 0 <= self.xgboost_residual_weight <= 1:
            raise ValueError("xgboost_residual_weight must be in [0, 1]")
        if self.xgboost_residual_clip <= 0:
            raise ValueError("xgboost_residual_clip must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FixedReservoir(nn.Module):
    """Fixed leaky reservoir with an operator-norm contraction guarantee.

    ``||W||_2 < 1`` is sufficient here; scaling the eigenvalue radius alone would
    not establish the same one-step Euclidean contraction bound. The reservoir
    is reset at the start of EACH supplied trailing window, including inference.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, leak: float = 0.3, contraction: float = 0.9, seed: int = 0
    ) -> None:
        super().__init__()
        if not 0 < leak <= 1 or not 0 < contraction < 1:
            raise ValueError("Reservoir requires 0 < leak <= 1 and 0 < contraction < 1")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        recurrent = torch.randn(hidden_dim, hidden_dim, generator=generator)
        norm = torch.linalg.matrix_norm(recurrent, ord=2).clamp_min(1e-12)
        recurrent = recurrent * (contraction / norm)
        incoming = torch.randn(hidden_dim, input_dim, generator=generator)
        incoming = incoming / math.sqrt(input_dim)
        bias = torch.empty(hidden_dim).uniform_(-0.1, 0.1, generator=generator)
        self.register_buffer("recurrent", recurrent)
        self.register_buffer("incoming", incoming)
        self.register_buffer("bias", bias)
        self.leak = float(leak)
        self.contraction = float(contraction)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

    @property
    def contraction_bound(self) -> float:
        return 1 - self.leak + self.leak * self.contraction

    def forward(self, x: Tensor, initial_state: Tensor | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim or x.shape[1] < 1:
            raise ValueError("Reservoir input must have shape [batch, time, input_dim]")
        state = x.new_zeros(x.shape[0], self.hidden_dim) if initial_state is None else initial_state
        if state.shape != (x.shape[0], self.hidden_dim):
            raise ValueError("initial_state has an incompatible reservoir shape")
        states = []
        for observation in x.unbind(dim=1):
            drive = F.linear(observation, self.incoming) + F.linear(state, self.recurrent)
            state = (1 - self.leak) * state + self.leak * torch.tanh(drive + self.bias)
            states.append(state)
        return torch.stack(states, dim=1)


class CausalFeatureGate(nn.Module):
    """Input-channel and time relevance gates using only each observed prefix.

    'Spatial' here means input channels. Cross-asset peer factors must be
    constructed separately with point-in-time availability; this is not a GNN.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.feature = nn.Linear(2 * input_dim, input_dim)
        self.temporal = nn.Conv1d(input_dim, 1, kernel_size=3)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        denominator = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        prefix_mean = x.cumsum(dim=1) / denominator[None, :, None]
        feature_gate = torch.sigmoid(self.feature(torch.cat((x, prefix_mean), dim=-1)))
        temporal_gate = torch.sigmoid(self.temporal(F.pad(x.transpose(1, 2), (2, 0))).transpose(1, 2))
        return x * (2 * feature_gate) * (2 * temporal_gate), feature_gate, temporal_gate


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, use_feature_gate: bool) -> None:
        super().__init__()
        self.gate = CausalFeatureGate(input_dim) if use_feature_gate else None
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.pool = nn.Linear(hidden_dim, 1)
        self.project = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.gate is None:
            gated = x
            feature_gate = torch.ones_like(x)
            temporal_gate = x.new_ones(x.shape[0], x.shape[1], 1)
        else:
            gated, feature_gate, temporal_gate = self.gate(x)
        sequence, final = self.gru(gated)
        weights = torch.softmax(self.pool(sequence), dim=1)
        pooled = (weights * sequence).sum(dim=1)
        latent = self.project(torch.cat((final[-1], pooled), dim=-1))
        return latent, feature_gate, temporal_gate


class CausalSpectralEncoder(nn.Module):
    """Trailing-window DFT, train-reference surprise, and learnable band weights.

    This modifies weighting and retention around an ordinary DFT; it does not
    introduce a new Fourier transform. No centered/full-series filter is used.
    """

    def __init__(self, input_dim: int, context_length: int, bands: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_length = context_length
        self.bands = bands
        frequencies = context_length // 2 + 1
        if bands > frequencies:
            raise ValueError("spectral_bands exceeds the number of real FFT frequency bins")
        self.register_buffer("window", torch.hann_window(context_length, periodic=False))
        assignment = torch.linspace(0, frequencies, bands + 1).long()
        pool = torch.zeros(bands, frequencies)
        for index in range(bands):
            lo, hi = int(assignment[index]), int(assignment[index + 1])
            pool[index, lo:hi] = 1.0 / max(hi - lo, 1)
        self.register_buffer("pool", pool)
        self.register_buffer("reference_mean", torch.zeros(bands, input_dim))
        self.register_buffer("reference_variance", torch.ones(bands, input_dim))
        self.register_buffer("reference_count", torch.tensor(0, dtype=torch.long))
        self.band_logits = nn.Parameter(torch.zeros(bands, 1))

    @property
    def descriptor_dim(self) -> int:
        return 3 * self.bands * self.input_dim

    @property
    def output_dim(self) -> int:
        return 4 * self.bands * self.input_dim

    def raw_descriptor(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.context_length, self.input_dim):
            raise ValueError("Spectral input must match the configured trailing window")
        centered = x - x.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(centered * self.window[None, :, None], dim=1, norm="ortho")
        power = torch.einsum("kf,bfd->bkd", self.pool, spectrum.abs().square())
        log_power = torch.log1p(power)
        real = torch.einsum("kf,bfd->bkd", self.pool, spectrum.real)
        imaginary = torch.einsum("kf,bfd->bkd", self.pool, spectrum.imag)
        descriptor = torch.stack((log_power, real, imaginary), dim=2)
        return descriptor, log_power

    @torch.no_grad()
    def fit_reference(self, batches: Any) -> None:
        """Fit once from an iterable of training-context tensors only."""
        if int(self.reference_count.item()) != 0:
            raise RuntimeError("Spectral reference is already fitted; instantiate a fresh fold")
        total = torch.zeros_like(self.reference_mean, dtype=torch.float64)
        squares = torch.zeros_like(total)
        count = 0
        for x in batches:
            _, power = self.raw_descriptor(x)
            power = power.double()
            total += power.sum(dim=0)
            squares += power.square().sum(dim=0)
            count += power.shape[0]
        if count == 0:
            raise ValueError("Cannot fit a spectral reference on empty training data")
        mean = total / count
        variance = (squares / count - mean.square()).clamp_min(1e-6)
        self.reference_mean.copy_(mean.to(self.reference_mean.dtype))
        self.reference_variance.copy_(variance.to(self.reference_variance.dtype))
        self.reference_count.fill_(count)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw, power = self.raw_descriptor(x)
        if int(self.reference_count.item()) == 0:
            raise RuntimeError("Fit the spectral reference on training contexts before prediction")
        surprise = (
            (power - self.reference_mean).abs() / self.reference_variance.sqrt().clamp_min(1e-3)
        ).clamp_max(20)
        weights = 2 * torch.sigmoid(self.band_logits)
        weighted = (raw * weights[None, :, None, :]).flatten(start_dim=1)
        weighted_surprise = (surprise * weights[None, :, :]).flatten(start_dim=1)
        return torch.cat((weighted, weighted_surprise), dim=-1), raw.flatten(1), surprise.mean((1, 2))


class FREMModel(nn.Module):
    """JEPA-inspired temporal encoder with fixed-key episodic residual memory."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = TemporalEncoder(
            config.input_dim, config.hidden_dim, config.dropout, config.use_feature_gate
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.jepa_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
        )
        self.reservoir = (
            FixedReservoir(
                config.input_dim,
                config.reservoir_dim,
                config.reservoir_leak,
                config.reservoir_contraction,
                config.reservoir_seed,
            )
            if config.use_reservoir
            else None
        )
        self.spectral = (
            CausalSpectralEncoder(config.input_dim, config.context_length, config.spectral_bands)
            if config.use_spectral
            else None
        )
        fusion_dim = config.hidden_dim
        if self.reservoir is not None:
            fusion_dim += config.reservoir_dim
        if self.spectral is not None:
            fusion_dim += self.spectral.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )
        self.return_head = nn.Linear(config.hidden_dim, 3)
        nn.init.zeros_(self.return_head.weight)
        with torch.no_grad():
            self.return_head.bias.copy_(
                torch.tensor(
                    [0.0, math.log(math.expm1(config.initial_step_scale)), math.log(math.expm1(5.0 - 2.1))]
                )
            )
        self.memory_gate_head = nn.Sequential(
            nn.Linear(config.hidden_dim + MEMORY_FEATURES, 16), nn.GELU(), nn.Linear(16, 1)
        )
        self.register_buffer("memory_expert_weight", torch.tensor(1.0))
        self.register_buffer(
            "deterministic_scale", torch.tensor(config.initial_step_scale * math.sqrt(config.horizon))
        )
        self.register_buffer("ngram_thresholds", torch.tensor([-1.0, -0.25, 0.25, 1.0]))

    @property
    def key_dim(self) -> int:
        config = self.config
        # Raw trailing summaries provide a fixed fallback even with both specialist
        # branches disabled. The scaler is frozen after training as well.
        dimension = 2 * config.input_dim
        if self.reservoir is not None:
            dimension += config.reservoir_dim
        if self.spectral is not None:
            dimension += self.spectral.descriptor_dim
        if config.use_ngram:
            dimension += 5 * config.memory_ngram
        return dimension

    def train(self, mode: bool = True) -> FREMModel:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _validate_x(self, x: Tensor) -> None:
        if x.ndim != 3 or x.shape[1:] != (self.config.context_length, self.config.input_dim):
            raise ValueError("x must have shape [batch, configured context_length, input_dim]")
        if not torch.isfinite(x).all():
            raise ValueError("Model context must be finite after train-only preprocessing")

    def _raw_keys(
        self, x: Tensor, reservoir_sequence: Tensor | None, raw_spectral: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        pieces = [x[:, -1], x.mean(dim=1)]
        if reservoir_sequence is not None:
            pieces.append(reservoir_sequence[:, -1])
        if raw_spectral is not None:
            pieces.append(raw_spectral)
        codes = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if self.config.use_ngram:
            values = x[:, -self.config.memory_ngram :, self.config.ngram_feature_index].contiguous()
            symbols = torch.bucketize(values, self.ngram_thresholds)
            pieces.append(F.one_hot(symbols, num_classes=5).to(x.dtype).flatten(1))
            for column in symbols.unbind(dim=1):
                codes = 5 * codes + column
        # Each component is normalized separately so one high-dimensional block
        # does not dominate simply because it has more features.
        key = torch.cat([F.normalize(component, dim=-1, eps=1e-8) for component in pieces], dim=-1)
        return F.normalize(key, dim=-1, eps=1e-8).detach(), codes.detach()

    @torch.no_grad()
    def make_keys(self, x: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_x(x)
        reservoir_sequence = self.reservoir(x) if self.reservoir is not None else None
        raw_spectral = self.spectral.raw_descriptor(x)[0].flatten(1) if self.spectral is not None else None
        return self._raw_keys(x, reservoir_sequence, raw_spectral)

    def forward(
        self, x: Tensor, asset_id: Tensor | None = None, memory_context: Tensor | None = None
    ) -> dict[str, Tensor]:
        self._validate_x(x)
        if asset_id is not None and asset_id.shape != (x.shape[0],):
            raise ValueError("asset_id must contain one identifier per context row")
        latent, feature_gate, temporal_gate = self.context_encoder(x)
        features = [latent]
        reservoir_sequence = self.reservoir(x) if self.reservoir is not None else None
        if reservoir_sequence is not None:
            features.append(reservoir_sequence[:, -1])
        raw_spectral = None
        surprise = x.new_zeros(x.shape[0])
        if self.spectral is not None:
            spectral, raw_spectral, surprise = self.spectral(x)
            features.append(spectral)
        fused = self.fusion(torch.cat(features, dim=-1))
        raw = self.return_head(fused)
        # H is a count of executable return intervals. This is a discrete-time
        # drift/scale parameterization, not a fitted continuous-time market law.
        base_prediction = raw[:, 0] * self.config.horizon
        scale = F.softplus(raw[:, 1]) * math.sqrt(self.config.horizon) + self.config.scale_floor
        df = 2.1 + F.softplus(raw[:, 2])
        if not self.config.use_stochastic:
            scale = self.deterministic_scale.clamp_min(self.config.scale_floor).expand_as(base_prediction)
            df = torch.full_like(scale, 1e6)
        if memory_context is None:
            memory_context = x.new_zeros(x.shape[0], MEMORY_FEATURES)
        if memory_context.shape != (x.shape[0], MEMORY_FEATURES):
            raise ValueError(f"memory_context must have shape [batch, {MEMORY_FEATURES}]")
        if not torch.isfinite(memory_context).all():
            raise ValueError("Memory summaries must be finite")
        availability = (memory_context[:, 4] > 0).to(x.dtype)
        gate = x.new_zeros(x.shape[0])
        if self.config.use_memory:
            gate = torch.sigmoid(self.memory_gate_head(torch.cat((fused, memory_context), dim=-1))).squeeze(
                -1
            )
            gate = gate * availability * self.memory_expert_weight.clamp(0, 1)
        correction = gate * memory_context[:, 0]
        prediction = base_prediction + correction
        key, ngram_code = self._raw_keys(x, reservoir_sequence, raw_spectral)
        return {
            "prediction": prediction,
            "base_prediction": base_prediction,
            "memory_expert_prediction": base_prediction + availability * memory_context[:, 0],
            "scale": scale,
            "df": df,
            "latent": latent,
            "predicted_latent": self.jepa_predictor(latent),
            "key": key,
            "ngram_code": ngram_code,
            "memory_gate": gate,
            "spectral_surprise": surprise,
            "feature_gate": feature_gate,
            "temporal_gate": temporal_gate,
        }

    @torch.no_grad()
    def encode_target(self, future_x: Tensor) -> Tensor:
        if future_x.ndim != 3 or future_x.shape[1:] != (self.config.horizon, self.config.input_dim):
            raise ValueError("future_x must have shape [batch, configured horizon, input_dim]")
        if not torch.isfinite(future_x).all():
            raise ValueError("Future target windows must be finite after preprocessing")
        self.target_encoder.eval()
        return self.target_encoder(future_x)[0]

    @torch.no_grad()
    def update_ema(self, momentum: float) -> None:
        if not 0 <= momentum < 1:
            raise ValueError("EMA momentum must be in [0, 1)")
        for target, online in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            target.mul_(momentum).add_(online, alpha=1 - momentum)
        for target, online in zip(self.target_encoder.buffers(), self.context_encoder.buffers(), strict=True):
            target.copy_(online)


def representation_losses(
    predicted: Tensor, target: Tensor, context: Tensor, variance_target: float = 0.5
) -> dict[str, Tensor]:
    """Latent prediction plus VICReg-style variance and covariance penalties.

    Singleton batches cannot identify empirical variance/covariance; those terms
    are explicitly skipped. Accumulate multiple time groups in the trainer when
    possible. These regularizers reduce a collapse risk; they do not prove that
    learned representations contain useful information.
    """
    jepa = F.mse_loss(F.normalize(predicted, dim=-1), F.normalize(target.detach(), dim=-1))
    if context.shape[0] < 2:
        zero = context.sum() * 0
        return {
            "jepa": jepa,
            "variance": zero,
            "covariance": zero,
            "regularizer_samples": context.new_tensor(0),
        }
    centered = context - context.mean(dim=0, keepdim=True)
    std = torch.sqrt(context.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(variance_target - std).mean()
    covariance_matrix = centered.T @ centered / (context.shape[0] - 1)
    diagonal = torch.diag_embed(torch.diagonal(covariance_matrix))
    covariance = (covariance_matrix - diagonal).square().sum() / context.shape[1]
    return {
        "jepa": jepa,
        "variance": variance,
        "covariance": covariance,
        "regularizer_samples": context.new_tensor(context.shape[0]),
    }


def predictive_loss(output: dict[str, Tensor], target: Tensor, stochastic: bool = True) -> Tensor:
    if target.shape != output["prediction"].shape:
        raise ValueError("Return targets must align with prediction rows")
    if not torch.isfinite(target).all():
        raise ValueError("Supervised targets contain unavailable or non-finite returns")
    if stochastic:
        return (
            -torch.distributions.StudentT(output["df"], output["prediction"], output["scale"])
            .log_prob(target)
            .mean()
        )
    return F.mse_loss(output["prediction"], target)
