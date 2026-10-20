"""Strict, serializable model and bounded development contracts (stdlib only)."""

import math
from dataclasses import asdict, dataclass, field

ARMS = (
    "full",
    "no_memory",
    "no_jepa",
    "no_neurogenesis",
    "no_reservoir",
    "no_spectral",
    "uniform_router",
    "no_risk",
    "zero",
    "ridge",
    "xgboost",
    "hybrid_xgboost",
    "fixed_full_capacity",
    "no_ngram",
    "gru",
    "no_spde",
    "no_multihorizon",
    "no_graph",
    "no_mask",
)


def positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 4
    context: int = 32
    horizon: int = 5
    latent: int = 32
    reservoir: int = 48
    initial_neurons: int = 24
    growth_step: int = 8
    growth_every: int = 2
    growth_threshold: float = 0.8
    contraction: float = 0.85
    leaks: tuple = (0.1, 0.4, 0.9)
    bands: int = 4
    memory_capacity: int = 512
    neighbors: int = 8
    memory_temperature: float = 0.2
    memory_half_life: float = 252.0
    ngram: int = 3
    ngram_bonus: float = 0.5
    experts: int = 3
    router_cap: float = 0.8
    student_df: float = 5.0
    ema: float = 0.99
    field_rows: int = 4
    graph_mix: float = 0.25
    mask_fraction: float = 0.25
    arm: str = "full"

    def __post_init__(self):
        for key in (
            "input_dim",
            "context",
            "horizon",
            "latent",
            "reservoir",
            "initial_neurons",
            "growth_step",
            "growth_every",
            "bands",
            "memory_capacity",
            "neighbors",
            "experts",
            "ngram",
            "field_rows",
        ):
            positive_integer(key, getattr(self, key))
        for key in (
            "growth_threshold",
            "contraction",
            "memory_temperature",
            "memory_half_life",
            "router_cap",
            "student_df",
            "ema",
            "ngram_bonus",
            "graph_mix",
            "mask_fraction",
        ):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if self.context < 4 or self.bands > self.context // 2:
            raise ValueError("context must cover the requested nonzero Fourier bands")
        if self.ngram > self.context or self.ngram_bonus < 0:
            raise ValueError("invalid causal ngram length or bonus")
        if self.initial_neurons > self.reservoir or self.neighbors > self.memory_capacity:
            raise ValueError("active neurons / neighbors exceed capacity")
        if not self.leaks or any(not math.isfinite(x) or not 0 < x <= 1 for x in self.leaks):
            raise ValueError("leaks must be finite and in (0, 1]")
        if not 0 < self.contraction < 1 or not 0 < self.ema < 1:
            raise ValueError("contraction and EMA must be in (0, 1)")
        if self.experts < 2 or not 1 / self.experts <= self.router_cap <= 1:
            raise ValueError("router cap is infeasible")
        if min(self.memory_temperature, self.memory_half_life, self.growth_threshold) <= 0:
            raise ValueError("memory and growth scales must be positive")
        if self.student_df <= 2 or self.arm not in ARMS:
            raise ValueError("finite-variance Student law / registered arm required")
        if self.latent % self.field_rows or self.latent // self.field_rows < 2:
            raise ValueError("latent field must have >=2 columns and divide latent width")
        if self.field_rows < 2 or not 0 <= self.graph_mix <= 1:
            raise ValueError("two-dimensional field / convex graph mixing required")
        if not 0 < self.mask_fraction < 1:
            raise ValueError("mask fraction must be in (0,1)")

    @property
    def target_offsets(self):
        return tuple(sorted({max(1, self.horizon // 2), self.horizon}))


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    seeds: tuple = (11, 29, 47)
    arms: tuple = ARMS
    epochs: int = 20
    batch_size: int = 128
    patience: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    jepa_weight: float = 0.1
    risk_weight: float = 0.1
    balance_weight: float = 0.01
    device: str = "cuda"
    amp: bool = False
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    embargo: int = 5
    max_seconds: int = 14400
    bootstrap_reps: int = 2000
    block_dates: int = 20
    cost_bps: float = 10.0
    max_weight: float = 0.25
    risk_aversion: float = 3.0
    xgb_rounds: int = 300
    protected_start: str = "2025-01-01T00:00:00+00:00"

    def __post_init__(self):
        if not isinstance(self.model, ModelConfig):
            raise ValueError("model must be ModelConfig")
        for key in (
            "epochs",
            "batch_size",
            "patience",
            "max_seconds",
            "bootstrap_reps",
            "block_dates",
            "xgb_rounds",
        ):
            positive_integer(key, getattr(self, key))
        if type(self.embargo) is not int or self.embargo < 0:
            raise ValueError("embargo must be nonnegative integer")
        if self.batch_size < 2 or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("batch >=2 and unique seeds required")
        if any(type(x) is not int or not 0 <= x < 2**31 for x in self.seeds):
            raise ValueError("invalid seed")
        if not self.arms or len(set(self.arms)) != len(self.arms) or set(self.arms) - set(ARMS):
            raise ValueError("arms must be unique registered names")
        for key in (
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "jepa_weight",
            "risk_weight",
            "balance_weight",
            "train_fraction",
            "validation_fraction",
            "cost_bps",
            "max_weight",
            "risk_aversion",
        ):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if min(self.learning_rate, self.gradient_clip, self.risk_aversion) <= 0:
            raise ValueError("optimizer and risk scales must be positive")
        if min(self.weight_decay, self.jepa_weight, self.risk_weight, self.balance_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("invalid chronological split")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("evaluation tail must be nonempty")
        if not 0 <= self.cost_bps < 10000 or not 0 < self.max_weight <= 1:
            raise ValueError("invalid transaction cost / allocation cap")
        if self.device not in ("cpu", "cuda") or type(self.amp) is not bool:
            raise ValueError("device must be cpu/cuda, amp boolean")
        if self.amp and self.device != "cuda":
            raise ValueError("AMP is CUDA only")
        from .runtime import timestamp

        if timestamp(self.protected_start) > timestamp("2025-01-01T00:00:00Z"):
            raise ValueError("protected boundary cannot move beyond 2025-01-01 UTC")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        obj = dict(obj)
        obj["model"] = ModelConfig(**obj.get("model", {}))
        return cls(**obj)
