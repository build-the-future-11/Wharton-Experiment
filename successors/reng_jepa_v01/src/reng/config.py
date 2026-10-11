"""Validated model and study configuration matching manuscript Section 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 18
    context_length: int = 128
    max_assets: int = 40
    encoder_hidden: int = 64
    latent_dim: int = 32
    reservoir_dim: int = 32
    reservoir_leaks: tuple[float, float, float] = (0.05, 0.20, 0.80)
    reservoir_norm: float = 0.9
    spectral_bins: int = 32
    spectral_bands: tuple[tuple[int, int], ...] = ((1, 4), (5, 16), (17, 32))
    graph_lookback: int = 63
    graph_min_pairs: int = 50
    graph_threshold: float = 0.2
    graph_scale: float = 0.8
    horizon_dim: int = 8
    horizons: tuple[int, ...] = (5, 21, 63)
    interpreters: int = 3
    ensemble_members: int = 3
    common_factors: int = 4
    memory_value_dim: int = 35
    memory_max_records: int = 1024
    memory_min_records: int = 8
    memory_stride: int = 5
    dropout: float = 0.1
    covariance_jitter: tuple[float, ...] = (1e-6, 1e-5, 1e-4)
    use_memory: bool = True
    use_spectral_kernel: bool = True
    use_fourier_features: bool = True
    use_reservoir: bool = True
    use_jepa_objective: bool = True
    stochastic_latent: bool = True
    uniform_gate: bool = False
    use_cross_asset_factors: bool = True

    def __post_init__(self) -> None:
        if self.input_channels != 18:
            raise ValueError("The manuscript contract requires exactly 18 input channels")
        if self.interpreters != 3 or len(self.reservoir_leaks) != 3:
            raise ValueError("RENG-JEPA requires exactly three interpretation branches")
        if self.latent_dim != 32 or self.reservoir_dim != 32:
            raise ValueError("The v0.1 coordinate contract uses 32-dimensional states")
        if self.spectral_bins != 32 or self.spectral_bands != ((1, 4), (5, 16), (17, 32)):
            raise ValueError("The v0.1 spectral contract is fixed at bins 1-32 and three bands")
        if not 0 < self.reservoir_norm <= 0.9:
            raise ValueError("Reservoir largest singular value must not exceed 0.9")
        if any(not 0 < leak <= 1 for leak in self.reservoir_leaks):
            raise ValueError("Reservoir leaks must be in (0, 1]")
        if self.memory_value_dim != self.latent_dim + 3:
            raise ValueError("Memory values contain latent target plus return/volatility/drawdown")
        if sorted(self.horizons) != list(self.horizons) or len(set(self.horizons)) != len(
            self.horizons
        ):
            raise ValueError("Horizons must be unique and increasing")

    @property
    def base_context_dim(self) -> int:
        return self.latent_dim * 5 + self.horizon_dim

    @property
    def interpreter_dim(self) -> int:
        return self.base_context_dim + self.latent_dim

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyConfig:
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    root_seeds: tuple[int, ...] = (11, 23, 47, 89, 131)
    primary_horizon: int = 21
    primary_training_origins: int = 504
    training_origin_sensitivities: tuple[int, ...] = (252, 504, 1008)
    primary_block_length: int = 63
    block_sensitivities: tuple[int, ...] = (21, 126)
    bootstrap_repetitions: int = 5000
    practical_effect: float = 0.02
    development_years: tuple[int, ...] = (2018, 2019, 2020, 2021)
    audit_years: tuple[int, ...] = (2022, 2023, 2024, 2025)
    max_search_per_family: int = 8
    per_fit_wall_hours: float = 8.0
    learning_rates: tuple[float, ...] = (1e-4, 3e-4)
    weight_decays: tuple[float, ...] = (1e-5, 1e-4)
    dropouts: tuple[float, ...] = (0.0, 0.1)
    scenario_count: int = 1024
    empirical_scenario_count: int = 1024
    calibration_temperatures: tuple[float, ...] = (0.5, 1.0, 2.0)
    calibration_multipliers: tuple[float, ...] = (0.75, 1.0, 1.5, 2.0, 3.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primary_horizon != 21 or self.primary_training_origins != 504:
            raise ValueError("Primary manuscript experiment is fixed at h=21 and n=504")
        if self.root_seeds != (11, 23, 47, 89, 131):
            raise ValueError("Root seed policy differs from the manuscript")
        if self.primary_block_length != 63 or self.bootstrap_repetitions != 5000:
            raise ValueError("Primary stationary-bootstrap contract differs from the manuscript")
        if not all(0 < q < 1 for q in self.quantiles):
            raise ValueError("Quantiles must lie in (0, 1)")
        if len(self.learning_rates) * len(self.weight_decays) * len(self.dropouts) != 8:
            raise ValueError("The declared development grid must have exactly eight cells")
