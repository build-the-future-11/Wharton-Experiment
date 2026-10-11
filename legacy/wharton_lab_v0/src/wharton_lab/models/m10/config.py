"""M10 configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M10Config:
    n_assets: int = 5
    path_steps: int = 5
    n_ode_steps: int = 10
    context_dim: int = 8
    training_tau_samples: int = 4
    student_t_df: float | None = None
    noise_scale: float = 0.02
    ridge_alpha: float = 1e-2
    random_state: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
