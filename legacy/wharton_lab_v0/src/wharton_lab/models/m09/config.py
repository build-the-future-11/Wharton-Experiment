"""M09 configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M09Config:
    fast_dim: int = 2
    slow_window: int = 20
    fusion_hidden: int = 8
    groupdro_steps: int = 5
    groupdro_eta: float = 0.5
    ridge_alpha: float = 1.0
    vol_window: int = 20
    vol_threshold: float = 0.02
    random_state: int = 42

    def to_dict(self) -> dict:
        return asdict(self)
