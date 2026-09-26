"""M11 configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M11Config:
    horizon: int = 5
    cvar_alpha: float = 0.95
    transaction_cost_bps: float = 5.0
    max_weight: float = 0.4
    min_cash_buffer: float = 0.0
    solver: str = "slsqp"

    def to_dict(self) -> dict:
        return asdict(self)
