"""M06 configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M06Config:
    seq_len: int = 8
    input_dim: int = 4
    latent_dim: int = 16
    hidden_dim: int = 32
    horizons: tuple[int, ...] = (1, 5, 20)
    ema_momentum: float = 0.99
    vicreg_coeff: float = 0.1
    aux_coeff: float = 0.5
    lr: float = 1e-2
    epochs: int = 3
    batch_size: int = 16
    holdout_future_frac: float = 0.2
    random_state: int = 42
    device: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> M06Config:
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})
