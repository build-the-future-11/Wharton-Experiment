"""M07 configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M07Config:
    windows: tuple[int, ...] = (20, 60)
    n_assets: int = 5
    latent_dim: int = 16
    hidden_dim: int = 32
    lr: float = 1e-2
    epochs: int = 5
    eigengap_threshold: float = 0.05
    random_state: int = 42
    device: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> M07Config:
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})
