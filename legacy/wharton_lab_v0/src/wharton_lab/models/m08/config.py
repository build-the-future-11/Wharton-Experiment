"""M08 configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M08Config:
    n_nodes: int = 6
    seq_len: int = 10
    node_dim: int = 3
    hidden_dim: int = 24
    latent_dim: int = 16
    graph_window: int = 20
    shrinkage: float = 0.2
    message_passes: int = 2
    lr: float = 1e-2
    epochs: int = 4
    random_state: int = 42
    device: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> M08Config:
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})
