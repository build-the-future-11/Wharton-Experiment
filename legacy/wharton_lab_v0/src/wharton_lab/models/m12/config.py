"""M12 configuration and ablation flags."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M12Config:
    n_assets: int = 5
    memory_slots: int = 8
    latent_dim: int = 4
    graph_k: int = 3
    joint_training: bool = False
    no_graph: bool = False
    no_memory: bool = False
    no_latent: bool = False
    no_liability: bool = False
    random_state: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
